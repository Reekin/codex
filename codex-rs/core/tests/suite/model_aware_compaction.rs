use anyhow::Result;
use codex_core::CodexThread;
use codex_core::TurnInputRequest;
use codex_core::compact::SUMMARIZATION_PROMPT;
use codex_features::Feature;
use codex_login::CodexAuth;
use codex_model_provider_info::ModelProviderInfo;
use codex_model_provider_info::built_in_model_providers;
use codex_models_manager::bundled_models_response;
use codex_protocol::openai_models::ModelInfo;
use codex_protocol::openai_models::ModelsResponse;
use codex_protocol::protocol::EventMsg;
use codex_protocol::protocol::Op;
use codex_protocol::protocol::ThreadSettingsOverrides;
use codex_protocol::user_input::UserInput;
use core_test_support::responses::ResponsesRequest;
use core_test_support::responses::ev_assistant_message;
use core_test_support::responses::ev_completed;
use core_test_support::responses::ev_completed_with_tokens;
use core_test_support::responses::mount_models_once;
use core_test_support::responses::mount_sse_sequence;
use core_test_support::responses::sse;
use core_test_support::skip_if_no_network;
use core_test_support::submit_thread_settings;
use core_test_support::test_codex::test_codex;
use pretty_assertions::assert_eq;
use serde_json::json;
use wiremock::MockServer;

const REMOTE_ENABLED_MODEL: &str = "model-remote-enabled";
const REMOTE_DISABLED_MODEL: &str = "model-remote-disabled";
const REMOTE_ENABLED_COMP_HASH: &str = "model-aware-compaction-enabled";
const REMOTE_DISABLED_COMP_HASH: &str = "model-aware-compaction-disabled";

fn test_model(slug: &str) -> ModelInfo {
    let mut model = bundled_models_response()
        .expect("bundled models.json should parse")
        .models
        .into_iter()
        .find(|model| model.slug == "gpt-5.4")
        .expect("bundled catalog should include gpt-5.4");
    model.slug = slug.to_string();
    model.display_name = slug.to_string();
    model.context_window = Some(273_000);
    model.max_context_window = Some(273_000);
    model.comp_hash = Some(REMOTE_ENABLED_COMP_HASH.to_string());
    model.supports_remote_compaction = true;
    model
}

fn remote_disabled_model(comp_hash: &str) -> ModelInfo {
    let mut model = test_model(REMOTE_DISABLED_MODEL);
    model.comp_hash = Some(comp_hash.to_string());
    model.supports_remote_compaction = false;
    model
}

fn model_catalog() -> ModelsResponse {
    ModelsResponse {
        models: vec![
            test_model(REMOTE_ENABLED_MODEL),
            remote_disabled_model(REMOTE_ENABLED_COMP_HASH),
        ],
    }
}

fn model_catalog_with_comp_hash_change() -> ModelsResponse {
    ModelsResponse {
        models: vec![
            test_model(REMOTE_ENABLED_MODEL),
            remote_disabled_model(REMOTE_DISABLED_COMP_HASH),
        ],
    }
}

fn remote_capable_provider(server: &MockServer) -> ModelProviderInfo {
    let mut provider = built_in_model_providers(/*openai_base_url*/ None)["openai"].clone();
    provider.base_url = Some(format!("{}/v1", server.uri()));
    provider.supports_websockets = false;
    provider
}

fn remote_unsupported_provider(server: &MockServer) -> ModelProviderInfo {
    let mut provider = remote_capable_provider(server);
    provider.name = "Custom model-aware compaction provider".to_string();
    provider
}

fn user_turn(text: &str) -> TurnInputRequest {
    TurnInputRequest::user_input(vec![UserInput::Text {
        text: text.to_string(),
        text_elements: Vec::new(),
    }])
}

async fn submit_user_turn(codex: &CodexThread, text: &str) -> Result<()> {
    codex.start_or_steer_turn(user_turn(text)).await?;
    wait_for_turn_complete(codex).await;
    Ok(())
}

async fn wait_for_turn_complete(codex: &CodexThread) {
    loop {
        let event = codex
            .next_event()
            .await
            .expect("event stream ended before turn completion");
        match event.msg {
            EventMsg::TurnComplete(_) => return,
            EventMsg::Error(error) => panic!("unexpected turn error: {}", error.message),
            _ => {}
        }
    }
}

fn remote_v2_response(response_id: &str) -> String {
    sse(vec![
        json!({
            "type": "response.output_item.done",
            "item": {
                "type": "compaction",
                "encrypted_content": format!("summary-{response_id}"),
            }
        }),
        ev_completed(response_id),
    ])
}

fn local_compaction_response(response_id: &str) -> String {
    sse(vec![
        ev_assistant_message(
            &format!("message-{response_id}"),
            &format!("local-summary-{response_id}"),
        ),
        ev_completed_with_tokens(response_id, /*total_tokens*/ 10),
    ])
}

fn assert_remote_v2_request(request: &ResponsesRequest, model: &str) {
    assert_eq!(request.path(), "/v1/responses");
    assert_eq!(request.body_json()["model"].as_str(), Some(model));
    let input = request.input();
    assert_eq!(
        input.last(),
        Some(&json!({
            "type": "compaction_trigger",
        }))
    );
    assert!(!request.body_contains_text(SUMMARIZATION_PROMPT));
}

fn assert_local_compaction_request(request: &ResponsesRequest, model: &str) {
    assert_eq!(request.path(), "/v1/responses");
    assert_eq!(request.body_json()["model"].as_str(), Some(model));
    assert!(request.body_contains_text(SUMMARIZATION_PROMPT));
    assert!(
        request
            .input()
            .iter()
            .all(|item| item["type"] != "compaction_trigger")
    );
}

fn assert_local_compaction_follow_up(request: &ResponsesRequest, model: &str, summary: &str) {
    assert_eq!(request.path(), "/v1/responses");
    assert_eq!(request.body_json()["model"].as_str(), Some(model));
    assert!(request.body_contains_text(summary));
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn manual_compaction_switches_from_remote_v2_to_local_with_the_active_model() -> Result<()> {
    skip_if_no_network!(Ok(()));

    let server = MockServer::start().await;
    let models_mock = mount_models_once(&server, model_catalog()).await;
    let responses = mount_sse_sequence(
        &server,
        vec![
            sse(vec![
                ev_assistant_message("initial-message", "initial-reply"),
                ev_completed_with_tokens("initial-response", /*total_tokens*/ 80),
            ]),
            remote_v2_response("manual-remote"),
            local_compaction_response("manual-local"),
            sse(vec![
                ev_assistant_message("follow-up-message", "follow-up-reply"),
                ev_completed_with_tokens("follow-up-response", /*total_tokens*/ 20),
            ]),
        ],
    )
    .await;
    let provider = remote_capable_provider(&server);
    let test = test_codex()
        .with_auth(CodexAuth::create_dummy_chatgpt_auth_for_testing())
        .with_model(REMOTE_ENABLED_MODEL)
        .with_config(move |config| {
            config.model_provider = provider;
            config
                .features
                .enable(Feature::RemoteCompactionV2)
                .expect("remote compaction v2 should be configurable");
        })
        .build_with_auto_env(&server)
        .await?;

    submit_user_turn(&test.codex, "seed remote compaction history").await?;
    test.codex.submit(Op::Compact).await?;
    wait_for_turn_complete(&test.codex).await;

    submit_thread_settings(
        &test.codex,
        ThreadSettingsOverrides {
            model: Some(REMOTE_DISABLED_MODEL.to_string()),
            ..Default::default()
        },
    )
    .await?;
    test.codex.submit(Op::Compact).await?;
    wait_for_turn_complete(&test.codex).await;
    submit_user_turn(&test.codex, "after local manual compaction").await?;

    assert_eq!(models_mock.requests().len(), 1);
    let requests = responses.requests();
    assert_eq!(requests.len(), 4);
    assert_remote_v2_request(&requests[1], REMOTE_ENABLED_MODEL);
    assert_local_compaction_request(&requests[2], REMOTE_DISABLED_MODEL);
    assert_local_compaction_follow_up(
        &requests[3],
        REMOTE_DISABLED_MODEL,
        "local-summary-manual-local",
    );

    Ok(())
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn automatic_compaction_switches_from_remote_v2_to_local_with_the_active_model() -> Result<()>
{
    skip_if_no_network!(Ok(()));

    let server = MockServer::start().await;
    let models_mock = mount_models_once(&server, model_catalog()).await;
    let responses = mount_sse_sequence(
        &server,
        vec![
            sse(vec![
                ev_assistant_message("first-message", "first-reply"),
                ev_completed_with_tokens("first-response", /*total_tokens*/ 150),
            ]),
            remote_v2_response("auto-remote"),
            sse(vec![
                ev_assistant_message("second-message", "second-reply"),
                ev_completed_with_tokens("second-response", /*total_tokens*/ 150),
            ]),
            local_compaction_response("auto-local"),
            sse(vec![
                ev_assistant_message("third-message", "third-reply"),
                ev_completed_with_tokens("third-response", /*total_tokens*/ 20),
            ]),
        ],
    )
    .await;
    let provider = remote_capable_provider(&server);
    let test = test_codex()
        .with_auth(CodexAuth::create_dummy_chatgpt_auth_for_testing())
        .with_model(REMOTE_ENABLED_MODEL)
        .with_config(move |config| {
            config.model_provider = provider;
            config.model_auto_compact_token_limit = Some(100);
            config
                .features
                .enable(Feature::RemoteCompactionV2)
                .expect("remote compaction v2 should be configurable");
        })
        .build_with_auto_env(&server)
        .await?;

    submit_user_turn(&test.codex, "first turn").await?;
    submit_user_turn(&test.codex, "trigger remote automatic compaction").await?;
    submit_thread_settings(
        &test.codex,
        ThreadSettingsOverrides {
            model: Some(REMOTE_DISABLED_MODEL.to_string()),
            ..Default::default()
        },
    )
    .await?;
    submit_user_turn(&test.codex, "trigger local automatic compaction").await?;

    assert_eq!(models_mock.requests().len(), 1);
    let requests = responses.requests();
    assert_eq!(requests.len(), 5);
    assert_remote_v2_request(&requests[1], REMOTE_ENABLED_MODEL);
    assert_local_compaction_request(&requests[3], REMOTE_DISABLED_MODEL);
    assert_local_compaction_follow_up(
        &requests[4],
        REMOTE_DISABLED_MODEL,
        "local-summary-auto-local",
    );

    Ok(())
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn model_switch_compaction_uses_the_new_models_capability_for_api_key_auth() -> Result<()> {
    skip_if_no_network!(Ok(()));

    let server = MockServer::start().await;
    let responses = mount_sse_sequence(
        &server,
        vec![
            sse(vec![
                ev_assistant_message("initial-message", "initial-reply"),
                ev_completed_with_tokens("initial-response", /*total_tokens*/ 80),
            ]),
            local_compaction_response("switch-local"),
            sse(vec![
                ev_assistant_message("switched-message", "switched-reply"),
                ev_completed_with_tokens("switched-response", /*total_tokens*/ 20),
            ]),
        ],
    )
    .await;
    let provider = remote_capable_provider(&server);
    let catalog = model_catalog_with_comp_hash_change();
    let test = test_codex()
        .with_auth(CodexAuth::from_api_key("test-api-key"))
        .with_model(REMOTE_ENABLED_MODEL)
        .with_config(move |config| {
            config.model_provider = provider;
            config.model_catalog = Some(catalog);
            config
                .features
                .enable(Feature::RemoteCompactionV2)
                .expect("remote compaction v2 should be configurable");
        })
        .build_with_auto_env(&server)
        .await?;

    submit_user_turn(&test.codex, "before model switch").await?;
    submit_thread_settings(
        &test.codex,
        ThreadSettingsOverrides {
            model: Some(REMOTE_DISABLED_MODEL.to_string()),
            ..Default::default()
        },
    )
    .await?;
    submit_user_turn(&test.codex, "after model switch").await?;

    let requests = responses.requests();
    assert_eq!(requests.len(), 3);
    assert_local_compaction_request(&requests[1], REMOTE_DISABLED_MODEL);
    assert_local_compaction_follow_up(
        &requests[2],
        REMOTE_DISABLED_MODEL,
        "local-summary-switch-local",
    );

    Ok(())
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn provider_without_remote_compaction_keeps_enabled_model_on_local_path() -> Result<()> {
    skip_if_no_network!(Ok(()));

    let server = MockServer::start().await;
    let responses = mount_sse_sequence(
        &server,
        vec![
            sse(vec![
                ev_assistant_message("initial-message", "initial-reply"),
                ev_completed_with_tokens("initial-response", /*total_tokens*/ 80),
            ]),
            local_compaction_response("provider-local"),
        ],
    )
    .await;
    let provider = remote_unsupported_provider(&server);
    let catalog = ModelsResponse {
        models: vec![test_model(REMOTE_ENABLED_MODEL)],
    };
    let test = test_codex()
        .with_auth(CodexAuth::create_dummy_chatgpt_auth_for_testing())
        .with_model(REMOTE_ENABLED_MODEL)
        .with_config(move |config| {
            config.model_provider = provider;
            config.model_catalog = Some(catalog);
            config
                .features
                .enable(Feature::RemoteCompactionV2)
                .expect("remote compaction v2 should be configurable");
        })
        .build_with_auto_env(&server)
        .await?;

    submit_user_turn(&test.codex, "seed local compaction history").await?;
    test.codex.submit(Op::Compact).await?;
    wait_for_turn_complete(&test.codex).await;

    let requests = responses.requests();
    assert_eq!(requests.len(), 2);
    assert_local_compaction_request(&requests[1], REMOTE_ENABLED_MODEL);

    Ok(())
}
