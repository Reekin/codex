use super::compact::local_support::*;
use anyhow::Result;
use codex_features::Feature;
use codex_login::CodexAuth;
use codex_models_manager::bundled_models_response;
use codex_protocol::openai_models::ModelInfo;
use codex_protocol::openai_models::ModelsResponse;
use codex_protocol::protocol::Op;
use codex_protocol::protocol::ThreadSettingsOverrides;
use core_test_support::responses::ev_completed;
use core_test_support::responses::sse;
use core_test_support::responses::start_mock_server;
use core_test_support::skip_if_no_network;
use core_test_support::submit_thread_settings;
use core_test_support::test_codex::test_codex;
use pretty_assertions::assert_eq;
use serde_json::json;

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn mid_turn_cleanup_uses_activated_model_settings_and_keeps_pending_work() -> Result<()> {
    use codex_protocol::protocol::EventMsg;
    use codex_protocol::protocol::TurnSettingsUpdate;
    use codex_protocol::protocol::TurnSettingsUpdateOutcome;
    use codex_protocol::request_user_input::RequestUserInputAnswer;
    use codex_protocol::request_user_input::RequestUserInputResponse;
    use core_test_support::responses::ev_function_call;
    use core_test_support::wait_for_event_match;
    use std::collections::HashMap;

    skip_if_no_network!(Ok(()));
    let server = start_mock_server().await;
    let responder = LocalModel::mount(&server).await;
    let test = test_codex()
        .with_auth(CodexAuth::from_api_key("test-key"))
        .with_model("remote-model")
        .with_config(|config| {
            configure(config);
            config.local_compaction.force_local = false;
            config.local_compaction.trigger_percent = 20;
            config.local_compaction.target_percent = 15;
            let mut next = model("local-model", false, "shared");
            next.default_reasoning_summary =
                codex_protocol::config_types::ReasoningSummary::Detailed;
            config.model_catalog = Some(ModelsResponse {
                models: vec![model("remote-model", true, "shared"), next],
            });
            config.model_reasoning_summary = None;
            for feature in [
                Feature::RemoteCompactionV2,
                Feature::StepModelSwitching,
                Feature::DefaultModeRequestUserInput,
            ] {
                let _ = config.features.enable(feature);
            }
        })
        .build_with_auto_env(&server)
        .await?;
    responder.text(&"Earlier evidence needs verification. ".repeat(5000));
    test.submit_turn("Keep the offline-only constraint.")
        .await?;
    responder.reply(sse(vec![ev_function_call("pause", "request_user_input", &json!({"questions":[{
        "id":"continue","header":"Continue","question":"Continue after changing model?",
        "options":[{"label":"Yes (Recommended)","description":"Continue."},{"label":"No","description":"Stop."}]
    }]}).to_string()), ev_completed("pause-response")]));
    test.codex
        .start_or_steer_turn(codex_core::TurnInputRequest::user_input(vec![
            codex_protocol::user_input::UserInput::Text {
                text: "Switch model while preserving pending work.".to_string(),
                text_elements: Vec::new(),
            },
        ]))
        .await?;
    let request = wait_for_event_match(&test.codex, |event| match event {
        EventMsg::RequestUserInput(request) => Some(request.clone()),
        _ => None,
    })
    .await;
    let (reply, outcome) = tokio::sync::oneshot::channel();
    test.codex
        .submit(Op::TurnSettings {
            turn_id: request.turn_id.clone(),
            update: TurnSettingsUpdate {
                model: Some("local-model".to_string()),
                ..Default::default()
            },
            reply,
        })
        .await?;
    assert_eq!(
        tokio::time::timeout(std::time::Duration::from_secs(10), outcome).await??,
        TurnSettingsUpdateOutcome::Applied
    );
    responder.reply(sse(vec![
        ev_function_call("pending-work", "unsupported_tool", "{}"),
        ev_completed("pending-response"),
    ]));
    responder.text("Continue with current settings.");
    test.codex
        .submit(Op::UserInputAnswer {
            id: request.turn_id,
            response: RequestUserInputResponse {
                answers: HashMap::from([(
                    "continue".to_string(),
                    RequestUserInputAnswer {
                        answers: vec!["Yes (Recommended)".to_string()],
                    },
                )]),
            },
        })
        .await?;
    complete(&test.codex).await;
    let bodies = responder.bodies();
    let analysis = bodies
        .iter()
        .find(|body| analysis_payload(body, SUMMARIZE).is_some())
        .expect("mid-turn tier analysis");
    assert_eq!(analysis["model"], "local-model");
    assert_eq!(analysis["reasoning"]["summary"], "detailed");
    assert!(
        analysis
            .to_string()
            .contains("Switch model while preserving pending work.")
    );
    let final_request = bodies.last().unwrap();
    assert_eq!(final_request["model"], "local-model");
    assert_eq!(final_request["reasoning"]["summary"], "detailed");
    assert!(final_request.to_string().contains(LEDGER));
    assert!(
        final_request["input"]
            .as_array()
            .unwrap()
            .iter()
            .any(|item| item["type"] == "function_call_output"
                && item["call_id"] == "pending-work"
                && item["output"].as_str().unwrap().contains("unsupported"))
    );
    Ok(())
}

fn model(slug: &str, remote: bool, hash: &str) -> ModelInfo {
    let mut model = bundled_models_response()
        .unwrap()
        .models
        .into_iter()
        .find(|model| model.slug == "gpt-5.4")
        .unwrap();
    model.slug = slug.to_string();
    model.display_name = slug.to_string();
    model.supports_remote_compaction = remote;
    model.comp_hash = Some(hash.to_string());
    model
}

fn remote_response() -> String {
    sse(vec![
        json!({"type":"response.output_item.done","item":{"type":"compaction","encrypted_content":"remote-original-protocol"}}),
        ev_completed("remote-response"),
    ])
}

#[test_case::test_case(true, true, false; "native remote default")]
#[test_case::test_case(true, true, true; "force local on capable model")]
#[test_case::test_case(true, false, false; "model lacks remote capability")]
#[test_case::test_case(false, true, false; "provider lacks remote capability")]
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn provider_model_capability_and_local_preference_select_independent_routes(
    provider_remote: bool,
    model_remote: bool,
    force_local: bool,
) -> Result<()> {
    skip_if_no_network!(Ok(()));
    let server = start_mock_server().await;
    let responder = LocalModel::mount(&server).await;
    let use_remote = provider_remote && model_remote && !force_local;
    let test = test_codex()
        .with_auth(CodexAuth::from_api_key("test-key"))
        .with_model("routing-model")
        .with_config(move |config| {
            configure(config);
            config.local_compaction.force_local = force_local;
            config.model_catalog = Some(ModelsResponse {
                models: vec![model("routing-model", model_remote, "same")],
            });
            if !provider_remote {
                config.model_provider.name = "Custom provider".to_string();
            }
            let _ = config.features.enable(Feature::RemoteCompactionV2);
            if force_local {
                let _ = config.features.enable(Feature::TokenBudget);
            }
        })
        .build_with_auto_env(&server)
        .await?;
    responder.reply(tool_turn());
    responder.text("Original dialogue.");
    test.submit_turn("Seed route selection.").await?;
    responder.text("Recent dialogue.");
    test.submit_turn("Recent direction.").await?;
    if use_remote {
        responder.reply(remote_response());
    }
    test.codex.submit(Op::Compact).await?;
    complete(&test.codex).await;
    responder.text("Continue selected route.");
    test.submit_turn("Read selected view.").await?;
    let bodies = responder.bodies();
    let requests = &bodies[3..bodies.len() - 1];
    assert_eq!(
        bodies[0]["tools"].to_string().contains("recall_read_item"),
        !use_remote
    );
    if !use_remote {
        assert!(
            !bodies[0]["tools"]
                .to_string()
                .contains("new_context_window")
        );
    }
    assert_eq!(requests.len(), 1);
    if use_remote {
        assert_eq!(
            requests[0]["input"].as_array().unwrap().last(),
            Some(&json!({"type":"compaction_trigger"}))
        );
        assert!(analysis_payload(&requests[0], CLASSIFY).is_none());
        assert!(
            bodies
                .last()
                .unwrap()
                .to_string()
                .contains("remote-original-protocol")
        );
    } else {
        assert!(analysis_payload(&requests[0], CLASSIFY).is_some());
        assert!(!requests[0].to_string().contains("compaction_trigger"));
        assert!(bodies.last().unwrap().to_string().contains(SHORTENED));
        assert!(
            !bodies
                .last()
                .unwrap()
                .to_string()
                .contains("remote-original-protocol")
        );
    }
    assert!(
        responder
            .requests()
            .iter()
            .all(|request| request.url.path() == "/v1/responses")
    );
    Ok(())
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn model_switch_uses_active_capability_and_preserves_selected_reasoning() -> Result<()> {
    skip_if_no_network!(Ok(()));
    let server = start_mock_server().await;
    let responder = LocalModel::mount(&server).await;
    let test = test_codex()
        .with_auth(CodexAuth::from_api_key("test-key"))
        .with_model("remote-model")
        .with_config(|config| {
            configure(config);
            config.local_compaction.force_local = false;
            let mut next = model("local-model", false, "local-hash");
            next.default_reasoning_summary =
                codex_protocol::config_types::ReasoningSummary::Detailed;
            config.model_catalog = Some(ModelsResponse {
                models: vec![model("remote-model", true, "remote-hash"), next],
            });
            config.model_reasoning_summary = None;
            let _ = config.features.enable(Feature::RemoteCompactionV2);
        })
        .build_with_auto_env(&server)
        .await?;
    responder.reply(tool_turn());
    responder.text("Before model switch.");
    test.submit_turn("Seed prior-model tool results.").await?;
    responder.text("Completed tool results are available for reassessment.");
    test.submit_turn("Continue after collecting the evidence.")
        .await?;
    submit_thread_settings(
        &test.codex,
        ThreadSettingsOverrides {
            model: Some("local-model".to_string()),
            ..Default::default()
        },
    )
    .await?;
    responder.text("After model switch.");
    test.submit_turn("Switch capability and continue.").await?;
    let bodies = responder.bodies();
    let classifier = bodies
        .iter()
        .find(|body| analysis_payload(body, CLASSIFY).is_some())
        .expect("model-switch cleanup");
    assert_eq!(classifier["model"], "local-model");
    assert_eq!(classifier["reasoning"]["summary"], "detailed");
    assert!(!classifier.to_string().contains("compaction_trigger"));
    assert_eq!(bodies.last().unwrap()["model"], "local-model");
    assert_eq!(bodies.last().unwrap()["reasoning"]["summary"], "detailed");
    assert!(bodies.last().unwrap().to_string().contains(SHORTENED));
    assert!(
        bodies
            .last()
            .unwrap()
            .to_string()
            .contains("Switch capability and continue.")
    );
    Ok(())
}

#[test_case::test_case(false; "missing previous hash")]
#[test_case::test_case(true; "same hash")]
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn compatible_model_switch_does_not_request_unnecessary_cleanup(
    same_hash: bool,
) -> Result<()> {
    skip_if_no_network!(Ok(()));
    let server = start_mock_server().await;
    let responder = LocalModel::mount(&server).await;
    let test = test_codex()
        .with_auth(CodexAuth::from_api_key("test-key"))
        .with_model("previous-model")
        .with_config(move |config| {
            configure(config);
            let mut previous = model("previous-model", false, "shared");
            if !same_hash {
                previous.comp_hash = None;
            }
            config.model_catalog = Some(ModelsResponse {
                models: vec![previous, model("next-model", false, "shared")],
            });
        })
        .build_with_auto_env(&server)
        .await?;
    responder.text("Initial answer.");
    test.submit_turn("Initial turn.").await?;
    submit_thread_settings(
        &test.codex,
        ThreadSettingsOverrides {
            model: Some("next-model".to_string()),
            ..Default::default()
        },
    )
    .await?;
    responder.text("Switched answer.");
    test.submit_turn("Next turn.").await?;
    let bodies = responder.bodies();
    assert_eq!(bodies.len(), 2);
    assert_eq!(bodies[1]["model"], "next-model");
    assert!(bodies[1].to_string().contains("Initial answer."));
    assert!(
        bodies
            .iter()
            .all(|body| analysis_payload(body, CLASSIFY).is_none()
                && analysis_payload(body, SUMMARIZE).is_none())
    );
    Ok(())
}
