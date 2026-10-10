//! Verifies that root service-tier changes reach existing child turns and future work.

use anyhow::Result;
use codex_core::TurnInputRequest;
use codex_core::config::AgentRoleConfig;
use codex_core::config::Config;
use codex_features::Feature;
use codex_protocol::ThreadId;
use codex_protocol::config_types::SERVICE_TIER_DEFAULT_REQUEST_VALUE;
use codex_protocol::protocol::EventMsg;
use codex_protocol::protocol::Op;
use codex_protocol::protocol::ThreadSettingsOverrides;
use codex_protocol::user_input::UserInput;
use core_test_support::responses::ev_assistant_message;
use core_test_support::responses::ev_completed;
use core_test_support::responses::ev_function_call_with_namespace;
use core_test_support::responses::ev_response_created;
use core_test_support::responses::mount_sse_once_match;
use core_test_support::responses::sse;
use core_test_support::responses::sse_response;
use core_test_support::responses::start_mock_server;
use core_test_support::submit_thread_settings;
use core_test_support::test_codex::test_codex;
use core_test_support::wait_for_event;
use core_test_support::wait_for_event_match;
use pretty_assertions::assert_eq;
use serde_json::Value;
use serde_json::json;
use std::sync::Arc;
use std::sync::Mutex;
use test_case::test_case;

const ROOT_PROMPT: &str = "spawn the service-tier worker";
const CHILD_PROMPT: &str = "pause the service-tier worker";
const FOLLOWUP_PROMPT: &str = "continue the service-tier worker later";
const FRESH_ROOT_PROMPT: &str = "spawn another service-tier worker";
const FRESH_CHILD_PROMPT: &str = "finish the new service-tier worker";
const SPAWN_CALL_ID: &str = "spawn-service-tier-worker";
const FRESH_SPAWN_CALL_ID: &str = "spawn-fresh-service-tier-worker";
const PAUSE_CALL_ID: &str = "pause-service-tier-worker";
const PRIORITY_ROLE: &str = "priority-worker";

fn request_body(request: &wiremock::Request) -> Value {
    let body = match request
        .headers
        .get("content-encoding")
        .and_then(|encoding| encoding.to_str().ok())
    {
        Some(encoding) if encoding.eq_ignore_ascii_case("zstd") => {
            zstd::stream::decode_all(std::io::Cursor::new(&request.body)).ok()
        }
        _ => Some(request.body.clone()),
    };
    serde_json::from_slice(&body.expect("request body should decode"))
        .expect("request body should be JSON")
}

fn body_contains(request: &wiremock::Request, text: &str) -> bool {
    request_body(request).to_string().contains(text)
}

async fn assert_request_service_tier(
    server: &wiremock::MockServer,
    child_id: ThreadId,
    expected_count: usize,
    expected: Option<&str>,
) {
    // Inspect all requests from this child, including attempts after a one-shot mock expired.
    let requests: Vec<_> = server
        .received_requests()
        .await
        .expect("server should record requests")
        .iter()
        .filter(|request| request.method == "POST")
        .map(request_body)
        .filter(|body| body["client_metadata"]["thread_id"] == json!(child_id))
        .collect();
    assert_eq!(
        requests.len(),
        expected_count,
        "child request bodies: {requests:#?}"
    );
    assert_eq!(
        requests
            .last()
            .expect("child should have made a request")
            .get("service_tier")
            .and_then(Value::as_str),
        expected
    );
}

fn input_texts(body: &Value) -> Vec<&str> {
    body["input"]
        .as_array()
        .expect("request should contain input")
        .iter()
        .filter_map(|item| item["content"].as_array())
        .flatten()
        .filter_map(|content| content["text"].as_str())
        .collect()
}

async fn mount_parent_result(
    server: &wiremock::MockServer,
    root_id: ThreadId,
    sender: &str,
    payload: &str,
    acknowledgement: &'static str,
) -> Arc<Mutex<Vec<Value>>> {
    let expected = format!(
        "Message Type: FINAL_ANSWER\nTask name: /root\nSender: {sender}\nPayload:\n{payload}"
    );
    let requests = Arc::new(Mutex::new(Vec::new()));
    let captured = Arc::clone(&requests);
    wiremock::Mock::given(wiremock::matchers::method("POST"))
        .and(wiremock::matchers::path("/v1/responses"))
        .and(move |request: &wiremock::Request| {
            let body = request_body(request);
            if body["client_metadata"]["thread_id"] != json!(root_id)
                || !input_texts(&body).contains(&expected.as_str())
            {
                return false;
            }
            assert_eq!(
                input_texts(&body)
                    .into_iter()
                    .filter(|text| *text == expected)
                    .collect::<Vec<_>>(),
                vec![expected.as_str()],
                "parent should consume this completion once"
            );
            captured.lock().expect("result log should lock").push(body);
            true
        })
        .respond_with(sse_response(sse(vec![
            ev_response_created(acknowledgement),
            ev_assistant_message(acknowledgement, acknowledgement),
            ev_completed(acknowledgement),
        ])))
        .with_priority(/*p*/ 1)
        .up_to_n_times(1)
        .expect(1)
        .mount(server)
        .await;
    requests
}

async fn assert_parent_result(
    root: &codex_core::CodexThread,
    requests: &Arc<Mutex<Vec<Value>>>,
    acknowledgement: &str,
    expected_tier: Option<&str>,
) {
    wait_for_event(root, |event| {
        matches!(event, EventMsg::TurnComplete(completed)
            if completed.last_agent_message.as_deref() == Some(acknowledgement))
    })
    .await;
    let requests = requests.lock().expect("result log should lock");
    assert_eq!(requests.len(), 1);
    assert_eq!(requests[0]["service_tier"].as_str(), expected_tier);
}

fn configure_priority_role(config: &mut Config) {
    for feature in [Feature::Collab, Feature::MultiAgentV2] {
        config
            .features
            .enable(feature)
            .expect("test config should allow feature update");
    }
    config.model_provider.request_max_retries = Some(0);
    config.model_provider.stream_max_retries = Some(0);
    let role_path = config.codex_home.join("priority-worker.toml");
    std::fs::write(&role_path, "service_tier = \"priority\"\n")
        .expect("priority role should be written");
    config.agent_roles.insert(
        PRIORITY_ROLE.to_string(),
        AgentRoleConfig {
            description: Some("Role with a configured priority tier".to_string()),
            config_file: Some(role_path.to_path_buf()),
            nickname_candidates: None,
        },
    );
}

async fn wait_for_turn_complete(thread: &codex_core::CodexThread) {
    wait_for_event(thread, |event| matches!(event, EventMsg::TurnComplete(_))).await;
}

async fn start_root_prompt(root: &codex_core::CodexThread, prompt: &str) -> Result<()> {
    root.start_or_steer_turn(TurnInputRequest::user_input(vec![UserInput::Text {
        text: prompt.to_string(),
        text_elements: Vec::new(),
    }]))
    .await?;
    Ok(())
}

async fn mount_root_collaboration_call(
    server: &wiremock::MockServer,
    root_id: ThreadId,
    prompt: &'static str,
    call_id: &'static str,
    arguments: serde_json::Value,
) {
    let response_id = format!("root-{call_id}");
    mount_sse_once_match(
        server,
        move |request: &wiremock::Request| {
            request_body(request)["client_metadata"]["thread_id"] == json!(root_id)
                && body_contains(request, prompt)
                && !body_contains(request, call_id)
        },
        sse(vec![
            ev_response_created(&response_id),
            ev_function_call_with_namespace(
                call_id,
                "collaboration",
                "spawn_agent",
                &arguments.to_string(),
            ),
            ev_completed(&response_id),
        ]),
    )
    .await;

    let completion_id = format!("root-{call_id}-finished");
    mount_sse_once_match(
        server,
        move |request: &wiremock::Request| {
            request_body(request)["client_metadata"]["thread_id"] == json!(root_id)
                && body_contains(request, prompt)
                && body_contains(request, call_id)
        },
        sse(vec![
            ev_response_created(&completion_id),
            ev_assistant_message(&format!("{completion_id}-message"), "worker started"),
            ev_completed(&completion_id),
        ]),
    )
    .await;
}

async fn mount_completed_child(
    server: &wiremock::MockServer,
    prompt: &'static str,
    root_prompt: &'static str,
    completion: &'static str,
) {
    mount_sse_once_match(
        server,
        move |request: &wiremock::Request| {
            body_contains(request, prompt) && !body_contains(request, root_prompt)
        },
        sse(vec![
            ev_response_created(prompt),
            ev_assistant_message(&format!("{prompt}-message"), completion),
            ev_completed(prompt),
        ]),
    )
    .await;
}

#[test_case(Some("priority"), None; "disabling fast mode updates active and idle child work")]
#[test_case(Some("priority"), Some("default"); "explicit default updates active and idle child work")]
#[test_case(None, Some("priority"); "enabling fast mode updates active and idle child work")]
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn root_service_tier_change_updates_existing_subagent(
    initial_service_tier: Option<&str>,
    updated_service_tier: Option<&str>,
) -> Result<()> {
    let server = start_mock_server().await;
    let initial_service_tier_owned = initial_service_tier.map(str::to_string);
    let updated_request_service_tier = updated_service_tier
        .filter(|service_tier| *service_tier != SERVICE_TIER_DEFAULT_REQUEST_VALUE);
    let mut builder = test_codex()
        .with_model("gpt-5.6-sol")
        .with_config(move |config| {
            config.service_tier = initial_service_tier_owned;
            configure_priority_role(config);
        });
    let test = builder.build_with_auto_env(&server).await?;
    assert!(!test.config.features.enabled(Feature::StepModelSwitching));
    let mut created_threads = test.thread_manager.subscribe_thread_created();

    mount_root_collaboration_call(
        &server,
        test.session_configured.thread_id,
        ROOT_PROMPT,
        SPAWN_CALL_ID,
        json!({
            "message": CHILD_PROMPT,
            "task_name": "worker",
            "agent_type": PRIORITY_ROLE,
            "fork_turns": "none",
        }),
    )
    .await;

    mount_sse_once_match(
        &server,
        |request: &wiremock::Request| {
            body_contains(request, CHILD_PROMPT)
                && !body_contains(request, ROOT_PROMPT)
                && !body_contains(request, PAUSE_CALL_ID)
        },
        sse(vec![
            ev_response_created("child-paused"),
            ev_function_call_with_namespace(
                PAUSE_CALL_ID,
                "collaboration",
                "wait_agent",
                &json!({ "timeout_ms": 30_000 }).to_string(),
            ),
            ev_completed("child-paused"),
        ]),
    )
    .await;
    mount_sse_once_match(
        &server,
        |request: &wiremock::Request| {
            body_contains(request, PAUSE_CALL_ID) && !body_contains(request, ROOT_PROMPT)
        },
        sse(vec![
            ev_response_created("child-continued"),
            ev_assistant_message("child-continued-message", "child continued"),
            ev_completed("child-continued"),
        ]),
    )
    .await;
    test.submit_text_turn(ROOT_PROMPT).await?;
    let child_thread_id = created_threads.recv().await?;
    let child = test.thread_manager.get_thread(child_thread_id).await?;
    let original_child_service_tier = child.config_snapshot().await.service_tier;
    assert_eq!(original_child_service_tier.as_deref(), initial_service_tier);
    wait_for_event_match(child.as_ref(), |event| match event {
        EventMsg::CollabWaitingBegin(request) if request.call_id == PAUSE_CALL_ID => Some(()),
        EventMsg::Error(error) => panic!("child failed before pausing: {}", error.message),
        EventMsg::TurnComplete(completed) => {
            panic!("child completed before pausing: {completed:?}")
        }
        _ => None,
    })
    .await;
    assert_request_service_tier(&server, child_thread_id, 1, initial_service_tier).await;

    submit_thread_settings(
        &test.codex,
        ThreadSettingsOverrides {
            service_tier: Some(updated_service_tier.map(str::to_string)),
            ..Default::default()
        },
    )
    .await?;
    assert_eq!(
        child.config_snapshot().await.service_tier,
        original_child_service_tier,
        "the root routing policy must not rewrite child-owned settings"
    );

    let continued_parent = mount_parent_result(
        &server,
        test.session_configured.thread_id,
        "/root/worker",
        "child continued",
        "root handled continued child",
    )
    .await;
    child
        .start_or_steer_turn(TurnInputRequest::user_input(vec![UserInput::Text {
            text: "continue after the service-tier update".to_string(),
            text_elements: Vec::new(),
        }]))
        .await?;
    wait_for_turn_complete(&child).await;
    assert_request_service_tier(&server, child_thread_id, 2, updated_request_service_tier).await;
    assert_parent_result(
        &test.codex,
        &continued_parent,
        "root handled continued child",
        updated_request_service_tier,
    )
    .await;

    mount_sse_once_match(
        &server,
        |request: &wiremock::Request| body_contains(request, "compaction_trigger"),
        sse(vec![
            ev_response_created("child-remote-compaction"),
            json!({
                "type": "response.output_item.done",
                "item": {
                    "type": "compaction",
                    "encrypted_content": "compacted child history",
                },
            }),
            ev_completed("child-remote-compaction"),
        ]),
    )
    .await;
    let compacted_parent = mount_parent_result(
        &server,
        test.session_configured.thread_id,
        "/root/worker",
        "",
        "root handled compacted child",
    )
    .await;
    child.submit(Op::Compact).await?;
    wait_for_turn_complete(&child).await;
    assert_request_service_tier(&server, child_thread_id, 3, updated_request_service_tier).await;
    assert_parent_result(
        &test.codex,
        &compacted_parent,
        "root handled compacted child",
        updated_request_service_tier,
    )
    .await;

    mount_completed_child(&server, FOLLOWUP_PROMPT, ROOT_PROMPT, "worker completed").await;
    let future_parent = mount_parent_result(
        &server,
        test.session_configured.thread_id,
        "/root/worker",
        "worker completed",
        "root handled future child",
    )
    .await;
    child
        .start_or_steer_turn(TurnInputRequest::user_input(vec![UserInput::Text {
            text: FOLLOWUP_PROMPT.to_string(),
            text_elements: Vec::new(),
        }]))
        .await?;
    wait_for_turn_complete(&child).await;
    assert_request_service_tier(&server, child_thread_id, 4, updated_request_service_tier).await;
    assert_parent_result(
        &test.codex,
        &future_parent,
        "root handled future child",
        updated_request_service_tier,
    )
    .await;

    mount_root_collaboration_call(
        &server,
        test.session_configured.thread_id,
        FRESH_ROOT_PROMPT,
        FRESH_SPAWN_CALL_ID,
        json!({
            "message": FRESH_CHILD_PROMPT,
            "task_name": "fresh",
            "agent_type": PRIORITY_ROLE,
            "fork_turns": "none",
        }),
    )
    .await;
    mount_completed_child(
        &server,
        FRESH_CHILD_PROMPT,
        FRESH_ROOT_PROMPT,
        "worker completed",
    )
    .await;
    let fresh_parent = mount_parent_result(
        &server,
        test.session_configured.thread_id,
        "/root/fresh",
        "worker completed",
        "root handled fresh child",
    )
    .await;
    start_root_prompt(&test.codex, FRESH_ROOT_PROMPT).await?;
    let fresh_thread_id = created_threads.recv().await?;
    let fresh_thread = test.thread_manager.get_thread(fresh_thread_id).await?;
    wait_for_turn_complete(&fresh_thread).await;
    assert_request_service_tier(&server, fresh_thread_id, 1, updated_request_service_tier).await;
    assert_parent_result(
        &test.codex,
        &fresh_parent,
        "root handled fresh child",
        updated_request_service_tier,
    )
    .await;
    Ok(())
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn evicted_role_subagent_uses_root_service_tier_after_reload() -> Result<()> {
    let server = start_mock_server().await;
    let mut builder = test_codex()
        .with_model("gpt-5.6-sol")
        .with_config(|config| {
            config.service_tier = Some("priority".to_string());
            config.multi_agent_v2.max_concurrent_threads_per_session = 2;
            configure_priority_role(config);
        });
    let test = builder.build_with_auto_env(&server).await?;
    let mut created_threads = test.thread_manager.subscribe_thread_created();

    mount_root_collaboration_call(
        &server,
        test.session_configured.thread_id,
        ROOT_PROMPT,
        SPAWN_CALL_ID,
        json!({
            "message": CHILD_PROMPT,
            "task_name": "original",
            "agent_type": PRIORITY_ROLE,
            "fork_turns": "none",
        }),
    )
    .await;
    mount_completed_child(&server, CHILD_PROMPT, ROOT_PROMPT, "worker completed").await;
    let original_parent = mount_parent_result(
        &server,
        test.session_configured.thread_id,
        "/root/original",
        "worker completed",
        "root handled original child",
    )
    .await;
    start_root_prompt(&test.codex, ROOT_PROMPT).await?;
    let original_thread_id = created_threads.recv().await?;
    let original_thread = test.thread_manager.get_thread(original_thread_id).await?;
    wait_for_turn_complete(&original_thread).await;
    assert_request_service_tier(&server, original_thread_id, 1, Some("priority")).await;
    assert_parent_result(
        &test.codex,
        &original_parent,
        "root handled original child",
        Some("priority"),
    )
    .await;
    drop(original_thread);

    mount_root_collaboration_call(
        &server,
        test.session_configured.thread_id,
        FRESH_ROOT_PROMPT,
        FRESH_SPAWN_CALL_ID,
        json!({
            "message": FRESH_CHILD_PROMPT,
            "task_name": "replacement",
            "fork_turns": "none",
        }),
    )
    .await;
    mount_completed_child(
        &server,
        FRESH_CHILD_PROMPT,
        FRESH_ROOT_PROMPT,
        "worker completed",
    )
    .await;
    let replacement_parent = mount_parent_result(
        &server,
        test.session_configured.thread_id,
        "/root/replacement",
        "worker completed",
        "root handled replacement child",
    )
    .await;
    start_root_prompt(&test.codex, FRESH_ROOT_PROMPT).await?;
    let replacement_thread_id = created_threads.recv().await?;
    let replacement_thread = test
        .thread_manager
        .get_thread(replacement_thread_id)
        .await?;
    wait_for_turn_complete(&replacement_thread).await;
    assert_request_service_tier(&server, replacement_thread_id, 1, Some("priority")).await;
    assert_parent_result(
        &test.codex,
        &replacement_parent,
        "root handled replacement child",
        Some("priority"),
    )
    .await;
    assert!(
        test.thread_manager
            .get_thread(original_thread_id)
            .await
            .is_err()
    );

    submit_thread_settings(
        &test.codex,
        ThreadSettingsOverrides {
            service_tier: Some(None),
            ..Default::default()
        },
    )
    .await?;
    test.thread_manager
        .ensure_multi_agent_v2_child_loaded(original_thread_id)
        .await?;
    let reloaded_thread = test.thread_manager.get_thread(original_thread_id).await?;
    assert_eq!(
        reloaded_thread.config_snapshot().await.service_tier,
        test.codex.config_snapshot().await.service_tier,
        "reload ignores the role tier and preserves the root-owned preference"
    );

    mount_completed_child(
        &server,
        FOLLOWUP_PROMPT,
        ROOT_PROMPT,
        "reloaded worker completed",
    )
    .await;
    let reloaded_parent = mount_parent_result(
        &server,
        test.session_configured.thread_id,
        "/root/original",
        "reloaded worker completed",
        "root handled reloaded child",
    )
    .await;
    reloaded_thread
        .start_or_steer_turn(TurnInputRequest::user_input(vec![UserInput::Text {
            text: FOLLOWUP_PROMPT.to_string(),
            text_elements: Vec::new(),
        }]))
        .await?;
    wait_for_turn_complete(&reloaded_thread).await;
    assert_request_service_tier(&server, original_thread_id, 2, /*expected*/ None).await;
    assert_parent_result(
        &test.codex,
        &reloaded_parent,
        "root handled reloaded child",
        /*expected_tier*/ None,
    )
    .await;
    reloaded_thread.shutdown_and_wait().await?;
    test.codex.shutdown_and_wait().await?;

    Ok(())
}
