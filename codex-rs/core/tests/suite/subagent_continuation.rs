//! Child results continue a parent whose turn ended naturally, through the production
//! model path and in both multi-agent versions.

use anyhow::Result;
use codex_core::TurnInputRequest;
use codex_features::Feature;
use codex_protocol::protocol::EventMsg;
use codex_protocol::protocol::Op;
use codex_protocol::user_input::UserInput;
use core_test_support::responses::ev_assistant_message;
use core_test_support::responses::ev_completed;
use core_test_support::responses::ev_function_call_with_namespace;
use core_test_support::responses::ev_response_created;
use core_test_support::responses::sse;
use core_test_support::responses::sse_response;
use core_test_support::responses::start_mock_server;
use core_test_support::responses::strip_metadata_from_json;
use core_test_support::responses::strip_response_item_ids_from_json;
use core_test_support::skip_if_no_network;
use core_test_support::streaming_sse::StreamingSseChunk;
use core_test_support::streaming_sse::start_streaming_sse_server;
use core_test_support::test_codex::TestCodex;
use core_test_support::test_codex::test_codex;
use core_test_support::wait_for_event;
use core_test_support::wait_for_event_with_timeout;
use pretty_assertions::assert_eq;
use serde_json::Value;
use serde_json::json;
use std::time::Duration;
use test_case::test_case;
use tokio::sync::oneshot;
use tokio::time::timeout;
use wiremock::Mock;
use wiremock::ResponseTemplate;
use wiremock::matchers::method;
use wiremock::matchers::path;

#[derive(Clone, Copy)]
enum Version {
    V1,
    V2,
}

#[derive(Clone, Copy)]
enum Arrival {
    /// The parent finished its reply before the child result arrives.
    Idle,
    /// The child result arrives while the parent is still sampling.
    Active,
    /// The user stopped the parent before the child result arrives.
    Interrupted,
}

#[derive(Clone, Copy)]
enum Outcome {
    Completed,
    Errored,
}

const RESUME_PROMPT: &str = "resume explicitly";

fn request_body(request: &wiremock::Request) -> Value {
    let bytes = if request
        .headers
        .get("content-encoding")
        .is_some_and(|value| value == "zstd")
    {
        zstd::stream::decode_all(std::io::Cursor::new(&request.body)).unwrap()
    } else {
        request.body.clone()
    };
    serde_json::from_slice(&bytes).unwrap()
}

fn request_thread_id(request: &wiremock::Request) -> String {
    request
        .headers
        .get("thread-id")
        .and_then(|value| value.to_str().ok())
        .expect("model requests carry their thread identity")
        .to_string()
}

fn carries_child_result(body: &Value) -> bool {
    let input = body["input"].to_string();
    input.contains("<subagent_notification>") || input.contains("Message Type: FINAL_ANSWER")
}

fn answer(id: &str, text: &str) -> String {
    sse(vec![
        ev_response_created(id),
        ev_assistant_message(&format!("msg-{id}"), text),
        ev_completed(id),
    ])
}

async fn parent_requests(server: &wiremock::MockServer, parent_id: &str) -> Vec<Value> {
    server
        .received_requests()
        .await
        .unwrap()
        .iter()
        .filter(|request| request.url.path() == "/v1/responses")
        .filter(|request| request_thread_id(request) == parent_id)
        .map(request_body)
        .collect()
}

async fn assert_no_turn_start(codex: &codex_core::CodexThread, reason: &str) {
    assert!(
        timeout(Duration::from_millis(300), async {
            loop {
                let event = codex.next_event().await.unwrap();
                if matches!(event.msg, EventMsg::TurnStarted(_)) {
                    break;
                }
            }
        })
        .await
        .is_err(),
        "{reason}"
    );
}

/// Builds a parent with collaboration in the selected multi-agent version.
async fn build_parent(
    server: &wiremock::MockServer,
    version: Version,
    subagent_continuation: bool,
) -> Result<TestCodex> {
    test_codex()
        .with_model("koffing")
        .with_config(move |config| {
            config
                .features
                .enable(Feature::Collab)
                .expect("enable collaboration");
            match version {
                Version::V1 => config.features.disable(Feature::MultiAgentV2),
                Version::V2 => config.features.enable(Feature::MultiAgentV2),
            }
            .expect("select multi-agent version");
            if subagent_continuation {
                config
                    .features
                    .enable(Feature::SubagentContinuation)
                    .expect("enable subagent continuation");
            }
            config.model_provider.request_max_retries = Some(0);
            config.model_provider.stream_max_retries = Some(0);
            config.model_provider.supports_websockets = false;
        })
        .build_with_auto_env(server)
        .await
}

fn spawn_child_call(version: Version) -> Value {
    match version {
        Version::V1 => ev_function_call_with_namespace(
            "spawn-child",
            "multi_agent_v1",
            "spawn_agent",
            &json!({"message": "do child work"}).to_string(),
        ),
        Version::V2 => ev_function_call_with_namespace(
            "spawn-child",
            "collaboration",
            "spawn_agent",
            &json!({"message": "do child work", "task_name": "worker", "fork_turns": "none"})
                .to_string(),
        ),
    }
}

#[test_case(Version::V1, Arrival::Idle, Outcome::Completed; "v1_idle_completion")]
#[test_case(Version::V2, Arrival::Idle, Outcome::Completed; "v2_idle_completion")]
#[test_case(Version::V1, Arrival::Idle, Outcome::Errored; "v1_idle_error")]
#[test_case(Version::V2, Arrival::Idle, Outcome::Errored; "v2_idle_error")]
#[test_case(Version::V1, Arrival::Active, Outcome::Completed; "v1_active")]
#[test_case(Version::V2, Arrival::Active, Outcome::Completed; "v2_active")]
#[test_case(Version::V1, Arrival::Interrupted, Outcome::Completed; "v1_interrupted")]
#[test_case(Version::V2, Arrival::Interrupted, Outcome::Completed; "v2_interrupted")]
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn child_result_continues_parent(
    version: Version,
    arrival: Arrival,
    outcome: Outcome,
) -> Result<()> {
    skip_if_no_network!(Ok(()));
    let server = start_mock_server().await;
    let (release_child, child_gate) = oneshot::channel();
    let child_response = match outcome {
        Outcome::Completed => answer("child", "child result"),
        Outcome::Errored => sse(vec![ev_response_created("child")]),
    };
    let (child_server, _) = start_streaming_sse_server(vec![vec![StreamingSseChunk {
        gate: Some(child_gate),
        body: child_response,
    }]])
    .await;
    let (release_parent, parent_gate) = oneshot::channel();
    let mut release_parent = Some(release_parent);
    let (parent_server, _) = start_streaming_sse_server(vec![vec![StreamingSseChunk {
        gate: Some(parent_gate),
        body: answer("parent-finish", "parent finished"),
    }]])
    .await;
    let test = build_parent(&server, version, /*subagent_continuation*/ true).await?;
    let parent_id = test.session_configured.thread_id.to_string();
    let mut created = test.thread_manager.subscribe_thread_created();
    let spawn = sse(vec![
        ev_response_created("spawn"),
        spawn_child_call(version),
        ev_completed("spawn"),
    ]);
    let parent_match_id = parent_id.clone();
    let child_url = format!("{}/v1/responses", child_server.uri());
    let parent_url = format!("{}/v1/responses", parent_server.uri());
    // Route concurrent parent/child requests independently; each streaming response
    // stays held until the test explicitly releases it. A 307 preserves the POST body.
    Mock::given(method("POST"))
        .and(path("/v1/responses"))
        .respond_with(move |request: &wiremock::Request| {
            if request_thread_id(request) != parent_match_id {
                return ResponseTemplate::new(307).insert_header("location", child_url.as_str());
            }
            let body = request_body(request);
            if carries_child_result(&body) {
                sse_response(answer("processed", "processed child"))
            } else if body["input"].to_string().contains(RESUME_PROMPT) {
                // Fresh user input is sampled before pending mailbox input. An
                // empty response lets the next sample drain the retained result.
                sse_response(sse(vec![
                    ev_response_created("resume-before-result"),
                    ev_completed("resume-before-result"),
                ]))
            } else if body["input"].to_string().contains("spawn-child") {
                ResponseTemplate::new(307).insert_header("location", parent_url.as_str())
            } else {
                sse_response(spawn.clone())
            }
        })
        .mount(&server)
        .await;

    test.codex
        .start_or_steer_turn(TurnInputRequest::user_input(vec![UserInput::Text {
            text: "delegate and finish".into(),
            text_elements: Vec::new(),
        }]))
        .await?;
    let EventMsg::TurnStarted(started) = wait_for_event(&test.codex, |event| {
        matches!(event, EventMsg::TurnStarted(_))
    })
    .await
    else {
        unreachable!()
    };
    let child_id = timeout(Duration::from_secs(10), created.recv()).await??;
    let child = test.thread_manager.get_thread(child_id).await?;
    timeout(Duration::from_secs(10), async {
        child_server.wait_for_request_count(1).await;
        parent_server.wait_for_request_count(1).await;
    })
    .await?;

    let mut active_turn = Some(started.turn_id.clone());
    match arrival {
        Arrival::Idle => {
            release_parent.take().unwrap().send(()).unwrap();
            wait_for_event(&test.codex, |event| {
                matches!(event, EventMsg::TurnComplete(_))
            })
            .await;
            active_turn = None;
        }
        Arrival::Interrupted => {
            test.codex.submit(Op::Interrupt).await?;
            wait_for_event(&test.codex, |event| {
                matches!(event, EventMsg::TurnAborted(_))
            })
            .await;
            active_turn = None;
            release_parent.take().unwrap().send(()).unwrap();
        }
        Arrival::Active => {}
    }
    release_child.send(()).unwrap();
    wait_for_event(&child, |event| matches!(event, EventMsg::TurnComplete(_))).await;

    if matches!(arrival, Arrival::Active | Arrival::Interrupted) {
        // Give delivery time to land while parent admission must stay blocked.
        assert_no_turn_start(
            &test.codex,
            "child result must not start a concurrent or interrupted parent",
        )
        .await;
        assert_eq!(parent_requests(&server, &parent_id).await.len(), 2);
    }
    match arrival {
        Arrival::Active => release_parent.take().unwrap().send(()).unwrap(),
        Arrival::Interrupted => {
            test.codex
                .start_or_steer_turn(TurnInputRequest::user_input(vec![UserInput::Text {
                    text: RESUME_PROMPT.into(),
                    text_elements: Vec::new(),
                }]))
                .await?;
        }
        Arrival::Idle => {}
    }
    let mut continuation_starts = 0;
    timeout(Duration::from_secs(10), async {
        loop {
            match test.codex.next_event().await.unwrap().msg {
                EventMsg::TurnStarted(event) => {
                    assert!(active_turn.replace(event.turn_id).is_none());
                    continuation_starts += 1;
                }
                EventMsg::TurnComplete(event) => {
                    assert_eq!(active_turn.take(), Some(event.turn_id));
                    if event.last_agent_message.as_deref() == Some("processed child") {
                        break;
                    }
                }
                _ => {}
            }
        }
    })
    .await?;
    match arrival {
        // A final assistant item can defer mailbox input until after TurnComplete.
        // Either delivery boundary is valid; the event loop enforces serial turns.
        Arrival::Active => assert!(continuation_starts <= 1),
        Arrival::Idle | Arrival::Interrupted => assert_eq!(continuation_starts, 1),
    }
    assert_no_turn_start(
        &test.codex,
        "a consumed result must not start another continuation",
    )
    .await;

    let parent_requests = parent_requests(&server, &parent_id).await;
    match arrival {
        Arrival::Interrupted => assert!((3..=4).contains(&parent_requests.len())),
        Arrival::Idle | Arrival::Active => assert_eq!(parent_requests.len(), 3),
    }
    let result_requests: Vec<_> = parent_requests
        .iter()
        .filter(|body| carries_child_result(body))
        .collect();
    assert_eq!(result_requests.len(), 1);
    let continued = result_requests[0];
    assert_eq!(continued["model"], parent_requests[0]["model"]);
    assert_eq!(
        continued["input"].to_string().contains(RESUME_PROMPT),
        matches!(arrival, Arrival::Interrupted)
    );
    let error = "stream disconnected before completion: stream closed before response.completed";
    match version {
        Version::V1 => {
            let notifications: Vec<_> = continued["input"]
                .as_array()
                .unwrap()
                .iter()
                .flat_map(|item| item["content"].as_array().into_iter().flatten())
                .filter_map(|content| content["text"].as_str())
                .filter_map(|text| text.split_once("<subagent_notification>"))
                .map(|(_, rest)| rest.split_once("</subagent_notification>").unwrap().0)
                .map(|text| serde_json::from_str::<Value>(text).unwrap())
                .collect();
            assert_eq!(
                notifications,
                vec![json!({
                    "agent_path": child_id.to_string(),
                    "status": match outcome {
                        Outcome::Completed => json!({"completed": "child result"}),
                        Outcome::Errored => json!({"errored": error}),
                    },
                })]
            );
        }
        Version::V2 => {
            let payload = match outcome {
                Outcome::Completed => "child result".to_string(),
                Outcome::Errored => format!(
                    "Agent errored: {error}\n\nThis agent's turn failed. If you still need this agent, use the available collaboration tools to give it another task."
                ),
            };
            let messages: Vec<_> = continued["input"]
                .as_array()
                .unwrap()
                .iter()
                .filter(|item| item["type"] == "agent_message")
                .cloned()
                .collect();
            assert_eq!(
                strip_response_item_ids_from_json(strip_metadata_from_json(json!(messages))),
                json!([{
                    "type": "agent_message",
                    "author": "/root/worker",
                    "recipient": "/root",
                    "content": [{"type": "input_text", "text": format!("Message Type: FINAL_ANSWER\nTask name: /root\nSender: /root/worker\nPayload:\n{payload}")}],
                }])
            );
        }
    }
    child_server.shutdown().await;
    parent_server.shutdown().await;
    Ok(())
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn v2_child_shutdown_notice_waits_for_next_user_turn() -> Result<()> {
    skip_if_no_network!(Ok(()));
    let server = start_mock_server().await;
    // The child stays running until it is shut down; its response is never released.
    let (release_child, child_gate) = oneshot::channel();
    let (child_server, _) = start_streaming_sse_server(vec![vec![StreamingSseChunk {
        gate: Some(child_gate),
        body: answer("child", "child result"),
    }]])
    .await;
    let test = build_parent(&server, Version::V2, /*subagent_continuation*/ true).await?;
    let parent_id = test.session_configured.thread_id.to_string();
    let mut created = test.thread_manager.subscribe_thread_created();
    let spawn = sse(vec![
        ev_response_created("spawn"),
        spawn_child_call(Version::V2),
        ev_completed("spawn"),
    ]);
    let parent_match_id = parent_id.clone();
    let child_url = format!("{}/v1/responses", child_server.uri());
    Mock::given(method("POST"))
        .and(path("/v1/responses"))
        .respond_with(move |request: &wiremock::Request| {
            if request_thread_id(request) != parent_match_id {
                return ResponseTemplate::new(307).insert_header("location", child_url.as_str());
            }
            let body = request_body(request);
            if carries_child_result(&body) {
                sse_response(answer("processed", "processed child"))
            } else if body["input"].to_string().contains(RESUME_PROMPT) {
                // Fresh user input is sampled before pending mailbox input. An
                // empty response lets the next sample drain the retained notice.
                sse_response(sse(vec![
                    ev_response_created("resume-before-result"),
                    ev_completed("resume-before-result"),
                ]))
            } else if body["input"].to_string().contains("spawn-child") {
                sse_response(answer("parent-finish", "parent finished"))
            } else {
                sse_response(spawn.clone())
            }
        })
        .mount(&server)
        .await;

    test.codex
        .start_or_steer_turn(TurnInputRequest::user_input(vec![UserInput::Text {
            text: "delegate and finish".into(),
            text_elements: Vec::new(),
        }]))
        .await?;
    let child_id = timeout(Duration::from_secs(10), created.recv()).await??;
    let child = test.thread_manager.get_thread(child_id).await?;
    timeout(
        Duration::from_secs(10),
        child_server.wait_for_request_count(1),
    )
    .await?;
    wait_for_event_with_timeout(
        &test.codex,
        |event| matches!(event, EventMsg::TurnComplete(_)),
        Duration::from_secs(10),
    )
    .await;
    assert_eq!(parent_requests(&server, &parent_id).await.len(), 2);

    // Shutting down the still-running child reports a lifecycle notice, not a result.
    child.submit(Op::Shutdown).await?;
    wait_for_event_with_timeout(
        &child,
        |event| matches!(event, EventMsg::ShutdownComplete),
        Duration::from_secs(10),
    )
    .await;
    tokio::time::sleep(Duration::from_millis(200)).await;
    assert_no_turn_start(
        &test.codex,
        "a child shutdown notice must not resume a parent whose turn ended",
    )
    .await;
    assert_eq!(parent_requests(&server, &parent_id).await.len(), 2);

    test.codex
        .start_or_steer_turn(TurnInputRequest::user_input(vec![UserInput::Text {
            text: RESUME_PROMPT.into(),
            text_elements: Vec::new(),
        }]))
        .await?;
    let mut turn_starts = 0;
    timeout(Duration::from_secs(10), async {
        loop {
            match test.codex.next_event().await.unwrap().msg {
                EventMsg::TurnStarted(_) => turn_starts += 1,
                EventMsg::TurnComplete(event)
                    if event.last_agent_message.as_deref() == Some("processed child") =>
                {
                    break;
                }
                _ => {}
            }
        }
    })
    .await?;
    assert_eq!(turn_starts, 1);

    let parent_requests = parent_requests(&server, &parent_id).await;
    assert!((3..=4).contains(&parent_requests.len()));
    let notice_requests: Vec<_> = parent_requests
        .iter()
        .filter(|body| carries_child_result(body))
        .collect();
    assert_eq!(notice_requests.len(), 1);
    let continued = notice_requests[0];
    assert!(continued["input"].to_string().contains(RESUME_PROMPT));
    let messages: Vec<_> = continued["input"]
        .as_array()
        .unwrap()
        .iter()
        .filter(|item| item["type"] == "agent_message")
        .cloned()
        .collect();
    assert_eq!(
        strip_response_item_ids_from_json(strip_metadata_from_json(json!(messages))),
        json!([{
            "type": "agent_message",
            "author": "/root/worker",
            "recipient": "/root",
            "content": [{"type": "input_text", "text": "Message Type: FINAL_ANSWER\nTask name: /root\nSender: /root/worker\nPayload:\nAgent shut down."}],
        }])
    );
    let _ = release_child.send(());
    child_server.shutdown().await;
    Ok(())
}

#[test_case(Version::V1; "v1")]
#[test_case(Version::V2; "v2")]
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn disabled_flag_keeps_child_results_passive(version: Version) -> Result<()> {
    skip_if_no_network!(Ok(()));
    let server = start_mock_server().await;
    let (release_child, child_gate) = oneshot::channel();
    let (child_server, _) = start_streaming_sse_server(vec![vec![StreamingSseChunk {
        gate: Some(child_gate),
        body: answer("child", "child result"),
    }]])
    .await;
    let test = build_parent(&server, version, /*subagent_continuation*/ false).await?;
    let parent_id = test.session_configured.thread_id.to_string();
    let mut created = test.thread_manager.subscribe_thread_created();
    let spawn = sse(vec![
        ev_response_created("spawn"),
        spawn_child_call(version),
        ev_completed("spawn"),
    ]);
    let parent_match_id = parent_id.clone();
    let child_url = format!("{}/v1/responses", child_server.uri());
    Mock::given(method("POST"))
        .and(path("/v1/responses"))
        .respond_with(move |request: &wiremock::Request| {
            if request_thread_id(request) != parent_match_id {
                return ResponseTemplate::new(307).insert_header("location", child_url.as_str());
            }
            let body = request_body(request);
            if carries_child_result(&body) {
                sse_response(answer("processed", "processed child"))
            } else if body["input"].to_string().contains("spawn-child") {
                sse_response(answer("parent-finish", "parent finished"))
            } else {
                sse_response(spawn.clone())
            }
        })
        .mount(&server)
        .await;

    test.codex
        .start_or_steer_turn(TurnInputRequest::user_input(vec![UserInput::Text {
            text: "delegate and finish".into(),
            text_elements: Vec::new(),
        }]))
        .await?;
    let child_id = timeout(Duration::from_secs(10), created.recv()).await??;
    let child = test.thread_manager.get_thread(child_id).await?;
    timeout(
        Duration::from_secs(10),
        child_server.wait_for_request_count(1),
    )
    .await?;
    wait_for_event_with_timeout(
        &test.codex,
        |event| matches!(event, EventMsg::TurnComplete(_)),
        Duration::from_secs(10),
    )
    .await;

    release_child.send(()).unwrap();
    wait_for_event(&child, |event| matches!(event, EventMsg::TurnComplete(_))).await;
    assert_no_turn_start(
        &test.codex,
        "without subagent_continuation a child result must not start a parent turn",
    )
    .await;
    let parent_requests = parent_requests(&server, &parent_id).await;
    assert_eq!(parent_requests.len(), 2);
    let tools = parent_requests[0]["tools"].to_string();
    assert!(!tools.contains("automatically resumes its parent"));
    assert!(!tools.contains("New user input ends the wait early"));
    child_server.shutdown().await;
    Ok(())
}
