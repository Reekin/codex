use std::fs;
use std::path::Path;
use std::path::PathBuf;
use std::time::Duration;

use anyhow::Context;
use codex_core::CodexThread;
use codex_core::TurnInputRequest;
use codex_core::TurnInputSubmission;
use codex_history::RolloutItem;
use codex_protocol::config_types::CollaborationMode;
use codex_protocol::config_types::ModeKind;
use codex_protocol::config_types::Settings;
use codex_protocol::items::TurnItem;
use codex_protocol::models::ContentItem;
use codex_protocol::models::ResponseItem;
use codex_protocol::protocol::EventMsg;
use codex_protocol::protocol::ThreadSettingsOverrides;
use codex_protocol::protocol::TurnCompleteEvent;
use codex_protocol::user_input::UserInput;
use core_test_support::fs_wait;
use core_test_support::hooks::trust_discovered_hooks;
use core_test_support::responses::ResponsesRequest;
use core_test_support::responses::ev_assistant_message;
use core_test_support::responses::ev_completed;
use core_test_support::responses::ev_custom_tool_call;
use core_test_support::responses::ev_reasoning_item;
use core_test_support::responses::ev_response_created;
use core_test_support::responses::mount_sse_once;
use core_test_support::responses::mount_sse_sequence;
use core_test_support::responses::sse;
use core_test_support::streaming_sse::StreamingSseChunk;
use core_test_support::streaming_sse::start_streaming_sse_server;
use core_test_support::test_codex::test_codex;
use core_test_support::wait_for_event;
use pretty_assertions::assert_eq;
use serde_json::Value;
use serde_json::json;
use tokio::sync::oneshot;
use tokio::time::sleep;

const RECOVERY_PROMPT_SNIPPET: &str = "previous response did not provide a visible final answer";
const RECOVERED_ANSWER: &str = "Recovered final answer.";

fn assistant_message_with_phase(id: &str, text: &str, phase: &str) -> Value {
    json!({
        "type": "response.output_item.done",
        "item": {
            "type": "message",
            "role": "assistant",
            "phase": phase,
            "id": id,
            "content": [{"type": "output_text", "text": text}]
        }
    })
}

fn empty_final_message(id: &str) -> Value {
    assistant_message_with_phase(id, " \n\t", "final_answer")
}

fn recovery_prompt_is_present(request: &ResponsesRequest) -> bool {
    request
        .message_input_texts("user")
        .iter()
        .any(|text| text.contains(RECOVERY_PROMPT_SNIPPET))
}

fn continued_response_completed(id: &str) -> Value {
    let mut completed = ev_completed(id);
    completed["response"]["end_turn"] = json!(false);
    completed
}

async fn start_user_turn(codex: &CodexThread, text: &str) -> anyhow::Result<TurnInputSubmission> {
    Ok(codex
        .start_or_steer_turn(TurnInputRequest::user_input(vec![UserInput::Text {
            text: text.to_string(),
            text_elements: Vec::new(),
        }]))
        .await?)
}

async fn wait_for_turn_complete(codex: &CodexThread) -> TurnCompleteEvent {
    let event = wait_for_event(codex, |event| matches!(event, EventMsg::TurnComplete(_))).await;
    let EventMsg::TurnComplete(completed) = event else {
        unreachable!("wait_for_event returned a non-matching event");
    };
    completed
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn empty_final_answer_retries_once_and_persists_recovery_prompt() -> anyhow::Result<()> {
    let server = wiremock::MockServer::start().await;
    let response_log = mount_sse_sequence(
        &server,
        vec![
            sse(vec![
                ev_response_created("resp-empty"),
                empty_final_message("msg-empty"),
                ev_completed("resp-empty"),
            ]),
            sse(vec![
                ev_response_created("resp-retry"),
                ev_assistant_message("msg-retry", RECOVERED_ANSWER),
                ev_completed("resp-retry"),
            ]),
        ],
    )
    .await;
    let test = test_codex().build_with_auto_env(&server).await?;

    let submission = start_user_turn(&test.codex, "please answer").await?;
    assert!(matches!(submission, TurnInputSubmission::Started { .. }));
    let completed = wait_for_turn_complete(&test.codex).await;

    assert_eq!(
        completed.last_agent_message.as_deref(),
        Some(RECOVERED_ANSWER)
    );
    let requests = response_log.requests();
    assert_eq!(requests.len(), 2);
    assert!(recovery_prompt_is_present(&requests[1]));

    test.codex.flush_rollout().await?;
    let history = test.codex.load_history(/*include_archived*/ false).await?;
    let persisted_recovery_prompts = history
        .items
        .iter()
        .filter(|item| match item {
            RolloutItem::ResponseItem(envelope) => match &envelope.item {
                ResponseItem::Message { role, content, .. } if role == "user" => {
                    content.iter().any(|content| {
                        matches!(
                            content,
                            ContentItem::InputText { text }
                                if text.contains(RECOVERY_PROMPT_SNIPPET)
                        )
                    })
                }
                _ => false,
            },
            _ => false,
        })
        .count();
    assert_eq!(persisted_recovery_prompts, 1);

    Ok(())
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn retry_budget_is_shared_across_missing_and_empty_responses() -> anyhow::Result<()> {
    let server = wiremock::MockServer::start().await;
    let response_log = mount_sse_sequence(
        &server,
        vec![
            sse(vec![
                ev_response_created("resp-reasoning"),
                ev_reasoning_item("reasoning-only", &["Still thinking."], &[]),
                ev_completed("resp-reasoning"),
            ]),
            sse(vec![
                ev_response_created("resp-empty"),
                empty_final_message("msg-empty"),
                ev_completed("resp-empty"),
            ]),
        ],
    )
    .await;
    let test = test_codex().build_with_auto_env(&server).await?;

    test.submit_turn("please answer").await?;

    let requests = response_log.requests();
    assert_eq!(requests.len(), 2);
    assert!(recovery_prompt_is_present(&requests[1]));

    Ok(())
}

#[test_case::test_case(false; "empty_then_untagged_non_empty")]
#[test_case::test_case(true; "explicit_non_empty_then_empty")]
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn non_empty_final_answer_dominates_empty_final_messages(
    non_empty_first: bool,
) -> anyhow::Result<()> {
    let server = wiremock::MockServer::start().await;
    let mut events = vec![ev_response_created("resp-mixed")];
    if non_empty_first {
        events.push(assistant_message_with_phase(
            "msg-answer",
            "Complete answer.",
            "final_answer",
        ));
        events.push(empty_final_message("msg-empty"));
    } else {
        events.push(empty_final_message("msg-empty"));
        events.push(ev_assistant_message("msg-answer", "Complete answer."));
    }
    events.push(ev_completed("resp-mixed"));
    let response_log = mount_sse_once(&server, sse(events)).await;
    let test = test_codex().build_with_auto_env(&server).await?;

    test.submit_turn("please answer").await?;

    assert_eq!(response_log.requests().len(), 1);
    Ok(())
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn commentary_only_response_retries_for_missing_final_answer() -> anyhow::Result<()> {
    let server = wiremock::MockServer::start().await;
    let response_log = mount_sse_sequence(
        &server,
        vec![
            sse(vec![
                ev_response_created("resp-commentary"),
                assistant_message_with_phase("msg-commentary", "Progress update.", "commentary"),
                ev_completed("resp-commentary"),
            ]),
            sse(vec![
                ev_response_created("resp-retry"),
                ev_assistant_message("msg-retry", RECOVERED_ANSWER),
                ev_completed("resp-retry"),
            ]),
        ],
    )
    .await;
    let test = test_codex().build_with_auto_env(&server).await?;

    test.submit_turn("please answer").await?;

    let requests = response_log.requests();
    assert_eq!(requests.len(), 2);
    assert!(recovery_prompt_is_present(&requests[1]));
    Ok(())
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn completed_response_without_output_does_not_retry() -> anyhow::Result<()> {
    let server = wiremock::MockServer::start().await;
    let response_log = mount_sse_once(
        &server,
        sse(vec![
            ev_response_created("resp-no-output"),
            ev_completed("resp-no-output"),
        ]),
    )
    .await;
    let test = test_codex().build_with_auto_env(&server).await?;

    test.submit_turn("please answer").await?;

    assert_eq!(response_log.requests().len(), 1);
    Ok(())
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn plan_mode_does_not_retry_empty_final_answer() -> anyhow::Result<()> {
    let server = wiremock::MockServer::start().await;
    let response_log = mount_sse_once(
        &server,
        sse(vec![
            ev_response_created("resp-plan"),
            empty_final_message("msg-plan"),
            ev_completed("resp-plan"),
        ]),
    )
    .await;
    let test = test_codex().build_with_auto_env(&server).await?;

    test.codex
        .start_or_steer_turn(
            TurnInputRequest::user_input(vec![UserInput::Text {
                text: "make a plan".to_string(),
                text_elements: Vec::new(),
            }])
            .with_thread_settings(ThreadSettingsOverrides {
                collaboration_mode: Some(CollaborationMode {
                    mode: ModeKind::Plan,
                    settings: Settings {
                        model: test.session_configured.model.clone(),
                        reasoning_effort: None,
                        developer_instructions: None,
                    },
                }),
                ..Default::default()
            }),
        )
        .await?;
    wait_for_turn_complete(&test.codex).await;

    assert_eq!(response_log.requests().len(), 1);
    Ok(())
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn tool_follow_up_precedes_missing_final_answer_recovery() -> anyhow::Result<()> {
    let server = wiremock::MockServer::start().await;
    let call_id = "unsupported-call";
    let response_log = mount_sse_sequence(
        &server,
        vec![
            sse(vec![
                ev_response_created("resp-tool"),
                ev_custom_tool_call(call_id, "unsupported_tool", "\"payload\""),
                ev_completed("resp-tool"),
            ]),
            sse(vec![
                ev_response_created("resp-reasoning"),
                ev_reasoning_item("reasoning-only", &["Preparing the answer."], &[]),
                ev_completed("resp-reasoning"),
            ]),
            sse(vec![
                ev_response_created("resp-retry"),
                ev_assistant_message("msg-retry", "Recovered after tool use."),
                ev_completed("resp-retry"),
            ]),
        ],
    )
    .await;
    let test = test_codex().build_with_auto_env(&server).await?;

    test.submit_turn("inspect and answer").await?;

    let requests = response_log.requests();
    assert_eq!(requests.len(), 3);
    requests[1].custom_tool_call_output(call_id);
    assert!(!recovery_prompt_is_present(&requests[1]));
    assert!(recovery_prompt_is_present(&requests[2]));
    Ok(())
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn output_presence_does_not_carry_across_tool_follow_up() -> anyhow::Result<()> {
    let server = wiremock::MockServer::start().await;
    let call_id = "unsupported-call";
    let response_log = mount_sse_sequence(
        &server,
        vec![
            sse(vec![
                ev_response_created("resp-tool"),
                ev_custom_tool_call(call_id, "unsupported_tool", "\"payload\""),
                ev_completed("resp-tool"),
            ]),
            sse(vec![
                ev_response_created("resp-no-output"),
                ev_completed("resp-no-output"),
            ]),
        ],
    )
    .await;
    let test = test_codex().build_with_auto_env(&server).await?;

    test.submit_turn("inspect and answer").await?;

    let requests = response_log.requests();
    assert_eq!(requests.len(), 2);
    requests[1].custom_tool_call_output(call_id);
    assert!(!recovery_prompt_is_present(&requests[1]));
    Ok(())
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn model_follow_up_precedes_final_answer_recovery() -> anyhow::Result<()> {
    let server = wiremock::MockServer::start().await;
    let response_log = mount_sse_sequence(
        &server,
        vec![
            sse(vec![
                ev_response_created("resp-continue"),
                empty_final_message("msg-empty"),
                continued_response_completed("resp-continue"),
            ]),
            sse(vec![
                ev_response_created("resp-reasoning"),
                ev_reasoning_item("reasoning-only", &["Continuing."], &[]),
                ev_completed("resp-reasoning"),
            ]),
            sse(vec![
                ev_response_created("resp-retry"),
                ev_assistant_message("msg-retry", RECOVERED_ANSWER),
                ev_completed("resp-retry"),
            ]),
        ],
    )
    .await;
    let test = test_codex().build_with_auto_env(&server).await?;

    test.submit_turn("please answer").await?;

    let requests = response_log.requests();
    assert_eq!(requests.len(), 3);
    assert!(!recovery_prompt_is_present(&requests[1]));
    assert!(recovery_prompt_is_present(&requests[2]));
    Ok(())
}

fn stream_chunk(events: Vec<Value>) -> StreamingSseChunk {
    StreamingSseChunk {
        gate: None,
        body: sse(events),
    }
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn pending_user_input_precedes_final_answer_recovery() -> anyhow::Result<()> {
    let (release_completion, completion_gate) = oneshot::channel();
    let first_chunks = vec![
        stream_chunk(vec![
            ev_response_created("resp-empty"),
            empty_final_message("msg-empty"),
        ]),
        StreamingSseChunk {
            gate: Some(completion_gate),
            body: sse(vec![ev_completed("resp-empty")]),
        },
    ];
    let second_chunks = vec![stream_chunk(vec![
        ev_response_created("resp-follow-up"),
        ev_assistant_message("msg-follow-up", "Answered after the user follow-up."),
        ev_completed("resp-follow-up"),
    ])];
    let (server, _completions) =
        start_streaming_sse_server(vec![first_chunks, second_chunks]).await;
    let test = test_codex().build_with_streaming_server(&server).await?;

    let submission = start_user_turn(&test.codex, "first prompt").await?;
    assert!(matches!(submission, TurnInputSubmission::Started { .. }));
    wait_for_event(&test.codex, |event| {
        matches!(
            event,
            EventMsg::ItemCompleted(completed)
                if matches!(&completed.item, TurnItem::AgentMessage(_))
        )
    })
    .await;

    let submission = start_user_turn(&test.codex, "follow-up while running").await?;
    assert!(matches!(submission, TurnInputSubmission::Steered { .. }));
    release_completion
        .send(())
        .map_err(|_| anyhow::anyhow!("failed to release response completion"))?;

    let completed = wait_for_turn_complete(&test.codex).await;
    assert_eq!(
        completed.last_agent_message.as_deref(),
        Some("Answered after the user follow-up.")
    );

    let requests = server.requests().await;
    assert_eq!(requests.len(), 2);
    let second_request: Value =
        serde_json::from_slice(&requests[1]).context("parse second request")?;
    let second_user_messages = second_request["input"]
        .as_array()
        .context("second request input")?
        .iter()
        .filter(|item| item["type"] == "message" && item["role"] == "user")
        .filter_map(|item| item["content"].as_array())
        .flatten()
        .filter_map(|content| content["text"].as_str())
        .collect::<Vec<_>>();
    assert!(second_user_messages.contains(&"follow-up while running"));
    assert!(
        !second_user_messages
            .iter()
            .any(|text| text.contains(RECOVERY_PROMPT_SNIPPET))
    );

    server.shutdown().await;
    Ok(())
}

#[cfg(windows)]
fn python_executable() -> &'static str {
    "python"
}

#[cfg(not(windows))]
fn python_executable() -> &'static str {
    "python3"
}

fn write_terminal_hook_fixtures(home: &Path) -> anyhow::Result<()> {
    let stop_script_path = home.join("final_answer_retry_stop.py");
    let stop_log_path = home.join("final_answer_retry_stop.jsonl");
    let stop_log_path_display = stop_log_path.display();
    let stop_script = format!(
        r#"import json
from pathlib import Path
import sys

payload = json.load(sys.stdin)
with Path(r"{stop_log_path_display}").open("a", encoding="utf-8") as handle:
    handle.write(json.dumps(payload) + "\n")
print("{{}}")
"#
    );
    fs::write(&stop_script_path, stop_script)?;

    let python = python_executable();
    let stop_script_path_display = stop_script_path.display();
    let hooks = json!({
        "hooks": {
            "Stop": [{
                "hooks": [{
                    "type": "command",
                    "command": format!("{python} {stop_script_path_display}"),
                }]
            }]
        }
    });
    fs::write(home.join("hooks.json"), hooks.to_string())?;

    let notify_script_path = home.join("final_answer_retry_after_agent.py");
    let notify_log_path = home.join("final_answer_retry_after_agent.jsonl");
    let notify_log_path_display = notify_log_path.display();
    let notify_script = format!(
        r#"from pathlib import Path
import sys

with Path(r"{notify_log_path_display}").open("a", encoding="utf-8") as handle:
    handle.write(sys.argv[-1] + "\n")
"#
    );
    fs::write(notify_script_path, notify_script)?;
    Ok(())
}

async fn read_spawned_hook_log(path: PathBuf) -> anyhow::Result<String> {
    fs_wait::wait_for_path_exists(path.clone(), Duration::from_secs(5)).await?;
    for _ in 0..50 {
        let text = fs::read_to_string(&path)?;
        if !text.trim().is_empty() {
            return Ok(text);
        }
        sleep(Duration::from_millis(20)).await;
    }
    anyhow::bail!("hook log remained empty at {}", path.display())
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn terminal_hooks_and_completion_observe_only_recovered_answer() -> anyhow::Result<()> {
    let server = wiremock::MockServer::start().await;
    let response_log = mount_sse_sequence(
        &server,
        vec![
            sse(vec![
                ev_response_created("resp-empty"),
                empty_final_message("msg-empty"),
                ev_completed("resp-empty"),
            ]),
            sse(vec![
                ev_response_created("resp-retry"),
                ev_assistant_message("msg-retry", RECOVERED_ANSWER),
                ev_completed("resp-retry"),
            ]),
        ],
    )
    .await;
    let mut builder = test_codex()
        .with_pre_build_hook(|home| {
            write_terminal_hook_fixtures(home).expect("write terminal hook fixtures");
        })
        .with_config(|config| {
            let notify_script = config.codex_home.join("final_answer_retry_after_agent.py");
            config.notify = Some(vec![
                python_executable().to_string(),
                notify_script.display().to_string(),
            ]);
            trust_discovered_hooks(config);
        });
    let test = builder.build_with_auto_env(&server).await?;

    start_user_turn(&test.codex, "please answer").await?;
    let completed = wait_for_turn_complete(&test.codex).await;

    assert_eq!(
        completed.last_agent_message.as_deref(),
        Some(RECOVERED_ANSWER)
    );
    assert_eq!(response_log.requests().len(), 2);

    let stop_log =
        fs::read_to_string(test.codex_home_path().join("final_answer_retry_stop.jsonl"))?;
    let stop_payloads = stop_log
        .lines()
        .map(serde_json::from_str::<Value>)
        .collect::<Result<Vec<_>, _>>()?;
    assert_eq!(stop_payloads.len(), 1);
    assert_eq!(
        stop_payloads[0]["last_assistant_message"],
        json!(RECOVERED_ANSWER)
    );

    let after_agent_log = read_spawned_hook_log(
        test.codex_home_path()
            .join("final_answer_retry_after_agent.jsonl"),
    )
    .await?;
    let after_agent_payloads = after_agent_log
        .lines()
        .map(serde_json::from_str::<Value>)
        .collect::<Result<Vec<_>, _>>()?;
    assert_eq!(after_agent_payloads.len(), 1);
    assert_eq!(
        after_agent_payloads[0]["last-assistant-message"],
        json!(RECOVERED_ANSWER)
    );

    Ok(())
}
