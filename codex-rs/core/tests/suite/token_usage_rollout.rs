//! Verifies observed Responses API usage is durably recorded in rollout history.

use anyhow::Result;
use codex_core::TurnInputRequest;
use codex_history::RolloutItem;
use codex_protocol::SessionId;
use codex_protocol::protocol::EventMsg;
use codex_protocol::protocol::TokenUsageRecord;
use codex_protocol::user_input::UserInput;
use core_test_support::responses::ev_assistant_message;
use core_test_support::responses::ev_completed_with_tokens;
use core_test_support::responses::ev_function_call;
use core_test_support::responses::ev_response_created;
use core_test_support::responses::ev_web_search_call_done;
use core_test_support::responses::mount_compact_json_once;
use core_test_support::responses::mount_sse_sequence;
use core_test_support::responses::sse;
use core_test_support::responses::start_mock_server;
use core_test_support::skip_if_no_network;
use core_test_support::test_codex::test_codex;
use core_test_support::wait_for_event;
use pretty_assertions::assert_eq;
use serde_json::json;

fn token_usage_records(path: &std::path::Path) -> Vec<TokenUsageRecord> {
    std::fs::read_to_string(path)
        .expect("read rollout")
        .lines()
        .filter_map(|line| codex_rollout::parse_rollout_line(line).ok())
        .filter_map(|line| match line.item {
            RolloutItem::TokenUsageRecord(record) => Some(record),
            _ => None,
        })
        .collect()
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn observed_response_usage_accumulates_per_turn_and_thread() -> Result<()> {
    skip_if_no_network!(Ok(()));

    let server = start_mock_server().await;
    let plan_args = json!({
        "plan": [{
            "step": "keep sampling",
            "status": "in_progress"
        }]
    })
    .to_string();
    mount_sse_sequence(
        &server,
        vec![
            sse(vec![
                ev_response_created("response-a"),
                ev_function_call("call-a", "update_plan", &plan_args),
                ev_completed_with_tokens("response-a", /*total_tokens*/ 120),
            ]),
            sse(vec![
                ev_response_created("response-b"),
                ev_assistant_message("message-b", "done"),
                ev_completed_with_tokens("response-b", /*total_tokens*/ 80),
            ]),
            sse(vec![
                ev_response_created("response-c"),
                ev_assistant_message("message-c", "next"),
                ev_completed_with_tokens("response-c", /*total_tokens*/ 30),
            ]),
            sse(vec![
                ev_response_created("response-without-usage"),
                ev_assistant_message("message-d", "no usage"),
                json!({
                    "type": "response.completed",
                    "response": {
                        "id": "response-without-usage"
                    }
                }),
            ]),
        ],
    )
    .await;
    let test = test_codex().build_with_auto_env(&server).await?;
    let rollout_path = test.codex.rollout_path().expect("rollout path");
    let home = test.home.clone();

    test.submit_turn("first").await?;
    test.codex.shutdown_and_wait().await?;

    let resumed = test_codex()
        .resume(&server, home, rollout_path.clone())
        .await?;
    for prompt in ["second", "third"] {
        resumed.submit_turn(prompt).await?;
    }
    resumed.codex.shutdown_and_wait().await?;

    let records = token_usage_records(&rollout_path);
    assert_eq!(records.len(), 3);
    assert_eq!(
        records
            .iter()
            .map(|record| {
                (
                    record.response_id.as_str(),
                    record.turn_token_usage.total_tokens,
                    record.thread_token_usage.total_tokens,
                )
            })
            .collect::<Vec<_>>(),
        vec![
            ("response-a", 120, 120),
            ("response-b", 200, 200),
            ("response-c", 30, 230),
        ]
    );
    assert_eq!(records[0].turn_id, records[1].turn_id);
    assert_ne!(records[1].turn_id, records[2].turn_id);
    assert!(records.iter().all(|record| {
        record.session_id == SessionId::from(record.thread_id)
            && record.root_turn_id == record.turn_id
    }));

    Ok(())
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn server_web_search_usage_does_not_inflate_active_context() -> Result<()> {
    skip_if_no_network!(Ok(()));

    let server = start_mock_server().await;
    // A server-side search re-reads the whole context inside one response, so the provider
    // reports far more input than the context actually holds.
    let request_log = mount_sse_sequence(
        &server,
        vec![
            sse(vec![
                ev_assistant_message("m1", "FIRST_FINAL"),
                ev_completed_with_tokens("r1", /*total_tokens*/ 100),
            ]),
            sse(vec![
                ev_web_search_call_done("ws-1", "completed", "weather"),
                ev_assistant_message("m2", "SEARCHED_FINAL"),
                ev_completed_with_tokens("r2", /*total_tokens*/ 5_000),
            ]),
            sse(vec![
                ev_assistant_message("m3", "THIRD_FINAL"),
                ev_completed_with_tokens("r3", /*total_tokens*/ 120),
            ]),
        ],
    )
    .await;
    let compact_mock = mount_compact_json_once(&server, json!({ "output": [] })).await;
    let test = test_codex()
        .with_config(|config| config.model_auto_compact_token_limit = Some(1_000))
        .build_with_auto_env(&server)
        .await?;

    let mut search_info = None;
    for user in ["FIRST_USER", "SEARCH_USER", "THIRD_USER"] {
        test.codex
            .start_or_steer_turn(TurnInputRequest::user_input(vec![UserInput::Text {
                text: user.into(),
                text_elements: Vec::new(),
            }]))
            .await?;
        loop {
            match wait_for_event(&test.codex, |event| {
                matches!(event, EventMsg::TurnComplete(_) | EventMsg::TokenCount(_))
            })
            .await
            {
                EventMsg::TokenCount(count) if user == "SEARCH_USER" => {
                    search_info = count.info.or(search_info);
                }
                EventMsg::TurnComplete(_) => break,
                _ => {}
            }
        }
    }

    let search_info = search_info.expect("search turn token count");
    assert!(
        search_info.last_token_usage.total_tokens < 1_000,
        "active context must not include the repeated search passes: {search_info:?}"
    );
    assert_eq!(search_info.total_token_usage.total_tokens, 5_100);
    // Neither remote nor local compaction ran: no compact call and no extra model request.
    assert!(compact_mock.requests().is_empty());
    assert_eq!(request_log.requests().len(), 3);
    Ok(())
}
