//! Verifies observed Responses API usage is durably recorded in rollout history.

use anyhow::Result;
use codex_core::TurnInputRequest;
use codex_history::RolloutItem;
use codex_protocol::SessionId;
use codex_protocol::protocol::EventMsg;
use codex_protocol::protocol::TokenUsageInfo;
use codex_protocol::protocol::TokenUsageRecord;
use codex_protocol::user_input::UserInput;
use core_test_support::responses::ev_assistant_message;
use core_test_support::responses::ev_completed_with_tokens;
use core_test_support::responses::ev_function_call;
use core_test_support::responses::ev_reasoning_item;
use core_test_support::responses::ev_response_created;
use core_test_support::responses::ev_web_search_call_done;
use core_test_support::responses::mount_sse_sequence;
use core_test_support::responses::sse;
use core_test_support::responses::start_mock_server;
use core_test_support::skip_if_no_network;
use core_test_support::test_codex::TestCodex;
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

/// Runs one user turn per prompt and returns the last token usage reported during each turn.
async fn run_turns(test: &TestCodex, prompts: &[&str]) -> Result<Vec<Option<TokenUsageInfo>>> {
    let mut per_turn = Vec::new();
    for prompt in prompts {
        test.codex
            .start_or_steer_turn(TurnInputRequest::user_input(vec![UserInput::Text {
                text: (*prompt).into(),
                text_elements: Vec::new(),
            }]))
            .await?;
        let mut last_info = None;
        loop {
            match wait_for_event(&test.codex, |event| {
                matches!(event, EventMsg::TurnComplete(_) | EventMsg::TokenCount(_))
            })
            .await
            {
                EventMsg::TokenCount(count) => last_info = count.info.or(last_info),
                EventMsg::TurnComplete(_) => break,
                _ => {}
            }
        }
        per_turn.push(last_info);
    }
    Ok(per_turn)
}

fn compaction_request_count(request_log: &core_test_support::responses::ResponseMock) -> usize {
    request_log
        .requests()
        .iter()
        .filter(|request| !request.inputs_of_type("compaction_trigger").is_empty())
        .count()
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
async fn active_context_uses_provider_usage_without_reasoning_surcharge() -> Result<()> {
    skip_if_no_network!(Ok(()));

    let server = start_mock_server().await;
    // Encrypted reasoning retained from earlier turns is already part of the provider total.
    let request_log = mount_sse_sequence(
        &server,
        vec![
            sse(vec![
                ev_reasoning_item("pre-reasoning", &["pre"], &[&"a".repeat(2_400)]),
                ev_assistant_message("m1", "FIRST_FINAL"),
                ev_completed_with_tokens("r1", /*total_tokens*/ 10),
            ]),
            sse(vec![
                ev_reasoning_item("post-reasoning", &["post"], &[&"b".repeat(4_000)]),
                ev_assistant_message("m2", "SECOND_FINAL"),
                ev_completed_with_tokens("r2", /*total_tokens*/ 80),
            ]),
            sse(vec![
                ev_assistant_message("m3", "THIRD_FINAL"),
                ev_completed_with_tokens("r3", /*total_tokens*/ 90),
            ]),
        ],
    )
    .await;
    let test = test_codex()
        .with_config(|config| config.model_auto_compact_token_limit = Some(300))
        .build_with_auto_env(&server)
        .await?;

    let per_turn = run_turns(&test, &["FIRST_USER", "SECOND_USER", "THIRD_USER"]).await?;

    // Each turn ends on a model-generated message, so nothing is pending on top of the
    // provider total and the reported active context equals it exactly.
    let reported = per_turn
        .iter()
        .map(|info| {
            info.as_ref()
                .and_then(|info| info.context_usage.as_ref())
                .map(|usage| usage.tokens)
        })
        .collect::<Vec<_>>();
    assert_eq!(reported, vec![Some(10), Some(80), Some(90)]);
    let limit = per_turn[2]
        .as_ref()
        .and_then(|info| info.context_usage.as_ref())
        .and_then(|usage| usage.auto_compact_token_limit);
    assert!(
        limit.is_some_and(|limit| limit >= 300),
        "configured trigger should be reported: {limit:?}"
    );
    // Re-estimating the first turn's reasoning would cross the limit before the third turn.
    assert_eq!(compaction_request_count(&request_log), 0);
    assert_eq!(request_log.requests().len(), 3);
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
    let test = test_codex()
        .with_config(|config| config.model_auto_compact_token_limit = Some(1_000))
        .build_with_auto_env(&server)
        .await?;

    let per_turn = run_turns(&test, &["FIRST_USER", "SEARCH_USER", "THIRD_USER"]).await?;

    let search_info = per_turn[1].clone().expect("search turn token count");
    assert!(
        search_info.last_token_usage.total_tokens < 1_000,
        "active context must not include the repeated search passes: {search_info:?}"
    );
    assert_eq!(
        search_info.context_usage.map(|usage| usage.tokens),
        Some(search_info.last_token_usage.total_tokens)
    );
    assert_eq!(search_info.total_token_usage.total_tokens, 5_100);
    // Neither remote nor local compaction ran: no compaction request and no extra model request.
    assert_eq!(compaction_request_count(&request_log), 0);
    assert_eq!(request_log.requests().len(), 3);
    Ok(())
}
