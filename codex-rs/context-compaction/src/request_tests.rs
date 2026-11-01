use std::collections::HashSet;

use codex_history::ResponseItemEnvelope;
use codex_protocol::models::ResponseItem;
use pretty_assertions::assert_eq;
use serde_json::json;

use super::Batch;
use super::BatchLimits;
use super::CLASSIFY_MARKER;
use super::next_batch;
use super::summary_request;
use crate::StagedDecisions;
use crate::fragments::LocalCompactionRequest;

fn item(value: serde_json::Value) -> ResponseItemEnvelope {
    ResponseItemEnvelope::new(serde_json::from_value(value).unwrap())
}

/// Completed results of `sizes` bytes, then a newest protected work group.
fn history(sizes: &[usize]) -> Vec<ResponseItemEnvelope> {
    let mut items = vec![item(
        json!({"type":"message","id":"user_1","role":"user","content":[{"type":"input_text","text":"Investigate."}]}),
    )];
    for (index, size) in sizes.iter().enumerate() {
        items.push(item(json!({"type":"function_call","id":format!("call_{index}"),"call_id":format!("c{index}"),"name":"exec","arguments":"{}"})));
        items.push(item(json!({"type":"function_call_output","id":format!("result_{index}"),"call_id":format!("c{index}"),"output":"x".repeat(*size)})));
    }
    items.push(item(json!({"type":"function_call","id":"call_new","call_id":"c_new","name":"exec","arguments":"{}"})));
    items
}

/// Bytes as cost, so the fixture controls volume exactly.
fn cost(item: &ResponseItem) -> usize {
    serde_json::to_string(item).unwrap().len()
}

fn sent(batch: Batch) -> (Vec<String>, Option<HashSet<String>>) {
    match batch {
        Batch::Send { ids, backlog, .. } => (ids, backlog),
        other => panic!("expected a batch, got {other:?}"),
    }
}

#[test]
fn marking_starts_at_the_volume_share_and_ignores_record_counts() {
    let limits = BatchLimits {
        window_tokens: 100_000,
        start_percent: 5,
    };
    // Many small results stay below 5% of the window and start nothing.
    let small = history(&[100; 20]);
    assert!(matches!(
        next_batch(&small, &StagedDecisions::default(), None, limits, "", cost),
        Batch::Idle
    ));
    let large = history(&[3_000, 3_000]);
    let (ids, backlog) = sent(next_batch(
        &large,
        &StagedDecisions::default(),
        None,
        limits,
        "",
        cost,
    ));
    assert_eq!(ids, vec!["result_0", "result_1"]);
    assert_eq!(backlog, None);
}

#[test]
fn a_backlog_is_worked_through_oldest_first_and_a_small_remainder_waits() {
    let limits = BatchLimits {
        window_tokens: 100_000,
        start_percent: 1,
    };
    // Each worst-case replacement is capped at the fragment size, so half the response limit
    // holds about forty results.
    let source = history(&[4_000; 50]);
    let (ids, backlog) = sent(next_batch(
        &source,
        &StagedDecisions::default(),
        None,
        limits,
        "",
        cost,
    ));
    assert_eq!(ids.first().map(String::as_str), Some("result_0"));
    let backlog = backlog.expect("the rest waits in the backlog");
    assert_eq!(ids.len() + backlog.len(), 50);
    assert!(ids.iter().all(|id| !backlog.contains(id)));
    // Only the backlog is offered next; a remainder below 1% of the window is not sent.
    let small: HashSet<_> = ["result_49".to_string()].into();
    let tiny = BatchLimits {
        window_tokens: 10_000_000,
        start_percent: 1,
    };
    assert!(matches!(
        next_batch(
            &source,
            &StagedDecisions::default(),
            Some(small),
            tiny,
            "",
            cost
        ),
        Batch::Idle
    ));
}

#[test]
fn requests_carry_their_marker_and_stay_within_their_caps() {
    let source = history(&[3_000, 3_000]);
    let Batch::Send { request, .. } = next_batch(
        &source,
        &StagedDecisions::default(),
        None,
        BatchLimits {
            window_tokens: 1_000,
            start_percent: 5,
        },
        "Prefer dropping logs.",
        cost,
    ) else {
        panic!("expected a batch");
    };
    let text = format!("{request:?}");
    assert!(text.contains(CLASSIFY_MARKER));
    assert!(text.contains("\\\"call_id\\\":\\\"c0\\\""));
    assert!(summary_request(&"x".repeat(LocalCompactionRequest::MAX_BYTES)).is_err());
}
