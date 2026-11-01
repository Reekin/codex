use codex_history::LocalCompactionKind;
use codex_history::LocalCompactionSource;
use codex_history::ResponseItemEnvelope;
use pretty_assertions::assert_eq;
use serde_json::json;

use super::SourceRange;
use super::WindowPlan;
use crate::Removal;
use crate::remove_parts;
use codex_protocol::models::ResponseItem;
use std::collections::HashSet;

fn item(value: serde_json::Value) -> ResponseItemEnvelope {
    ResponseItemEnvelope::new(serde_json::from_value(value).unwrap())
}

fn message(id: &str, role: &str, text: &str) -> ResponseItemEnvelope {
    item(json!({"type":"message", "id":id, "role":role,
        "content":[{"type":"input_text", "text":text}]}))
}

fn instructions() -> ResponseItemEnvelope {
    item(json!({"type":"message", "id":"instructions", "role":"user",
        "content":[{"type":"input_text", "text":"Canonical instructions."}],
        "internal_chat_message_metadata_passthrough":{"content_item_kinds":["agents_md.instructions"]}}))
}

fn call(id: &str) -> ResponseItemEnvelope {
    item(
        json!({"type":"function_call", "id":format!("call_{id}"), "call_id":id,
        "name":"exec", "arguments":"{}"}),
    )
}

fn output(id: &str) -> ResponseItemEnvelope {
    item(
        json!({"type":"function_call_output", "id":format!("out_{id}"), "call_id":id,
        "output":"evidence"}),
    )
}

fn summary(first: &str, last: &str) -> ResponseItemEnvelope {
    let mut summary = message("summary", "user", "Earlier work.");
    summary.metadata.get_or_insert_default().local_compaction = Some(LocalCompactionSource {
        first_item_id: first.to_string(),
        last_item_id: last.to_string(),
        kind: LocalCompactionKind::OldestOverview,
    });
    summary
}

fn previous(mut item: ResponseItemEnvelope) -> ResponseItemEnvelope {
    item.metadata.get_or_insert_default().previous_window = true;
    item
}

fn range(first: &str, last: &str) -> Option<SourceRange> {
    Some(SourceRange {
        first_item_id: first.to_string(),
        last_item_id: last.to_string(),
    })
}

fn assistant(id: &str, text: &str) -> ResponseItemEnvelope {
    message(id, "assistant", text)
}

/// Records removed by the plan, by index.
fn removed(plan: &WindowPlan) -> Vec<usize> {
    plan.removed
        .iter()
        .enumerate()
        .filter_map(|(index, removed)| removed.then_some(index))
        .collect()
}

#[test]
fn first_window_within_budget_stays_verbatim_without_a_summary() {
    let items = vec![
        message("user_1", "user", "Investigate."),
        call("a"),
        output("a"),
        assistant("reply_1", "Found it."),
    ];
    let plan = WindowPlan::new(&items, &[10; 4], 100, 50).unwrap();
    assert_eq!(
        plan,
        WindowPlan {
            removed: vec![false; 4],
            window_start: 0,
            kept_context: Vec::new(),
            kept_input: None,
            summarized: None,
            projected_tokens: 40,
        }
    );
}

#[test]
fn earlier_summary_and_previous_window_are_summarized_and_the_current_window_kept() {
    let items = vec![
        previous(instructions()),
        summary("origin_1", "origin_9"),
        previous(message("user_1", "user", "Investigate.")),
        previous(call("a")),
        previous(output("a")),
        previous(assistant("reply_1", "Found it.")),
        message("user_2", "user", "Now fix it."),
        call("b"),
        output("b"),
    ];
    let plan = WindowPlan::new(&items, &[10; 9], 1_000, 50).unwrap();
    assert_eq!(
        plan,
        WindowPlan {
            removed: [vec![true; 6], vec![false; 3]].concat(),
            window_start: 6,
            kept_context: Vec::new(),
            kept_input: None,
            summarized: range("origin_1", "reply_1"),
            projected_tokens: 30 + 50,
        }
    );
}

#[test]
fn an_active_input_ahead_of_the_window_is_kept() {
    let items = vec![
        previous(message("user_1", "user", "Keep going.")),
        previous(call("a")),
        previous(output("a")),
        call("b"),
        output("b"),
    ];
    let plan = WindowPlan::new(&items, &[10; 5], 1_000, 50).unwrap();
    assert_eq!(
        (removed(&plan), plan.kept_input, plan.summarized),
        (vec![1, 2], Some(0), range("call_a", "out_a"))
    );
}

#[test]
fn earlier_turns_give_way_tool_records_then_interim_replies_then_whole_turns() {
    let items = vec![
        message("user_1", "user", "Investigate."),
        call("a"),
        output("a"),
        assistant("interim_1", "Checking."),
        call("b"),
        output("b"),
        assistant("reply_1", "Found it."),
        message("user_2", "user", "Fix it."),
        call("c"),
        output("c"),
        assistant("reply_2", "Fixed."),
        message("user_3", "user", "Verify."),
        call("d"),
        output("d"),
    ];
    let costs: Vec<_> = items
        .iter()
        .map(|item| {
            if matches!(item.item, ResponseItem::FunctionCallOutput { .. }) {
                100
            } else {
                10
            }
        })
        .collect();
    // Tool records go oldest first, until the rest and the summary fit.
    let plan = WindowPlan::new(&items, &costs, 400, 50).unwrap();
    assert_eq!(
        (removed(&plan), plan.summarized, plan.projected_tokens),
        (vec![1, 2, 4, 5], range("call_a", "out_b"), 280 + 50)
    );
    let plan = WindowPlan::new(&items, &costs, 230, 50).unwrap();
    assert_eq!(removed(&plan), vec![1, 2, 4, 5, 8, 9]);
    // Then intermediate replies, then whole turns; the current turn's input and newest work
    // group stay even when the rest cannot fit.
    let plan = WindowPlan::new(&items, &costs, 210, 50).unwrap();
    assert_eq!(removed(&plan), vec![1, 2, 3, 4, 5, 8, 9]);
    let plan = WindowPlan::new(&items, &costs, 150, 50).unwrap();
    assert_eq!(
        (removed(&plan), plan.projected_tokens),
        ((0..11).collect::<Vec<_>>(), 120 + 50)
    );
}

#[test]
fn the_current_turn_gives_way_last_outside_its_newest_work_group() {
    let items = vec![
        message("user_1", "user", "Work through it."),
        call("a"),
        output("a"),
        assistant("interim_1", "Next."),
        call("b"),
        output("b"),
        assistant("interim_2", "Next."),
        call("c"),
        output("c"),
    ];
    let costs = [10, 10, 100, 10, 10, 100, 10, 10, 100];
    let plan = WindowPlan::new(&items, &costs, 200, 50).unwrap();
    assert_eq!(
        (removed(&plan), plan.projected_tokens),
        (vec![1, 2, 4, 5], 140 + 50)
    );
}

#[test]
fn repeated_compaction_keeps_history_bounded() {
    let mut history = Vec::new();
    for round in 0..6 {
        history.push(message(&format!("user_{round}"), "user", "Continue."));
        for step in 0..3 {
            let id = format!("{round}_{step}");
            history.push(call(&id));
            history.push(output(&id));
        }
        history.push(assistant(&format!("reply_{round}"), "Progress."));
        let costs = vec![100; history.len()];
        let plan = WindowPlan::new(&history, &costs, 700, 200).unwrap();
        assert!(plan.projected_tokens <= 700, "round {round}: {plan:?}");
        let mut next: Vec<_> = plan
            .kept_context
            .iter()
            .map(|index| history[*index].clone())
            .collect();
        if let Some(range) = &plan.summarized {
            next.push(summary(&range.first_item_id, &range.last_item_id));
        }
        next.extend(plan.kept_input.map(|index| history[index].clone()));
        next.extend(
            history
                .iter()
                .zip(&plan.removed)
                .skip(plan.window_start)
                .filter(|(_, removed)| !**removed)
                .map(|(item, _)| previous(item.clone())),
        );
        history = next;
        assert!(
            history.len() <= 12,
            "round {round}: {} records",
            history.len()
        );
    }
}

#[test]
fn rendered_records_ahead_of_the_window_go_unsummarized_and_client_context_is_kept_once() {
    let client = |id: &str, text: &str| {
        item(json!({"type":"message", "id":id, "role":"developer",
            "content":[{"type":"input_text", "text":text}],
            "internal_chat_message_metadata_passthrough":{"content_item_kinds":["unknown"]}}))
    };
    let items = vec![
        previous(instructions()),
        previous(client("session_1", "Session A.")),
        previous(message("user_1", "user", "Investigate.")),
        previous(client("session_2", "Session A.")),
        previous(client("session_b", "Session B.")),
        message("user_2", "user", "Continue."),
        call("a"),
        output("a"),
    ];
    let plan = WindowPlan::new(&items, &[100; 8], 1_000, 50).unwrap();
    assert_eq!(
        plan,
        WindowPlan {
            removed: [vec![true; 3], vec![false; 5]].concat(),
            window_start: 5,
            kept_context: vec![3, 4],
            kept_input: None,
            summarized: range("user_1", "user_1"),
            projected_tokens: 500 + 50,
        }
    );
}

#[test]
fn removing_rendered_kinds_keeps_the_other_parts_of_a_message() {
    let mut combined: ResponseItem = serde_json::from_value(json!({"type":"message", "role":"user",
        "content":[{"type":"input_text", "text":"Rules."}, {"type":"input_text", "text":"Notice."}],
        "internal_chat_message_metadata_passthrough":{"content_item_kinds":["agents_md.instructions", "permissions.approved_command_prefix_saved"]}}))
    .unwrap();
    let kinds = HashSet::from(["agents_md.instructions".to_string()]);
    assert_eq!(remove_parts(&mut combined, &kinds), Removal::Partial);
    let expected: ResponseItem = serde_json::from_value(json!({"type":"message", "role":"user",
        "content":[{"type":"input_text", "text":"Notice."}],
        "internal_chat_message_metadata_passthrough":{"content_item_kinds":["permissions.approved_command_prefix_saved"]}}))
    .unwrap();
    assert_eq!(combined, expected);
    let kinds = HashSet::from(["permissions.approved_command_prefix_saved".to_string()]);
    assert_eq!(remove_parts(&mut combined, &kinds), Removal::All);
    assert_eq!(
        remove_parts(&mut message("user_1", "user", "Hi.").item, &kinds),
        Removal::Unchanged
    );
}
