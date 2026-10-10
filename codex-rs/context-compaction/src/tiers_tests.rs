use codex_history::LocalCompactionKind;
use codex_history::LocalCompactionSource;
use codex_history::ResponseItemEnvelope;
use pretty_assertions::assert_eq;
use serde_json::json;

use super::SourceRange;
use super::WindowPlan;

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

#[test]
fn first_window_within_budget_stays_verbatim_without_a_summary() {
    let items = vec![
        instructions(),
        message("user_1", "user", "Investigate."),
        call("a"),
        output("a"),
        message("reply_1", "assistant", "Found it."),
    ];
    let plan = WindowPlan::new(&items, &[10; 5], 100, 50).unwrap();
    assert_eq!(
        plan,
        WindowPlan {
            cut: 0,
            kept_instructions: Vec::new(),
            superseded: Vec::new(),
            kept_input: None,
            summarized: None,
            projected_tokens: 50,
        }
    );
}

#[test]
fn earlier_summary_and_previous_window_are_summarized_and_the_current_window_kept() {
    let items = vec![
        instructions(),
        summary("origin_1", "origin_9"),
        previous(message("user_1", "user", "Investigate.")),
        previous(call("a")),
        previous(output("a")),
        previous(message("reply_1", "assistant", "Found it.")),
        message("user_2", "user", "Now fix it."),
        call("b"),
        output("b"),
    ];
    let plan = WindowPlan::new(&items, &[10; 9], 1_000, 50).unwrap();
    assert_eq!(
        plan,
        WindowPlan {
            cut: 6,
            kept_instructions: vec![0],
            superseded: Vec::new(),
            kept_input: None,
            summarized: range("origin_1", "reply_1"),
            projected_tokens: 10 + 30 + 50,
        }
    );
}

#[test]
fn an_oversized_window_summarizes_its_oldest_part_but_keeps_the_active_input() {
    let items = vec![
        message("user_1", "user", "Investigate everything."),
        call("a"),
        output("a"),
        message("reply_1", "assistant", "Next."),
        call("b"),
        output("b"),
        call("pending"),
    ];
    let costs = [10, 10, 1_000, 10, 10, 1_000, 10];
    let plan = WindowPlan::new(&items, &costs, 1_100, 50).unwrap();
    assert_eq!(
        plan,
        WindowPlan {
            cut: 3,
            kept_instructions: Vec::new(),
            superseded: Vec::new(),
            kept_input: Some(0),
            summarized: range("call_a", "out_a"),
            projected_tokens: 10 + 1_030 + 50,
        }
    );
    // Nothing fits: the latest cut is used, and the unresolved call is never summarized.
    assert_eq!(WindowPlan::new(&items, &costs, 10, 50).unwrap().cut, 6);
}

#[test]
fn repeated_compaction_keeps_history_bounded() {
    let mut history = vec![instructions()];
    for round in 0..6 {
        history.push(message(&format!("user_{round}"), "user", "Continue."));
        for step in 0..3 {
            let id = format!("{round}_{step}");
            history.push(call(&id));
            history.push(output(&id));
        }
        history.push(message(&format!("reply_{round}"), "assistant", "Progress."));
        let costs = vec![100; history.len()];
        let plan = WindowPlan::new(&history, &costs, 1_000, 200).unwrap();
        assert!(plan.projected_tokens <= 1_000, "round {round}: {plan:?}");
        let mut next: Vec<_> = plan
            .kept_instructions
            .iter()
            .map(|index| history[*index].clone())
            .collect();
        if let Some(range) = plan.summarized {
            next.push(summary(&range.first_item_id, &range.last_item_id));
        }
        next.extend(plan.kept_input.map(|index| history[index].clone()));
        next.extend(history[plan.cut..].iter().cloned().map(previous));
        history = next;
        assert!(
            history.len() <= 12,
            "round {round}: {} records",
            history.len()
        );
    }
}

#[test]
fn only_the_newest_copy_of_each_instruction_is_kept() {
    let typed = |id: &str, text: &str| {
        item(json!({"type":"message", "id":id, "role":"user",
            "content":[{"type":"input_text", "text":text}],
            "internal_chat_message_metadata_passthrough":{"content_item_kinds":["agents_md.instructions"]}}))
    };
    let untyped = |id: &str, text: &str| {
        item(json!({"type":"message", "id":id, "role":"developer",
            "content":[{"type":"input_text", "text":text}],
            "internal_chat_message_metadata_passthrough":{"content_item_kinds":["unknown"]}}))
    };
    let items = vec![
        typed("agents_1", "First rules."),
        untyped("session_1", "Session A."),
        message("user_1", "user", "Investigate."),
        untyped("session_2", "Session A."),
        untyped("session_b", "Session B."),
        typed("agents_2", "Updated rules."),
        message("user_2", "user", "Continue."),
        call("a"),
        output("a"),
    ];
    // Superseded copies cost nothing, wherever the cut falls.
    let plan = WindowPlan::new(&items, &[100; 9], 1_000, 50).unwrap();
    assert_eq!(
        plan,
        WindowPlan {
            cut: 0,
            kept_instructions: Vec::new(),
            superseded: vec![0, 1],
            kept_input: None,
            summarized: None,
            projected_tokens: 700,
        }
    );
    // Older copies ahead of the cut are neither kept nor summarized.
    let plan = WindowPlan::new(&items, &[100; 9], 650, 50).unwrap();
    assert_eq!(
        plan,
        WindowPlan {
            cut: 3,
            kept_instructions: Vec::new(),
            superseded: vec![0, 1],
            kept_input: None,
            summarized: range("user_1", "user_1"),
            projected_tokens: 600 + 50,
        }
    );
}
