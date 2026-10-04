use codex_history::ResponseItemEnvelope;
use codex_protocol::models::FunctionCallOutputBody;
use codex_protocol::models::ResponseItem;
use pretty_assertions::assert_eq;
use serde_json::json;

use crate::Budget;
use crate::StagedDecisions;
use crate::dropped_reasoning;
use crate::eligible_results;
use crate::large_calls;

fn item(value: serde_json::Value) -> ResponseItemEnvelope {
    ResponseItemEnvelope::new(serde_json::from_value(value).unwrap())
}

fn message(id: &str, role: &str, text: &str) -> ResponseItemEnvelope {
    item(
        json!({"type":"message","id":id,"role":role,"content":[{"type":"input_text","text":text}]}),
    )
}

fn history() -> Vec<ResponseItemEnvelope> {
    let mut items = vec![message(
        "user_1",
        "user",
        "Keep the database read-only; verify errors before reporting success.",
    )];
    for index in 0..4 {
        items.push(item(json!({"type":"function_call","id":format!("call_{index}"),"call_id":format!("c{index}"),"name":"exec","arguments":"{}"})));
        items.push(item(json!({"type":"function_call_output","id":format!("result_{index}"),"call_id":format!("c{index}"),"output":"long evidence ".repeat(200)})));
    }
    items.push(message(
        "assistant_1",
        "assistant",
        "Checking the newest result.",
    ));
    items
}

#[test]
fn cleanup_preserves_dialogue_pairs_ids_and_error_status_in_one_long_turn() {
    let mut original = history();
    if let ResponseItem::FunctionCallOutput { output, .. } = &mut original[4].item {
        output.success = Some(false);
    }
    let stage = StagedDecisions::parse(original.clone(), &json!({"decisions":[
        {"id":"result_0","action":"keep"},
        {"id":"result_1","action":"shorten","text":"Permission denied; operation was not verified."},
        {"id":"result_2","action":"drop"}
    ]}).to_string()).unwrap();
    let replacement = stage.apply(&original).unwrap();
    let mut expected = original.clone();
    for index in [4, 6] {
        expected[index] = replacement[index].clone();
    }
    assert_eq!(replacement, expected);
    assert_eq!(replacement[4].item.id(), original[4].item.id());
    match &replacement[4].item {
        ResponseItem::FunctionCallOutput { output, .. } => {
            assert_eq!(output.success, Some(false));
            assert_eq!(output.body, FunctionCallOutputBody::Text("Permission denied; operation was not verified.\n[Original item: result_1; use recall_read_item.]".to_string()));
        }
        _ => panic!("result changed type"),
    }
    assert_eq!(
        eligible_results(&original),
        vec!["result_0", "result_1", "result_2"]
    );
}

#[test]
fn invalid_analysis_is_rejected_as_a_whole() {
    let source = history();
    for decisions in [
        json!([{"id":"result_0","action":"drop"},{"id":"foreign","action":"keep"}]),
        json!([{"id":"result_0","action":"drop"},{"id":"result_0","action":"keep"}]),
        json!([{"id":"result_3","action":"drop"}]),
        json!([{"id":"result_0","action":"shorten","text":"x".repeat(3001)}]),
    ] {
        assert!(
            StagedDecisions::parse(source.clone(), &json!({"decisions":decisions}).to_string())
                .is_err()
        );
    }
    assert!(StagedDecisions::parse(source.clone(), "not JSON").is_err());
    assert_eq!(source, history());
}

#[test]
fn staged_decisions_preserve_new_outputs_and_user_direction() {
    let source = history();
    let stage = StagedDecisions::parse(source.clone(), &json!({"decisions":[
        {"id":"result_0","action":"drop"},{"id":"result_1","action":"keep"},{"id":"result_2","action":"keep"}
    ]}).to_string()).unwrap();
    let mut newer = source.clone();
    newer.push(message("assistant_2", "assistant", "Additional finding"));
    let replacement = stage.apply(&newer).unwrap();
    assert_eq!(replacement.last(), newer.last());
    newer.push(message("user_2", "user", "Keep all evidence now."));
    let replacement = stage.apply(&newer).unwrap();
    assert_eq!(replacement.last(), newer.last());
    assert_ne!(replacement[2], newer[2]);
}

#[test]
fn image_results_are_candidates_and_opaque_results_are_not() {
    let mut source = history();
    source[2] = item(
        json!({"type":"function_call_output","id":"result_0","call_id":"c0","output":[{"type":"input_image","image_url":"data:image/png;base64,AAAA"}]}),
    );
    source[4] = item(
        json!({"type":"function_call_output","id":"result_1","call_id":"c1","output":[{"type":"encrypted_content","encrypted_content":"opaque"}]}),
    );
    assert_eq!(eligible_results(&source), vec!["result_0", "result_2"]);
}

#[test]
fn cleanup_strips_images_and_summarizes_large_paired_calls() {
    let mut source = history();
    let screenshot = format!("data:image/png;base64,{}", "A".repeat(4000));
    source[2] = item(
        json!({"type":"function_call_output","id":"result_0","call_id":"c0","output":[
            {"type":"input_text","text":"Screenshot of the settings page."},
            {"type":"input_image","image_url":screenshot},
            {"type":"input_image","image_url":screenshot}
        ]}),
    );
    let patch = format!("*** Begin Patch\n{}*** End Patch", "+line\n".repeat(400));
    source[3] = item(
        json!({"type":"custom_tool_call","id":"call_1","call_id":"c1","name":"apply_patch","input":patch}),
    );
    source[4] = item(
        json!({"type":"custom_tool_call_output","id":"result_1","call_id":"c1","output":"long evidence ".repeat(200)}),
    );
    let script = json!({"code": "x".repeat(2000)}).to_string();
    source[5] = item(
        json!({"type":"function_call","id":"call_2","call_id":"c2","name":"js","arguments":script}),
    );
    assert_eq!(
        large_calls(&source),
        [("result_1".to_string(), 3), ("result_2".to_string(), 5)].into()
    );
    let stage = StagedDecisions::parse(source.clone(), &json!({"decisions":[
        {"id":"result_0","action":"drop"},
        {"id":"result_1","action":"shorten","text":"Patch applied.","call_text":"Rewrote config.toml"},
        {"id":"result_2","action":"drop"}
    ]}).to_string()).unwrap();
    let replacement = stage.apply(&source).unwrap();

    let ResponseItem::FunctionCallOutput { output, .. } = &replacement[2].item else {
        panic!("tool result")
    };
    assert_eq!(
        output.body,
        FunctionCallOutputBody::Text(
            "[Output omitted with 2 images; original item: result_0; use recall_read_item.]"
                .to_string()
        )
    );
    let ResponseItem::CustomToolCall { input, call_id, .. } = &replacement[3].item else {
        panic!("custom call")
    };
    assert_eq!(
        (input.as_str(), call_id.as_str()),
        (
            "Rewrote config.toml\n[Original item: call_1; use recall_read_item.]",
            "c1"
        )
    );
    // Function arguments stay a JSON object so providers can still parse the call.
    let ResponseItem::FunctionCall { arguments, .. } = &replacement[5].item else {
        panic!("function call")
    };
    assert_eq!(
        serde_json::from_str::<serde_json::Value>(arguments).unwrap(),
        json!({"summary": "[Arguments omitted; original item: call_2; use recall_read_item.]"})
    );
    for index in [2, 4, 6] {
        assert!(
            replacement[index]
                .metadata
                .as_ref()
                .is_some_and(|metadata| metadata.local_compaction.is_some())
        );
    }
    // The optimistic bound counts the same image and call reductions.
    let bound = StagedDecisions::default().optimistic_replacement(&source);
    assert!(
        matches!(&bound[3].item, ResponseItem::CustomToolCall { input, .. } if input.len() < 200)
    );
    assert!(
        matches!(&bound[2].item, ResponseItem::FunctionCallOutput { output, .. } if output.content_items().is_none())
    );

    let oversized = json!({"decisions":[
        {"id":"result_0","action":"keep"},{"id":"result_2","action":"keep"},
        {"id":"result_1","action":"drop","call_text":"x".repeat(401)}
    ]})
    .to_string();
    assert!(StagedDecisions::parse(source, &oversized).is_err());
}

#[test]
fn new_classification_batches_extend_staged_decisions_without_live_edits() {
    let source = history();
    let mut stage = StagedDecisions::parse_candidates(
        source.clone(),
        &["result_0".to_string()],
        &json!({"decisions":[{"id":"result_0","action":"drop"}]}).to_string(),
    )
    .unwrap();
    let additional = StagedDecisions::parse_candidates(source.clone(), &["result_1".to_string()],
        &json!({"decisions":[{"id":"result_1","action":"shorten","text":"Verified relevant evidence."}]}).to_string()).unwrap();
    stage.merge(additional);
    let cleaned = stage.apply(&source).unwrap();
    assert_ne!(cleaned[2], source[2]);
    assert_ne!(cleaned[4], source[4]);
    assert_eq!(&cleaned[5..], &source[5..]);
    assert_eq!(source, history());
}

#[test]
fn unresolved_group_blocks_promotion_across_its_start() {
    let mut source = history();
    source.remove(2);
    assert_eq!(eligible_results(&source), Vec::<String>::new());
    assert_eq!(crate::groups::safe_cuts(&source), vec![0, 1]);
}

#[test]
fn decisions_survive_disjoint_cleanup_but_changed_sources_stay_unmarked() {
    let source = history();
    let mut stage = StagedDecisions::parse(
        source.clone(),
        &json!({"decisions":[
            {"id":"result_0","action":"keep"}, {"id":"result_1","action":"drop"},
            {"id":"result_2","action":"shorten","text":"Existing concise evidence"}
        ]})
        .to_string(),
    )
    .unwrap();
    assert!(stage.contains(&source[2]));
    let mut newer = source.clone();
    newer.push(message(
        "assistant_growth",
        "assistant",
        "More completed analysis.",
    ));
    let first_view = stage.apply(&source).unwrap();
    newer[4] = first_view[4].clone();
    stage.retain_current(&newer);
    assert!(stage.contains(&newer[2]));
    assert!(!stage.contains(&newer[4]));
    let next_view = stage.apply(&newer).unwrap();
    assert_eq!(&next_view[..source.len()], &first_view);
    assert_eq!(next_view.last(), newer.last());
    assert!(stage.contains(&source[2]));
    if let ResponseItem::FunctionCallOutput { output, .. } = &mut newer[6].item {
        output.body = FunctionCallOutputBody::Text("changed source content".to_string());
    }
    assert_eq!(stage.apply(&newer).unwrap()[6], newer[6]);
}

#[test]
fn optimistic_bound_counts_protected_output_but_excludes_known_keeps_and_calls() {
    let source = history();
    let stage = StagedDecisions::parse_candidates(
        source.clone(),
        &["result_0".to_string()],
        &json!({"decisions":[{"id":"result_0","action":"keep"}]}).to_string(),
    )
    .unwrap();
    let optimistic = stage.optimistic_replacement(&source);
    assert_eq!(&optimistic[..3], &source[..3]);
    for index in [3, 5, 7, 9] {
        assert_eq!(optimistic[index], source[index]);
    }
    assert_ne!(optimistic[8], source[8]);
    assert!(!eligible_results(&source).contains(&"result_3".to_string()));
}

#[test]
fn completed_batch_merges_after_a_disjoint_cleanup_and_new_input() {
    let source = history();
    let mut stage = StagedDecisions::parse_candidates(
        source.clone(),
        &["result_0".to_string()],
        &json!({"decisions":[{"id":"result_0","action":"drop"}]}).to_string(),
    )
    .unwrap();
    let inflight = StagedDecisions::parse_candidates(
        source.clone(),
        &["result_1".to_string()],
        &json!({"decisions":[{"id":"result_1","action":"drop"}]}).to_string(),
    )
    .unwrap();
    let mut current = stage.apply(&source).unwrap();
    current.push(message("new_user", "user", "Continue"));
    stage.merge(inflight);
    stage.retain_current(&current);
    let result = stage.apply(&current).unwrap();
    assert_eq!(result[2], current[2]);
    assert_ne!(result[4], current[4]);
    assert_eq!(result.last(), current.last());
}

#[test]
fn cleanup_uses_window_percentage_points_and_optimistic_headroom() {
    let budget = Budget {
        window_tokens: 20_000,
        fixed_tokens: 4_000,
        reclaim_percent: 30,
        compact_target_percent: 30,
        keep_reasoning_percent: 5,
    };
    assert_eq!(budget.history_target(), 2_000);
    assert_eq!(budget.reasoning_tokens(), 1_000);
    assert_eq!(budget.required_savings(), 6_000);
    // Fixed 4k + history6k=50%; a drop to total30% only saves20 percentage points.
    assert!(!budget.useful(6_000, 2_000));
    // Fixed4k + history10k=70%; total40% saves the required30 percentage points.
    assert!(budget.useful(10_000, 4_000));
    assert!(budget.can_reach(2_000, 2_000, 14_000, 18_000));
    assert!(!budget.can_reach(1_000, 2_000, 16_000, 18_000));
    assert!(budget.can_reach(1_000, 2_000, 16_000, 20_000));
}

#[test]
fn reasoning_trim_keeps_the_current_turn_and_the_newest_earlier_reasoning() {
    let reasoning = |id: &str| {
        item(
            json!({"type":"reasoning","id":id,"summary":[],"content":[{"type":"reasoning_text","text":"thinking"}],"encrypted_content":""}),
        )
    };
    let items = vec![
        message("user_1", "user", "First request."),
        reasoning("old"),
        message("assistant_1", "assistant", "Done."),
        message("user_2", "user", "Second request."),
        reasoning("previous"),
        message("assistant_2", "assistant", "Done again."),
        message("user_3", "user", "Third request."),
        reasoning("current"),
        reasoning("current_more"),
    ];
    let costs = vec![100; items.len()];
    // The current turn exceeds the budget alone and is kept; earlier reasoning beyond it goes.
    assert_eq!(
        dropped_reasoning(&items, 150, &costs),
        vec![false, true, false, false, true, false, false, false, false]
    );
    assert_eq!(
        dropped_reasoning(&items, 300, &costs),
        vec![false, true, false, false, false, false, false, false, false]
    );
    assert_eq!(
        dropped_reasoning(&items, usize::MAX, &costs),
        vec![false; items.len()]
    );
}
