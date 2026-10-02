use codex_history::LocalCompactionKind;
use codex_history::ResponseItemEnvelope;
use codex_protocol::models::FunctionCallOutputBody;
use codex_protocol::models::ResponseItem;
use pretty_assertions::assert_eq;
use serde_json::json;

use crate::Budget;
use crate::StagedDecisions;
use crate::TierPlan;
use crate::eligible_results;

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
fn staged_decisions_preserve_new_outputs_and_invalidate_on_user_direction() {
    let source = history();
    let stage = StagedDecisions::parse(source.clone(), &json!({"decisions":[
        {"id":"result_0","action":"drop"},{"id":"result_1","action":"keep"},{"id":"result_2","action":"keep"}
    ]}).to_string()).unwrap();
    let mut newer = source.clone();
    newer.push(message("assistant_2", "assistant", "Additional finding"));
    let replacement = stage.apply(&newer).unwrap();
    assert_eq!(replacement.last(), newer.last());
    newer.push(message("user_2", "user", "Keep all evidence now."));
    assert!(stage.apply(&newer).is_err());
}

#[test]
fn opaque_and_multimodal_results_are_not_classifier_candidates() {
    let mut source = history();
    source[2] = item(
        json!({"type":"function_call_output","id":"result_0","call_id":"c0","output":[{"type":"input_image","image_url":"data:image/png;base64,AAAA"}]}),
    );
    source[4] = item(
        json!({"type":"function_call_output","id":"result_1","call_id":"c1","output":[{"type":"encrypted_content","encrypted_content":"opaque"}]}),
    );
    assert_eq!(eligible_results(&source), vec!["result_2"]);
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
    stage.merge(additional).unwrap();
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
fn kept_results_can_be_reassessed_after_append_only_growth() {
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
    assert_eq!(stage.kept_ids(), vec!["result_0"]);
    let mut newer = source.clone();
    newer.push(message(
        "assistant_growth",
        "assistant",
        "More completed analysis.",
    ));
    let reassessed = StagedDecisions::parse_candidates(newer.clone(), &stage.kept_ids(),
        &json!({"decisions":[{"id":"result_0","action":"shorten","text":"Now only this fact remains relevant"}]}).to_string()).unwrap();
    let first_view = stage.apply(&source).unwrap();
    stage.merge(reassessed).unwrap();
    let next_view = stage.apply(&newer).unwrap();
    assert_ne!(next_view[2], first_view[2]);
    assert_eq!(&next_view[3..source.len()], &first_view[3..]);
    assert_eq!(next_view.last(), newer.last());
    assert_eq!(stage.kept_ids(), Vec::<String>::new());
}

#[test]
fn budget_includes_fixed_context_and_batches_meaningful_savings() {
    let budget = Budget {
        window_tokens: 20_000,
        fixed_tokens: 4_000,
        trigger_percent: 50,
        target_percent: 30,
        minimum_savings_percent: 5,
    };
    assert!(budget.should_analyze(6_000));
    assert!(!budget.should_analyze(5_999));
    assert_eq!(budget.history_target(), 2_000);
    assert!(budget.useful(6_000, 2_000));
    assert!(!budget.useful(6_000, 5_001));
}

#[test]
fn repeated_tiers_merge_old_ranges_and_bound_the_ledger() {
    let mut source = history();
    for round in 0..3 {
        // Older groups cost enough to require promotion while the newest group fits.
        let mut costs = vec![1_000; source.len()];
        let len = costs.len();
        costs[len - 3..].fill(20);
        costs[0] = 20;
        let plan = TierPlan::new(&source, 2_000, &costs).unwrap();
        let fragments = plan.parse(&json!({
            "l2": if plan.l2.is_some() { "Older dialogue, with failures still unverified." } else { "" },
            "l3": if plan.l3.is_some() { "Prior investigation overview; exact evidence remains archived." } else { "" },
            "ledger": "Database stays read-only; verify errors before reporting success."
        }).to_string()).unwrap();
        assert!(
            plan.parse(
                &json!({"l2":"x".repeat(plan.max_fragment_bytes + 1),"l3":"x","ledger":"x"})
                    .to_string()
            )
            .is_err()
        );
        let mut next = plan.retained_prefix;
        for (index, fragment) in fragments.into_iter().enumerate() {
            let mut envelope = message(&format!("summary_{round}_{index}"), "user", &fragment.text);
            envelope.metadata.get_or_insert_default().local_compaction = Some(fragment.source);
            next.push(envelope);
        }
        next.extend(plan.retained_tail);
        assert_eq!(
            next.iter()
                .filter(|item| item
                    .metadata
                    .as_ref()
                    .and_then(|metadata| metadata.local_compaction.as_ref())
                    .is_some_and(|source| source.kind == LocalCompactionKind::ConstraintsLedger))
                .count(),
            1
        );
        let active_user = source
            .iter()
            .rfind(|item| crate::is_user_direction(&item.item))
            .unwrap();
        assert!(next.contains(active_user));
        next.push(message(
            &format!("user_next_{round}"),
            "user",
            "Continue the read-only investigation.",
        ));
        source = next;
    }
}
