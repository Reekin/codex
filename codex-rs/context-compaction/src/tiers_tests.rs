use codex_history::LocalCompactionKind;
use codex_history::ResponseItemEnvelope;
use pretty_assertions::assert_eq;
use serde_json::json;

use super::TierPlan;
use super::is_summary;

fn item(value: serde_json::Value) -> ResponseItemEnvelope {
    ResponseItemEnvelope::new(serde_json::from_value(value).unwrap())
}

fn message(id: &str, role: &str, text: &str) -> ResponseItemEnvelope {
    item(json!({"type":"message", "id":id, "role":role,
        "content":[{"type":"input_text", "text":text}]}))
}

fn install(plan: TierPlan, round: usize) -> Vec<ResponseItemEnvelope> {
    let fragments = plan.parse(&json!({
        "l2": if plan.l2.is_some() { "Recent findings remain provisional." } else { "" },
        "l3": if plan.l3.is_some() { "Prior investigation; originals remain available." } else { "" },
        "ledger": "Keep the database read-only."
    }).to_string()).unwrap();
    let mut next = plan.retained_prefix;
    for (index, fragment) in fragments.into_iter().enumerate() {
        let mut envelope = message(&format!("summary_{round}_{index}"), "user", &fragment.text);
        envelope.metadata.get_or_insert_default().local_compaction = Some(fragment.source);
        next.push(envelope);
    }
    next.extend(plan.retained_tail);
    next
}

#[test]
fn newest_completed_group_and_long_active_turn_are_promotable() {
    let active = message(
        "active",
        "user",
        "Investigate without changing the database.",
    );
    let mut history = vec![active.clone()];
    history.extend((0..400).map(|index| {
        message(
            &format!("analysis_{index}"),
            "assistant",
            "Completed investigation step.",
        )
    }));
    history.push(item(
        json!({"type":"function_call", "id":"call", "call_id":"c",
        "name":"exec", "arguments":"{}"}),
    ));
    history.push(item(
        json!({"type":"function_call_output", "id":"result", "call_id":"c",
        "output":"large completed evidence"}),
    ));
    let mut costs = vec![1_000; history.len()];
    costs[0] = 100;
    *costs.last_mut().unwrap() = 40_000;
    let plan = TierPlan::new(&history, 4_000, 12_000, &costs).unwrap();
    assert_eq!(plan.retained_prefix, vec![active]);
    assert_eq!(plan.retained_tail, Vec::<ResponseItemEnvelope>::new());
    assert_eq!(plan.l2.as_ref().unwrap().last_item_id, "result");
    assert!(plan.projected_tokens <= 4_000);
    assert!(plan.max_fragment_bytes <= 3_000);
}

#[test]
fn repeated_promotions_bound_verbatim_and_merge_old_summary_ranges() {
    let active = message("active", "user", "Keep the database read-only.");
    let mut history = vec![active.clone()];
    for round in 0..20 {
        history.extend((0..80).map(|index| {
            message(
                &format!("message_{round}_{index}"),
                "assistant",
                "Completed work.",
            )
        }));
        let costs = vec![100; history.len()];
        let plan = TierPlan::new(&history, 3_000, 10_000, &costs).unwrap();
        assert!(plan.retained_tokens <= 1_100);
        assert!(plan.projected_tokens <= 3_000);
        assert_eq!(plan.l3.as_ref().unwrap().first_item_id, "message_0_0");
        assert!(
            !plan
                .retained_prefix
                .iter()
                .chain(&plan.retained_tail)
                .any(is_summary)
        );
        history = install(plan, round);
        assert!(history.contains(&active));
        assert!(history.len() <= 14);
        assert_eq!(
            history
                .iter()
                .filter(|entry| entry
                    .metadata
                    .as_ref()
                    .and_then(|metadata| metadata.local_compaction.as_ref())
                    .is_some_and(|source| source.kind == LocalCompactionKind::ConstraintsLedger))
                .count(),
            1
        );
    }
}

#[test]
fn canonical_instructions_and_unresolved_calls_survive_exactly() {
    let canonical = vec![
        message("system", "system", "Canonical system instruction."),
        message("developer", "developer", "Canonical developer instruction."),
        item(json!({"type":"message", "id":"agents", "role":"user",
            "content":[{"type":"input_text", "text":"Canonical project instruction."}],
            "internal_chat_message_metadata_passthrough":{
                "content_item_kinds":["agents_md.instructions"]}})),
    ];
    let active = message("active", "user", "Continue investigating.");
    let pending = vec![
        item(
            json!({"type":"function_call", "id":"pending", "call_id":"c",
            "name":"exec", "arguments":"{}"}),
        ),
        message("after_call", "assistant", "Waiting for the result."),
    ];
    let mut history = canonical.clone();
    history.push(active.clone());
    history.extend(
        (0..50).map(|index| message(&format!("old_{index}"), "assistant", "Previous finding.")),
    );
    history.extend(pending.clone());
    let mut costs = vec![1_000; history.len()];
    costs[..4].fill(10);
    let len = costs.len();
    costs[len - 2..].fill(100);
    let plan = TierPlan::new(&history, 2_000, 5_000, &costs).unwrap();
    let mut expected = canonical;
    expected.push(active);
    assert_eq!(plan.retained_prefix, expected);
    assert_eq!(plan.retained_tail, pending);
    assert_eq!(plan.retained_tokens, 240);
}

#[test]
fn above_ideal_active_input_uses_a_safe_chosen_budget() {
    let history = vec![
        message("old", "assistant", "Old completed work."),
        message("active", "user", "Large active user input."),
    ];
    let plan = TierPlan::new(&history, 2_000, 9_000, &[40_000, 5_000]).unwrap();
    assert!(plan.result_budget_tokens > 2_000);
    assert!(plan.projected_tokens <= plan.result_budget_tokens);
    assert!(plan.result_budget_tokens <= plan.hard_cap_tokens);
    assert!(plan.max_fragment_bytes > 128);
    let evidence = "Verified finding. ".repeat(30);
    assert!(
        plan.parse(&json!({"l2": evidence, "l3": "", "ledger": evidence}).to_string())
            .is_ok()
    );
    assert_eq!(plan.retained_tail, vec![history[1].clone()]);
    assert_eq!(plan.retained_tokens, 5_000);
    assert!(TierPlan::new(&history, 2_000, 4_999, &[40_000, 5_000]).is_err());
}

#[test]
fn history_below_preferred_budget_needs_no_summary() {
    let history = vec![
        message("user", "user", "Short instruction."),
        message("answer", "assistant", "Short answer."),
    ];
    let plan = TierPlan::new(&history, 2_000, 8_000, &[20, 20]).unwrap();
    assert_eq!((plan.l2.is_none(), plan.l3.is_none()), (true, true));
    assert_eq!(plan.retained_tail, history);
    assert_eq!(plan.projected_tokens, 40);
}

#[test]
fn active_only_input_can_fit_above_the_ideal_without_a_summary() {
    let history = vec![message("active", "user", "Irreducible active input.")];
    let plan = TierPlan::new(&history, 2_000, 8_000, &[5_000]).unwrap();
    assert_eq!((plan.l2.is_none(), plan.l3.is_none()), (true, true));
    assert_eq!(plan.result_budget_tokens, 5_000);
    assert_eq!(plan.retained_tail, history);
    assert!(
        plan.parse(r#"{"l2":"","l3":"","ledger":""}"#)
            .unwrap()
            .is_empty()
    );
    assert!(TierPlan::new(&history, 2_000, 4_000, &[5_000]).is_err());
}

#[test]
fn summary_fields_obey_byte_budgets_including_endpoint_overhead() {
    let history = vec![
        message(&"old".repeat(200), "assistant", "Old work."),
        message("active", "user", "Continue."),
    ];
    let plan = TierPlan::new(&history, 5_000, 8_000, &[20_000, 100]).unwrap();
    let text = "界".repeat(plan.max_fragment_bytes / 3);
    let output = json!({"l2":text, "l3":"", "ledger":text});
    assert!(plan.parse(&output.to_string()).is_ok());
    assert!(plan.projected_tokens <= 5_000);
    for field in ["l2", "l3", "ledger"] {
        let mut oversized = output.clone();
        oversized[field] = json!("x".repeat(plan.max_fragment_bytes + 1));
        assert!(plan.parse(&oversized.to_string()).is_err());
    }
}
