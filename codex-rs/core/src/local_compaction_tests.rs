use super::insufficient_room;
use codex_context_compaction::Budget;
use pretty_assertions::assert_eq;

#[test]
fn exhausted_target_budget_defers_until_the_mandatory_input_limit() {
    let budget = Budget {
        window_tokens: 20_000,
        fixed_tokens: 6_000,
        trigger_percent: 50,
        target_percent: 30,
        minimum_savings_percent: 5,
    };
    assert_eq!(budget.history_target(), 0);
    assert!(!insufficient_room(budget, 4_000, 18_000).unwrap());
    assert!(insufficient_room(budget, 12_000, 18_000).is_err());
}
