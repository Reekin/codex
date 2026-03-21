use super::*;
use crate::codex::make_session_configuration_for_tests;
use crate::protocol::ChatTreeCurrentNodeChangedEvent;
use crate::protocol::EventMsg;
use crate::protocol::RateLimitWindow;
use crate::protocol::TurnCompleteEvent;
use crate::protocol::TurnStartedEvent;
use codex_protocol::config_types::ModeKind;
use codex_protocol::models::ContentItem;
use codex_protocol::models::ResponseItem;
use codex_protocol::protocol::RolloutItem;
use pretty_assertions::assert_eq;

#[tokio::test]
// Verifies connector merging deduplicates repeated IDs.
async fn merge_connector_selection_deduplicates_entries() {
    let session_configuration = make_session_configuration_for_tests().await;
    let mut state = SessionState::new(session_configuration);
    let merged = state.merge_connector_selection([
        "calendar".to_string(),
        "calendar".to_string(),
        "drive".to_string(),
    ]);

    assert_eq!(
        merged,
        HashSet::from(["calendar".to_string(), "drive".to_string()])
    );
}

#[tokio::test]
// Verifies clearing connector selection removes all saved IDs.
async fn clear_connector_selection_removes_entries() {
    let session_configuration = make_session_configuration_for_tests().await;
    let mut state = SessionState::new(session_configuration);
    state.merge_connector_selection(["calendar".to_string()]);

    state.clear_connector_selection();

    assert_eq!(state.get_connector_selection(), HashSet::new());
}

#[tokio::test]
async fn set_rate_limits_defaults_limit_id_to_codex_when_missing() {
    let session_configuration = make_session_configuration_for_tests().await;
    let mut state = SessionState::new(session_configuration);

    state.set_rate_limits(RateLimitSnapshot {
        limit_id: None,
        limit_name: None,
        primary: Some(RateLimitWindow {
            used_percent: 12.0,
            window_minutes: Some(60),
            resets_at: Some(100),
        }),
        secondary: None,
        credits: None,
        plan_type: None,
    });

    assert_eq!(
        state
            .latest_rate_limits
            .as_ref()
            .and_then(|v| v.limit_id.clone()),
        Some("codex".to_string())
    );
}

#[tokio::test]
async fn set_rate_limits_defaults_to_codex_when_limit_id_missing_after_other_bucket() {
    let session_configuration = make_session_configuration_for_tests().await;
    let mut state = SessionState::new(session_configuration);

    state.set_rate_limits(RateLimitSnapshot {
        limit_id: Some("codex_other".to_string()),
        limit_name: Some("codex_other".to_string()),
        primary: Some(RateLimitWindow {
            used_percent: 20.0,
            window_minutes: Some(60),
            resets_at: Some(200),
        }),
        secondary: None,
        credits: None,
        plan_type: None,
    });
    state.set_rate_limits(RateLimitSnapshot {
        limit_id: None,
        limit_name: None,
        primary: Some(RateLimitWindow {
            used_percent: 30.0,
            window_minutes: Some(60),
            resets_at: Some(300),
        }),
        secondary: None,
        credits: None,
        plan_type: None,
    });

    assert_eq!(
        state
            .latest_rate_limits
            .as_ref()
            .and_then(|v| v.limit_id.clone()),
        Some("codex".to_string())
    );
}

#[tokio::test]
async fn set_rate_limits_carries_credits_and_plan_type_from_codex_to_codex_other() {
    let session_configuration = make_session_configuration_for_tests().await;
    let mut state = SessionState::new(session_configuration);

    state.set_rate_limits(RateLimitSnapshot {
        limit_id: Some("codex".to_string()),
        limit_name: Some("codex".to_string()),
        primary: Some(RateLimitWindow {
            used_percent: 10.0,
            window_minutes: Some(60),
            resets_at: Some(100),
        }),
        secondary: None,
        credits: Some(crate::protocol::CreditsSnapshot {
            has_credits: true,
            unlimited: false,
            balance: Some("50".to_string()),
        }),
        plan_type: Some(codex_protocol::account::PlanType::Plus),
    });

    state.set_rate_limits(RateLimitSnapshot {
        limit_id: Some("codex_other".to_string()),
        limit_name: None,
        primary: Some(RateLimitWindow {
            used_percent: 30.0,
            window_minutes: Some(120),
            resets_at: Some(200),
        }),
        secondary: None,
        credits: None,
        plan_type: None,
    });

    assert_eq!(
        state.latest_rate_limits,
        Some(RateLimitSnapshot {
            limit_id: Some("codex_other".to_string()),
            limit_name: None,
            primary: Some(RateLimitWindow {
                used_percent: 30.0,
                window_minutes: Some(120),
                resets_at: Some(200),
            }),
            secondary: None,
            credits: Some(crate::protocol::CreditsSnapshot {
                has_credits: true,
                unlimited: false,
                balance: Some("50".to_string()),
            }),
            plan_type: Some(codex_protocol::account::PlanType::Plus),
        })
    );
}

fn user_message(text: &str) -> ResponseItem {
    ResponseItem::Message {
        id: None,
        role: "user".to_string(),
        content: vec![ContentItem::InputText {
            text: text.to_string(),
        }],
        end_turn: None,
        phase: None,
    }
}

fn assistant_message(text: &str) -> ResponseItem {
    ResponseItem::Message {
        id: None,
        role: "assistant".to_string(),
        content: vec![ContentItem::OutputText {
            text: text.to_string(),
        }],
        end_turn: None,
        phase: None,
    }
}

#[tokio::test]
async fn chat_tree_switch_restores_branch_snapshot_without_diverged_turns() {
    let session_configuration = make_session_configuration_for_tests().await;
    let mut state = SessionState::new(session_configuration);

    let node_a = state
        .create_chat_tree_child_node("node-a".to_string())
        .expect("create node-a");
    assert_eq!(node_a.parent_node_id, None);

    let items_a = [
        user_message("turn-a-user"),
        assistant_message("turn-a-assistant"),
    ];
    state.record_items(items_a.iter(), TruncationPolicy::Bytes(usize::MAX));
    state.finalize_chat_tree_node("node-a", Some("summary-a".to_string()));

    let snapshot_a = state.clone_history().raw_items().to_vec();

    let node_b = state
        .create_chat_tree_child_node("node-b".to_string())
        .expect("create node-b");
    assert_eq!(node_b.parent_node_id, Some("node-a".to_string()));
    let items_b = [
        user_message("turn-b-user"),
        assistant_message("turn-b-assistant"),
    ];
    state.record_items(items_b.iter(), TruncationPolicy::Bytes(usize::MAX));
    state.finalize_chat_tree_node("node-b", Some("summary-b".to_string()));

    state
        .set_current_chat_tree_node("node-a")
        .expect("switch to node-a");
    assert_eq!(state.clone_history().raw_items(), snapshot_a.as_slice());

    let node_c = state
        .create_chat_tree_child_node("node-c".to_string())
        .expect("create node-c");
    assert_eq!(node_c.parent_node_id, Some("node-a".to_string()));
    let items_c = vec![
        user_message("turn-c-user"),
        assistant_message("turn-c-assistant"),
    ];
    state.record_items(items_c.iter(), TruncationPolicy::Bytes(usize::MAX));
    state.finalize_chat_tree_node("node-c", Some("summary-c".to_string()));

    let mut expected = snapshot_a;
    expected.extend(items_c);
    assert_eq!(state.clone_history().raw_items(), expected.as_slice());
}

#[tokio::test]
async fn chat_tree_late_summary_keeps_original_snapshot() {
    let session_configuration = make_session_configuration_for_tests().await;
    let mut state = SessionState::new(session_configuration);

    state
        .create_chat_tree_child_node("node-a".to_string())
        .expect("create node-a");
    let items_a = [
        user_message("turn-a-user"),
        assistant_message("turn-a-assistant"),
    ];
    state.record_items(items_a.iter(), TruncationPolicy::Bytes(usize::MAX));
    state.finalize_chat_tree_node("node-a", None);
    let snapshot_a = state.clone_history().raw_items().to_vec();

    state
        .create_chat_tree_child_node("node-b".to_string())
        .expect("create node-b");
    let items_b = [
        user_message("turn-b-user"),
        assistant_message("turn-b-assistant"),
    ];
    state.record_items(items_b.iter(), TruncationPolicy::Bytes(usize::MAX));
    state.finalize_chat_tree_node("node-b", Some("summary-b".to_string()));

    let updated = state
        .finalize_chat_tree_node("node-a", Some("summary-a".to_string()))
        .expect("update node-a");
    assert_eq!(updated.node_id, "node-a");
    assert_eq!(updated.parent_node_id, None);
    assert_eq!(updated.summary, Some("summary-a".to_string()));

    state
        .set_current_chat_tree_node("node-a")
        .expect("switch to node-a");
    assert_eq!(state.clone_history().raw_items(), snapshot_a.as_slice());
}

#[tokio::test]
async fn sync_current_chat_tree_snapshot_keeps_compacted_branch_after_switching_away_and_back() {
    let session_configuration = make_session_configuration_for_tests().await;
    let mut state = SessionState::new(session_configuration);

    state
        .create_chat_tree_child_node("node-a".to_string())
        .expect("create node-a");
    let items_a = [
        user_message("turn-a-user"),
        assistant_message("turn-a-assistant"),
    ];
    state.record_items(items_a.iter(), TruncationPolicy::Bytes(usize::MAX));
    state.finalize_chat_tree_node("node-a", Some("summary-a".to_string()));

    state
        .create_chat_tree_child_node("node-b".to_string())
        .expect("create node-b");
    let items_b = [
        user_message("turn-b-user"),
        assistant_message("turn-b-assistant"),
    ];
    state.record_items(items_b.iter(), TruncationPolicy::Bytes(usize::MAX));
    state.finalize_chat_tree_node("node-b", Some("summary-b".to_string()));

    state
        .set_current_chat_tree_node("node-a")
        .expect("switch to node-a");
    state
        .create_chat_tree_child_node("node-c".to_string())
        .expect("create node-c");
    let items_c = [
        user_message("turn-c-user"),
        assistant_message("turn-c-assistant"),
    ];
    state.record_items(items_c.iter(), TruncationPolicy::Bytes(usize::MAX));
    state.finalize_chat_tree_node("node-c", Some("summary-c".to_string()));

    let compacted_history = vec![
        user_message("compacted-user"),
        assistant_message("compacted-summary"),
    ];
    state.replace_history(compacted_history.clone(), None);
    state.sync_current_chat_tree_snapshot();

    state
        .set_current_chat_tree_node("node-b")
        .expect("switch to node-b");
    state
        .set_current_chat_tree_node("node-c")
        .expect("switch to node-c");
    assert_eq!(
        state.clone_history().raw_items(),
        compacted_history.as_slice()
    );
}

#[tokio::test]
async fn restore_from_rollout_uses_persisted_parent_node_id() {
    let session_configuration = make_session_configuration_for_tests().await;
    let mut state = SessionState::new(session_configuration);
    let policy = TruncationPolicy::Bytes(usize::MAX);

    let turn_started = |turn_id: &str| {
        RolloutItem::EventMsg(EventMsg::TurnStarted(TurnStartedEvent {
            turn_id: turn_id.to_string(),
            model_context_window: None,
            collaboration_mode_kind: ModeKind::Default,
        }))
    };
    let turn_complete = |turn_id: &str, parent_node_id: Option<&str>, summary: &str| {
        RolloutItem::EventMsg(EventMsg::TurnComplete(TurnCompleteEvent {
            turn_id: turn_id.to_string(),
            last_agent_message: None,
            chat_tree: Some(ChatTreeTurnInfo {
                node_id: turn_id.to_string(),
                parent_node_id: parent_node_id.map(std::string::ToString::to_string),
                summary: Some(summary.to_string()),
            }),
        }))
    };

    let rollout_items = vec![
        turn_started("node-a"),
        RolloutItem::ResponseItem(user_message("turn-a-user")),
        RolloutItem::ResponseItem(assistant_message("turn-a-assistant")),
        turn_complete("node-a", None, "summary-a"),
        turn_started("node-b"),
        RolloutItem::ResponseItem(user_message("turn-b-user")),
        RolloutItem::ResponseItem(assistant_message("turn-b-assistant")),
        turn_complete("node-b", Some("node-a"), "summary-b"),
        turn_started("node-c"),
        RolloutItem::ResponseItem(user_message("turn-c-user")),
        RolloutItem::ResponseItem(assistant_message("turn-c-assistant")),
        turn_complete("node-c", Some("node-a"), "summary-c"),
    ];

    state.restore_from_rollout(&rollout_items, &[], policy);

    state
        .set_current_chat_tree_node("node-b")
        .expect("switch to node-b");
    assert_eq!(
        state.clone_history().raw_items(),
        &[
            user_message("turn-a-user"),
            assistant_message("turn-a-assistant"),
            user_message("turn-b-user"),
            assistant_message("turn-b-assistant"),
        ]
    );

    state
        .set_current_chat_tree_node("node-c")
        .expect("switch to node-c");
    assert_eq!(
        state.clone_history().raw_items(),
        &[
            user_message("turn-a-user"),
            assistant_message("turn-a-assistant"),
            user_message("turn-c-user"),
            assistant_message("turn-c-assistant"),
        ]
    );
}

#[tokio::test]
async fn restore_from_rollout_replays_explicit_current_node_change() {
    let session_configuration = make_session_configuration_for_tests().await;
    let mut state = SessionState::new(session_configuration);
    let policy = TruncationPolicy::Bytes(usize::MAX);

    let turn_started = |turn_id: &str| {
        RolloutItem::EventMsg(EventMsg::TurnStarted(TurnStartedEvent {
            turn_id: turn_id.to_string(),
            model_context_window: None,
            collaboration_mode_kind: ModeKind::Default,
        }))
    };
    let turn_complete = |turn_id: &str, parent_node_id: Option<&str>, summary: &str| {
        RolloutItem::EventMsg(EventMsg::TurnComplete(TurnCompleteEvent {
            turn_id: turn_id.to_string(),
            last_agent_message: None,
            chat_tree: Some(ChatTreeTurnInfo {
                node_id: turn_id.to_string(),
                parent_node_id: parent_node_id.map(std::string::ToString::to_string),
                summary: Some(summary.to_string()),
            }),
        }))
    };

    let rollout_items = vec![
        turn_started("node-a"),
        RolloutItem::ResponseItem(user_message("turn-a-user")),
        RolloutItem::ResponseItem(assistant_message("turn-a-assistant")),
        turn_complete("node-a", None, "summary-a"),
        turn_started("node-b"),
        RolloutItem::ResponseItem(user_message("turn-b-user")),
        RolloutItem::ResponseItem(assistant_message("turn-b-assistant")),
        turn_complete("node-b", Some("node-a"), "summary-b"),
        turn_started("node-c"),
        RolloutItem::ResponseItem(user_message("turn-c-user")),
        RolloutItem::ResponseItem(assistant_message("turn-c-assistant")),
        turn_complete("node-c", Some("node-a"), "summary-c"),
        RolloutItem::EventMsg(EventMsg::ChatTreeCurrentNodeChanged(
            ChatTreeCurrentNodeChangedEvent {
                node_id: "node-b".to_string(),
            },
        )),
    ];

    state.restore_from_rollout(&rollout_items, &[], policy);

    assert_eq!(state.current_chat_tree_node_id().as_deref(), Some("node-b"));
    assert_eq!(
        state.clone_history().raw_items(),
        &[
            user_message("turn-a-user"),
            assistant_message("turn-a-assistant"),
            user_message("turn-b-user"),
            assistant_message("turn-b-assistant"),
        ]
    );
}
