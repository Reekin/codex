use super::*;
use codex_protocol::context_usage::ContextUsage;
use codex_protocol::context_usage::ContextUsageBreakdown;
use codex_protocol::protocol::TokenCountEvent;
use codex_protocol::protocol::TokenUsage;
use codex_protocol::protocol::TokenUsageInfo;
use codex_protocol::protocol::TurnCompleteEvent;
use codex_protocol::protocol::TurnStartedEvent;
use pretty_assertions::assert_eq;

fn turn_started(turn_id: &str) -> RolloutItem {
    RolloutItem::EventMsg(EventMsg::TurnStarted(TurnStartedEvent {
        turn_id: turn_id.to_string(),
        trace_id: None,
        started_at: None,
        model_context_window: None,
        collaboration_mode_kind: Default::default(),
    }))
}

fn turn_complete(turn_id: &str) -> RolloutItem {
    RolloutItem::EventMsg(EventMsg::TurnComplete(TurnCompleteEvent {
        turn_id: turn_id.to_string(),
        started_at: None,
        last_agent_message: None,
        error: None,
        completed_at: None,
        duration_ms: None,
        time_to_first_token_ms: None,
    }))
}

fn token_count(info: Option<TokenUsageInfo>) -> RolloutItem {
    RolloutItem::EventMsg(EventMsg::TokenCount(TokenCountEvent {
        info,
        rate_limits: None,
    }))
}

fn usage_info(total_tokens: i64, context_usage: Option<ContextUsage>) -> TokenUsageInfo {
    let usage = TokenUsage {
        total_tokens,
        ..Default::default()
    };
    TokenUsageInfo {
        total_token_usage: usage.clone(),
        last_token_usage: usage,
        model_context_window: Some(1_000),
        context_usage,
    }
}

#[test]
fn keeps_last_snapshot_per_turn_including_legacy_snapshots() {
    let context_usage = ContextUsage {
        tokens: 20,
        auto_compact_token_limit: Some(900),
        breakdown: ContextUsageBreakdown {
            user_messages: 20,
            ..Default::default()
        },
    };
    let first_final = usage_info(20, Some(context_usage));
    // Snapshots recorded before context usage existed carry no record.
    let second_final = usage_info(35, None);
    let items = vec![
        turn_started("turn-1"),
        token_count(Some(usage_info(10, None))),
        token_count(Some(first_final.clone())),
        turn_complete("turn-1"),
        turn_started("turn-2"),
        token_count(None),
        token_count(Some(second_final.clone())),
        turn_complete("turn-2"),
    ];

    assert_eq!(
        turn_token_usages(&items),
        vec![
            TurnTokenUsage {
                turn_id: "turn-1".to_string(),
                token_usage: first_final.into(),
            },
            TurnTokenUsage {
                turn_id: "turn-2".to_string(),
                token_usage: second_final.into(),
            },
        ]
    );
}
