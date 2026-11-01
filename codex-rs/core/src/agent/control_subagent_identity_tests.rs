//! Fork-history boundaries for full-history forks, enabled by `Feature::SubagentIdentity`.

use super::*;
use pretty_assertions::assert_eq;

const FORK_START: &str = "forked parent conversation history begins";
const FORK_END: &str = "forked parent conversation history ends";

#[derive(Clone, Copy, Debug)]
enum ParentHistory {
    Plain,
    ReplacementCompaction,
    LegacyCompaction,
}

fn compaction(message: &str, replacement_history: Option<Vec<ResponseItem>>) -> RolloutItem {
    RolloutItem::Compacted(Box::new(CompactedItem {
        message: message.to_string(),
        replacement_history: replacement_history
            .map(|items| items.into_iter().map(Into::into).collect()),
        retained_context: None,
        guardian_history: None,
        mcp_resource_origins: None,
        window_number: None,
        first_window_id: None,
        previous_window_id: None,
        window_id: None,
        compaction_response_id: None,
        latest_token_usage_record: None,
        resume_metadata: None,
    }))
}

fn history_text_index(history_items: &[ResponseItem], needle: &str) -> Option<usize> {
    history_items
        .iter()
        .position(|item| history_contains_text(std::iter::once(item), needle))
}

#[test_case::test_case(ParentHistory::Plain, true; "plain")]
#[test_case::test_case(ParentHistory::ReplacementCompaction, true; "replacement compaction")]
#[test_case::test_case(ParentHistory::LegacyCompaction, true; "legacy compaction")]
#[test_case::test_case(ParentHistory::Plain, false; "flag off keeps upstream history")]
#[tokio::test]
async fn full_history_fork_wraps_inherited_history_in_identity_boundaries(
    parent_history: ParentHistory,
    subagent_identity: bool,
) {
    let mut harness = AgentControlHarness::new().await;
    if subagent_identity {
        harness
            .config
            .features
            .enable(Feature::SubagentIdentity)
            .expect("test config should allow subagent identity");
    }
    let new_thread = harness
        .manager
        .start_thread(StartThreadOptions::new(harness.config.clone()))
        .await
        .expect("start parent thread");
    let parent_thread_id = new_thread.thread_id;
    let parent_thread = new_thread.thread;
    let parent_spawn_call_id = "spawn-call-identity-boundaries".to_string();
    let inherited_text = "inherited parent context";
    let mut rollout_items = match parent_history {
        ParentHistory::Plain => vec![rollout_response_item(user_message(inherited_text))],
        ParentHistory::ReplacementCompaction => {
            vec![compaction("", Some(vec![user_message(inherited_text)]))]
        }
        ParentHistory::LegacyCompaction => vec![
            rollout_response_item(user_message(inherited_text)),
            compaction(
                "legacy compacted summary",
                /*replacement_history*/ None,
            ),
        ],
    };
    rollout_items.push(rollout_response_item(spawn_agent_call(
        &parent_spawn_call_id,
    )));
    parent_thread
        .session
        .persist_rollout_items(&rollout_items)
        .await;
    parent_thread
        .session
        .ensure_rollout_materialized(PersistContext::Standard)
        .await;
    parent_thread
        .session
        .flush_rollout()
        .await
        .expect("parent rollout should flush");

    let child_thread_id = harness
        .spawn_anonymous_child(
            parent_thread_id,
            SpawnAgentOptions {
                fork_parent_spawn_call_id: Some(parent_spawn_call_id),
                fork_mode: Some(SpawnAgentForkMode::FullHistory),
                ..Default::default()
            },
        )
        .await;
    let child_thread = harness
        .manager
        .get_thread(child_thread_id)
        .await
        .expect("child thread should be registered");
    let history = child_thread
        .session
        .clone_history()
        .await
        .raw_items()
        .cloned()
        .collect::<Vec<_>>();

    let inherited = history_text_index(&history, inherited_text)
        .expect("inherited parent history should remain visible");
    let start = history_text_index(&history, FORK_START);
    let end = history_text_index(&history, FORK_END);
    if subagent_identity {
        let (start, end) = (
            start.expect("full fork should include a start boundary"),
            end.expect("full fork should include an end boundary"),
        );
        assert!(
            start < inherited && inherited < end,
            "{parent_history:?}: inherited history should sit between fork boundaries: start={start}, inherited={inherited}, end={end}"
        );
        assert!(
            history_contains_text(
                std::iter::once(&history[end]),
                "You do not have a canonical agent path in this session"
            ),
            "{parent_history:?}: the end boundary should carry the child's identity"
        );
    } else {
        assert_eq!(
            (start, end),
            (None, None),
            "without subagent identity the fork history matches upstream"
        );
    }

    let _ = harness
        .control
        .shutdown_live_agent(child_thread_id)
        .await
        .expect("child shutdown should submit");
    let _ = parent_thread
        .submit(Op::Shutdown {})
        .await
        .expect("parent shutdown should submit");
}
