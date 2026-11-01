//! Reconstruction of the fork-history boundary that subagent identity records for forked children.

use super::*;
use pretty_assertions::assert_eq;
use crate::context::ForkedHistoryBoundary;
use codex_context_fragments::AnnotatedContent;
use codex_context_fragments::RenderedFragment;
use codex_protocol::models::ContentItemKind;
use codex_protocol::models::input_text;

fn typed_fork_boundary_message(text: &str) -> ResponseItem {
    RenderedFragment::new(
        "developer",
        AnnotatedContent::new(
            input_text(text),
            ContentItemKind(ForkedHistoryBoundary::CONTENT_KIND.to_string()),
        ),
    )
    .into()
}

fn legacy_compaction(message: &str) -> RolloutItem {
    RolloutItem::Compacted(Box::new(CompactedItem {
        message: message.to_string(),
        replacement_history: None,
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

#[tokio::test]
async fn reconstruct_history_legacy_compaction_uses_typed_fork_boundary_not_wording() {
    let (session, turn_context) = make_session_and_context().await;
    let parent_message = user_message("inherited parent task");
    let renamed_boundary = typed_fork_boundary_message("Updated opening boundary wording.");

    let typed = session
        .reconstruct_history_from_rollout(
            &turn_context,
            &[
                RolloutItem::ResponseItem(parent_message.clone().into()),
                RolloutItem::ResponseItem(renamed_boundary.clone().into()),
                legacy_compaction("legacy summary"),
            ],
        )
        .await;
    assert_eq!(
        typed.history,
        annotated(vec![
            renamed_boundary,
            parent_message,
            ContextualUserFragment::into(CompactionSummary::new("legacy summary")),
        ])
    );

    let ordinary_message =
        user_message("ordinary user text: forked parent conversation history begins");
    let ordinary = session
        .reconstruct_history_from_rollout(
            &turn_context,
            &[
                RolloutItem::ResponseItem(ordinary_message.clone().into()),
                legacy_compaction("legacy summary"),
            ],
        )
        .await;
    assert_eq!(
        ordinary.history,
        annotated(vec![
            ordinary_message,
            ContextualUserFragment::into(CompactionSummary::new("legacy summary")),
        ])
    );
}
