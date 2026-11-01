//! Out-of-turn tool cleanup status and manual application for app-server clients.

use std::sync::Arc;

use codex_context_compaction::cleaned;
use codex_context_compaction::visible_tokens;
use codex_protocol::error::Result as CodexResult;

use super::CleanupTrigger;
use super::LocalCompactionContext;
use super::LocalCompactionState;
use super::budget;
use super::history_tokens;
use super::install_marks;
use super::invalid;
use super::item_tokens;
use super::prepared_state;
use super::reasoning_budget;
use super::to_reported;
use crate::session::session::Session;
use codex_features::Feature;

/// Tool cleanup progress for the thread.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ToolCleanupStatus {
    /// Whether a background marking request is in flight.
    pub marking: bool,
    /// Estimated context tokens released by applying every current validated mark and
    /// trimming earlier turns' reasoning.
    pub pending_savings_tokens: usize,
    /// Release required before automatic tool cleanup applies the marks.
    pub required_savings_tokens: usize,
}

impl ToolCleanupStatus {
    /// Reported while local context compaction is off: nothing is marked or cleaned.
    const NOTHING: Self = Self {
        marking: false,
        pending_savings_tokens: 0,
        required_savings_tokens: 0,
    };
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ToolCleanupOutcome {
    /// Estimated context tokens released by this cleanup; zero when nothing changed.
    pub released_tokens: usize,
    pub status: ToolCleanupStatus,
}

pub(crate) async fn tool_cleanup_status(sess: &Arc<Session>) -> CodexResult<ToolCleanupStatus> {
    if !sess.enabled(Feature::LocalContextCompaction) {
        return Ok(ToolCleanupStatus::NOTHING);
    }
    let _boundary = sess.lock_local_compaction_boundary().await;
    let context = LocalCompactionContext::from_turn(sess.local_compaction_turn().await);
    let mut state = prepared_state(sess, &context).await;
    let status = status(sess, &context, &mut state).await;
    sess.set_local_compaction_state(state).await;
    status
}

/// Applies every validated mark now, regardless of the automatic savings requirement.
/// Unfinished marking keeps running and is not awaited.
pub(crate) async fn apply_tool_cleanup(sess: &Arc<Session>) -> CodexResult<ToolCleanupOutcome> {
    if !sess.enabled(Feature::LocalContextCompaction) {
        return Ok(ToolCleanupOutcome {
            released_tokens: 0,
            status: ToolCleanupStatus::NOTHING,
        });
    }
    let _boundary = sess.lock_local_compaction_boundary().await;
    let context = LocalCompactionContext::from_turn(sess.local_compaction_turn().await);
    let mut state = prepared_state(sess, &context).await;
    let result = async {
        // Measured before cleanup replaces the provider-reported usage with an estimate.
        let (_, scale) = budget(sess, &context).await;
        let released_tokens =
            install_marks(sess, &context, &mut state, CleanupTrigger::Manual).await?;
        Ok(ToolCleanupOutcome {
            released_tokens: to_reported(released_tokens, scale),
            status: status(sess, &context, &mut state).await?,
        })
    }
    .await;
    sess.set_local_compaction_state(state).await;
    result
}

async fn status(
    sess: &Arc<Session>,
    context: &LocalCompactionContext,
    state: &mut LocalCompactionState,
) -> CodexResult<ToolCleanupStatus> {
    let history = sess.clone_history().await;
    let source = history.annotated_items();
    state.staged.retain_current(source);
    // Reported in provider tokens, the same unit as context usage.
    let (budget, scale) = budget(sess, context).await;
    let retained = cleaned(
        &state.staged,
        source,
        reasoning_budget(context, budget),
        item_tokens,
    )
    .map_err(invalid)?;
    let pending_savings_tokens = visible_tokens(
        source,
        context.settings.model_info.retains_prior_reasoning,
        item_tokens,
    )
    .saturating_sub(history_tokens(&retained));
    Ok(ToolCleanupStatus {
        marking: state.background.is_some(),
        pending_savings_tokens: to_reported(pending_savings_tokens, scale),
        required_savings_tokens: to_reported(budget.required_savings(), scale),
    })
}
