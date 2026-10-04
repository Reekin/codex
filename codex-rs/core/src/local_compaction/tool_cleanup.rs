//! Out-of-turn tool cleanup status and manual application for app-server clients.

use std::sync::Arc;

use codex_protocol::error::Result as CodexResult;

use super::CleanupTrigger;
use super::LocalCompactionState;
use super::budget;
use super::cleaned;
use super::history_tokens;
use super::install_marks;
use super::prepared_state;
use super::reasoning_budget;
use super::to_reported;
use super::uses_local_route;
use super::visible_tokens;
use crate::compact::LocalCompactionContext;
use crate::session::session::Session;

/// Tool cleanup progress for the thread's current model route.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ToolCleanupStatus {
    /// Whether background marking runs on the current model route.
    pub enabled: bool,
    /// Whether a background marking request is in flight.
    pub marking: bool,
    /// Estimated history tokens released by applying every current validated mark and
    /// trimming earlier turns' reasoning.
    pub pending_savings_tokens: usize,
    /// Release required before automatic tool cleanup applies the marks.
    pub required_savings_tokens: usize,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ToolCleanupOutcome {
    /// Estimated history tokens released by this cleanup; zero when nothing changed.
    pub released_tokens: usize,
    pub status: ToolCleanupStatus,
}

pub(crate) async fn tool_cleanup_status(sess: &Arc<Session>) -> CodexResult<ToolCleanupStatus> {
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
    let pending_savings_tokens =
        visible_tokens(source, &context.settings.model_info).saturating_sub(history_tokens(
            &cleaned(&state.staged, source, reasoning_budget(context, budget))?,
        ));
    Ok(ToolCleanupStatus {
        enabled: uses_local_route(&context.turn, &context.settings.model_info),
        marking: state.background.is_some(),
        pending_savings_tokens: to_reported(pending_savings_tokens, scale),
        required_savings_tokens: to_reported(budget.required_savings(), scale),
    })
}
