//! Private background marking and atomic installation at ordinary sampling boundaries.

mod full;
mod marking;
mod marks;
mod request;
mod tool_cleanup;

pub(crate) use full::run_pipeline;
pub(crate) use marking::start_marking;
pub use tool_cleanup::ToolCleanupOutcome;
pub use tool_cleanup::ToolCleanupStatus;
pub(crate) use tool_cleanup::apply_tool_cleanup;
pub(crate) use tool_cleanup::tool_cleanup_status;

use crate::compact::CompactedHistoryMetadata;
use crate::compact::LocalCompactionContext;
use crate::context_manager::estimate_item_token_count;
use crate::session::session::Session;
use crate::session::step_context::StepContext;
use crate::session::turn_context::TurnContext;
use codex_context_compaction::Budget;
use codex_context_compaction::StagedDecisions;
use codex_context_compaction::dropped_reasoning;
use codex_history::ResponseItemEnvelope;
use codex_model_provider::RemoteCompactionSupport;
use codex_protocol::error::CodexErr;
use codex_protocol::error::Result as CodexResult;
use codex_protocol::models::ResponseItem;
use codex_protocol::openai_models::ModelInfo;
use codex_utils_output_truncation::approx_token_count;
use std::sync::Arc;
use std::sync::Mutex;
use std::time::Duration;
use std::time::Instant;

#[derive(Clone, Copy, PartialEq, Eq)]
pub(crate) enum AnalysisKind {
    Background,
    Full,
}

#[derive(Clone, Debug, Default)]
pub(crate) struct LocalCompactionState {
    pub(crate) generation: u64,
    staged: StagedDecisions,
    /// Durable marks are replayed once per full window, after resume or full compaction.
    restored: bool,
    background: Option<Arc<BackgroundMarking>>,
    /// Unmarked results present when marking started, still to be marked.
    backlog: Option<std::collections::HashSet<String>>,
    retry_after: Option<Instant>,
    /// Serializes automatic and manual tool cleanup; survives full-window resets.
    boundary: Arc<tokio::sync::Mutex<()>>,
}

#[derive(Debug)]
struct BackgroundMarking {
    abort: tokio::task::AbortHandle,
    completed: Arc<Mutex<MarkingResult>>,
}

#[derive(Debug, Default)]
struct MarkingResult {
    value: Option<CodexResult<StagedDecisions>>,
    finished: bool,
}

impl Drop for BackgroundMarking {
    fn drop(&mut self) {
        // Consuming a ready result may precede its billing notification. Allow that
        // short final delivery to finish; full-window reset explicitly aborts either state.
        if !self.completed.lock().expect("marking result lock").finished {
            self.abort.abort();
        }
    }
}

impl LocalCompactionState {
    pub(crate) fn reset_window(&mut self) {
        if let Some(background) = self.background.take() {
            background.abort.abort();
        }
        *self = Self {
            generation: self.generation.wrapping_add(1),
            boundary: Arc::clone(&self.boundary),
            ..Self::default()
        };
    }

    pub(crate) fn boundary(&self) -> Arc<tokio::sync::Mutex<()>> {
        Arc::clone(&self.boundary)
    }
}

#[derive(Clone, Copy)]
enum CleanupTrigger {
    /// Install only when the configured share of the window is released.
    Savings,
    Manual,
}

/// Whether full compaction uses the local pipeline: the provider or the model lacks native
/// compaction. Tool cleanup runs on every route.
pub(crate) fn uses_local_route(turn: &TurnContext, model_info: &ModelInfo) -> bool {
    crate::compaction_policy::remote_compaction_support(
        turn.provider.capabilities().remote_compaction,
        model_info,
    ) == RemoteCompactionSupport::Unsupported
}

/// Installs validated marks before an ordinary request when they release enough.
pub(crate) async fn maybe_clean_history(
    sess: &Arc<Session>,
    step: &Arc<StepContext>,
) -> CodexResult<()> {
    let _boundary = sess.lock_local_compaction_boundary().await;
    let context = LocalCompactionContext::from_step(Arc::clone(step));
    let mut state = prepared_state(sess, &context).await;
    let result = install_marks(sess, &context, &mut state, CleanupTrigger::Savings).await;
    sess.set_local_compaction_state(state).await;
    result.map(|_| ())
}

/// Restores durable marks and takes a finished background batch. Callers hold the boundary lock.
async fn prepared_state(
    sess: &Arc<Session>,
    context: &LocalCompactionContext,
) -> LocalCompactionState {
    let mut state = sess.get_local_compaction_state().await;
    if !state.restored {
        state.restored = true;
        let history = sess.clone_history().await;
        let mut staged =
            marks::restore(sess, &context.turn.config, history.annotated_items()).await;
        staged.merge(std::mem::take(&mut state.staged));
        state.staged = staged;
    }
    let completed = state.background.as_ref().and_then(|task| {
        task.completed
            .lock()
            .expect("marking result lock")
            .value
            .take()
    });
    if let Some(completed) = completed {
        state.background = None;
        match completed {
            Ok(decisions) => {
                state.staged.merge(decisions);
            }
            Err(error) => {
                tracing::warn!(%error, "background local marking failed");
                state.retry_after = Some(Instant::now() + Duration::from_secs(60));
            }
        }
    }
    state
}

/// Installs the marked view and returns the released history tokens, or zero if unchanged.
async fn install_marks(
    sess: &Arc<Session>,
    context: &LocalCompactionContext,
    state: &mut LocalCompactionState,
    trigger: CleanupTrigger,
) -> CodexResult<usize> {
    // No model await between this snapshot and the persistence CAS.
    let history = sess.clone_history().await;
    let source = history.annotated_items();
    state.staged.retain_current(source);
    let (budget, _) = budget(sess, context).await;
    let before_tokens = visible_tokens(source, &context.settings.model_info);
    let replacement = cleaned(&state.staged, source, reasoning_budget(context, budget))?;
    let after_tokens = history_tokens(&replacement);
    let wanted = match trigger {
        CleanupTrigger::Savings => budget.useful(before_tokens, after_tokens),
        CleanupTrigger::Manual => after_tokens < before_tokens,
    };
    if !wanted
        || !sess
            .install_local_compaction(
                source,
                history
                    .conversation_history_snapshot()
                    .user_message_revision(),
                state.generation,
                replacement,
                sess.reference_context_item().await,
                None,
                checkpoint(context, None),
                // Tool cleanup has a savings threshold, not a target occupancy.
                usize::MAX,
            )
            .await?
    {
        return Ok(0);
    }
    sess.recompute_token_usage(&context.turn).await;
    Ok(before_tokens - after_tokens)
}

fn checkpoint(
    context: &LocalCompactionContext,
    response_id: Option<String>,
) -> CompactedHistoryMetadata {
    CompactedHistoryMetadata {
        message: "Local context cleanup; exact originals are available through recall_read_item."
            .to_string(),
        window_number: 0,
        window_ids: crate::state::AutoCompactWindowIds::new_initial(),
        compaction_response_id: response_id,
        compaction_model_hash: context.settings.model_info.comp_hash.clone(),
    }
}

/// Budgets in local-estimate units, plus provider tokens per estimated token.
///
/// Local estimates (about 4 bytes per token) run well below provider counts for code-heavy
/// history, while the window and the hard limit are provider counts. Converting those two to
/// estimate units keeps every comparison in one unit.
async fn budget(sess: &Session, context: &LocalCompactionContext) -> (Budget, f64) {
    let config = &context.turn.config.local_compaction;
    let model = &context.settings.model_info;
    let window_tokens = model
        .resolved_context_window()
        .unwrap_or(128_000)
        .saturating_mul(model.effective_context_window_percent)
        / 100;
    let base = sess.get_prompt_base_instructions().await;
    let tool_tokens = if context.tool_tokens == 0 {
        usize::try_from(sess.request_tools_tokens().await).unwrap_or(0)
    } else {
        context.tool_tokens
    };
    let fixed_tokens = approx_token_count(&base.text).saturating_add(tool_tokens);
    // Ratio of the latest provider-reported context to the estimate of what it was sent.
    // Bounded so a stale or estimated report (for example right after a cleanup) falls back
    // to raw estimates.
    let estimated = visible_tokens(sess.clone_history().await.annotated_items(), model)
        .saturating_add(fixed_tokens);
    let reported = sess.get_total_token_usage().await;
    let scale = if estimated == 0 {
        1.0
    } else {
        (reported as f64 / estimated as f64).clamp(1.0, 2.0)
    };
    let budget = Budget {
        window_tokens: to_estimate(usize::try_from(window_tokens).unwrap_or(0), scale),
        fixed_tokens,
        reclaim_percent: usize::from(config.reclaim_percent),
        compact_target_percent: usize::from(config.compact_target_percent),
        keep_reasoning_percent: usize::from(config.keep_reasoning_percent),
    };
    (budget, scale)
}

/// Earlier turns' reasoning that cleanup keeps. Models that drop it themselves are sent none
/// of it, so removing it locally changes nothing they see.
fn reasoning_budget(context: &LocalCompactionContext, budget: Budget) -> usize {
    if context.settings.model_info.retains_prior_reasoning {
        budget.reasoning_tokens()
    } else {
        0
    }
}

/// Private requests are traced like ordinary inference so their answers can be inspected.
fn inference_trace(
    sess: &Session,
    context: &LocalCompactionContext,
) -> codex_rollout_trace::InferenceTraceContext {
    sess.services.rollout_thread_trace.inference_trace_context(
        context.turn.sub_id.as_str(),
        context.settings.model_info.slug.as_str(),
        context.turn.provider.info().name.as_str(),
    )
}

/// The history after applying validated marks and trimming earlier reasoning.
fn cleaned(
    staged: &StagedDecisions,
    items: &[ResponseItemEnvelope],
    reasoning_tokens: usize,
) -> CodexResult<Vec<ResponseItemEnvelope>> {
    let items = staged.apply(items).map_err(invalid)?;
    let costs: Vec<_> = items.iter().map(|item| item_tokens(&item.item)).collect();
    let dropped = dropped_reasoning(&items, reasoning_tokens, &costs);
    Ok(items
        .into_iter()
        .zip(dropped)
        .filter_map(|(item, dropped)| (!dropped).then_some(item))
        .collect())
}

/// Estimated tokens the model is sent; earlier turns' reasoning does not count for models
/// that drop it.
fn visible_tokens(items: &[ResponseItemEnvelope], model: &ModelInfo) -> usize {
    if model.retains_prior_reasoning {
        return history_tokens(items);
    }
    let costs: Vec<_> = items.iter().map(|item| item_tokens(&item.item)).collect();
    let dropped = dropped_reasoning(items, 0, &costs);
    costs
        .into_iter()
        .zip(dropped)
        .filter_map(|(cost, dropped)| (!dropped).then_some(cost))
        .sum()
}

fn to_estimate(tokens: usize, scale: f64) -> usize {
    (tokens as f64 / scale) as usize
}

fn to_reported(tokens: usize, scale: f64) -> usize {
    (tokens as f64 * scale) as usize
}

fn item_tokens(item: &ResponseItem) -> usize {
    usize::try_from(estimate_item_token_count(item)).unwrap_or(0)
}

fn history_tokens(items: &[ResponseItemEnvelope]) -> usize {
    items.iter().map(|item| item_tokens(&item.item)).sum()
}

fn invalid(error: impl std::fmt::Display) -> CodexErr {
    CodexErr::InvalidRequest(format!("Local compaction failed: {error}"))
}
