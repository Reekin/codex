//! Runtime adapter for local context compaction: background marking, installation of validated
//! marks at ordinary sampling boundaries, and the local full-compaction route. Policy lives in
//! `codex_context_compaction`.
//!
//! Everything here is enabled by `[features] local_context_compaction`; with it off every route
//! keeps the upstream behavior and the entry points below are not reached.

mod full;
mod marking;
mod request;
mod tool_cleanup;

pub(crate) use full::run_pipeline;
pub(crate) use marking::start_marking;
pub use tool_cleanup::ToolCleanupOutcome;
pub use tool_cleanup::ToolCleanupStatus;
pub(crate) use tool_cleanup::apply_tool_cleanup;
pub(crate) use tool_cleanup::tool_cleanup_status;

use std::collections::HashSet;
use std::path::PathBuf;
use std::sync::Arc;
use std::sync::Mutex;
use std::time::Duration;
use std::time::Instant;

use codex_context_compaction::Budget;
use codex_context_compaction::StagedDecisions;
use codex_context_compaction::cleaned;
use codex_context_compaction::marks_path;
use codex_context_compaction::restore_marks;
use codex_context_compaction::visible_tokens;
use codex_features::Feature;
use codex_history::ResponseItemEnvelope;
use codex_model_provider::RemoteCompactionSupport;
use codex_protocol::error::CodexErr;
use codex_protocol::error::Result as CodexResult;
use codex_protocol::models::ResponseItem;
use codex_protocol::openai_models::ModelInfo;

use crate::config::Config;
use crate::context_manager::estimate_item_token_count;
use crate::session::session::Session;
use crate::session::step_context::StepContext;
use crate::session::step_settings::ResolvedStepSettings;
use crate::session::turn_context::TurnContext;

#[derive(Clone, Copy, PartialEq, Eq)]
pub(crate) enum AnalysisKind {
    Background,
    Full,
}

/// Session-owned marking state for the current full window.
#[derive(Clone, Debug, Default)]
pub(crate) struct LocalCompactionState {
    /// Advances with every full-window reset; results of an older window are ignored.
    pub(crate) generation: u64,
    staged: StagedDecisions,
    /// Durable marks are restored once per full window, after resume or full compaction.
    restored: bool,
    background: Option<Arc<BackgroundMarking>>,
    /// Unmarked results present when marking started, still to be marked.
    backlog: Option<HashSet<String>>,
    retry_after: Option<Instant>,
    /// Serializes automatic and manual cleanup cycles; survives full-window resets.
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
        // Consuming a ready result may precede its billing notification. Allow that short
        // final delivery to finish; a full-window reset explicitly aborts either state.
        if !self.completed.lock().expect("marking result lock").finished {
            self.abort.abort();
        }
    }
}

impl LocalCompactionState {
    /// Starts a new full window: in-flight marking is aborted and its late result ignored.
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

/// The turn and model settings a cleanup, marking or full-compaction step works with.
pub(crate) struct LocalCompactionContext {
    pub(crate) turn: Arc<TurnContext>,
    pub(crate) settings: Arc<ResolvedStepSettings>,
}

impl LocalCompactionContext {
    pub(crate) fn from_step(step: &Arc<StepContext>) -> Self {
        Self {
            turn: Arc::clone(&step.turn),
            settings: Arc::clone(&step.settings),
        }
    }

    pub(crate) fn from_turn(turn: Arc<TurnContext>) -> Self {
        Self {
            settings: Arc::clone(&turn.initial_settings),
            turn,
        }
    }

    fn model_info(&self) -> &ModelInfo {
        &self.settings.model_info
    }
}

#[derive(Clone, Copy)]
enum CleanupTrigger {
    /// Install only when the configured share of the window is released.
    Savings,
    Manual,
}

/// Whether local context compaction is on (`[features] local_context_compaction`).
pub(crate) fn enabled(config: &Config) -> bool {
    config.features.enabled(Feature::LocalContextCompaction)
}

/// Whether local full compaction owns the context window of `model_info`: the feature is on and
/// the provider or the model lacks native compaction.
pub(crate) fn owns_window(
    config: &Config,
    provider_support: RemoteCompactionSupport,
    model_info: &ModelInfo,
) -> bool {
    enabled(config) && is_local_route(provider_support, model_info)
}

/// Whether full compaction uses the local route: the provider or the model lacks native
/// compaction. Tool cleanup runs on every route.
pub(crate) fn uses_local_route(turn: &TurnContext, model_info: &ModelInfo) -> bool {
    is_local_route(turn.provider.capabilities().remote_compaction, model_info)
}

/// [`uses_local_route`] for a provider's compaction support, where no turn exists yet.
pub(crate) fn is_local_route(
    provider_support: RemoteCompactionSupport,
    model_info: &ModelInfo,
) -> bool {
    crate::compaction_policy::remote_compaction_support(provider_support, model_info)
        == RemoteCompactionSupport::Unsupported
}

/// Whether history holds locally compacted records whose originals recall can read.
pub(crate) async fn has_local_records(sess: &Session) -> bool {
    sess.clone_history()
        .await
        .annotated_items()
        .iter()
        .any(|item| {
            item.metadata
                .as_ref()
                .is_some_and(|metadata| metadata.local_compaction.is_some())
        })
}

/// Cancels marking for the current full window; full compaction and shutdown start here.
pub(crate) async fn cancel(sess: &Session) {
    sess.cancel_local_compaction().await;
}

/// Native full compaction starts a new window, so marking for the old one stops. The local
/// pipeline cancels it itself after applying the marks already validated.
pub(crate) async fn before_full_compaction(
    sess: &Session,
    turn: &TurnContext,
    model_info: &ModelInfo,
) {
    if enabled(&turn.config) && !uses_local_route(turn, model_info) {
        cancel(sess).await;
    }
}

/// Installs validated marks before an ordinary request when they release enough.
pub(crate) async fn maybe_clean_history(
    sess: &Arc<Session>,
    step: &Arc<StepContext>,
) -> CodexResult<()> {
    let _boundary = sess.lock_local_compaction_boundary().await;
    let context = LocalCompactionContext::from_step(step);
    let mut state = prepared_state(sess, &context).await;
    let result = install_marks(sess, &context, &mut state, CleanupTrigger::Savings).await;
    sess.set_local_compaction_state(state).await;
    result.map(|_| ())
}

/// Restores durable marks and takes a finished background batch. Callers hold the boundary.
async fn prepared_state(
    sess: &Arc<Session>,
    context: &LocalCompactionContext,
) -> LocalCompactionState {
    let mut state = sess.local_compaction_state().await;
    if !state.restored {
        state.restored = true;
        let history = sess.clone_history().await;
        let mut staged = match marks_paths(sess, context).await {
            Some((own, parent)) => {
                restore_marks(&own, parent.as_deref(), history.annotated_items()).await
            }
            None => StagedDecisions::default(),
        };
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
            Ok(decisions) => state.staged.merge(decisions),
            Err(error) => {
                tracing::warn!(%error, "background local marking failed");
                state.retry_after = Some(Instant::now() + Duration::from_secs(60));
            }
        }
    }
    state
}

/// This thread's mark file and the one of the thread it was forked from. Only threads with a
/// rollout can resume, so ephemeral threads keep their marks in memory.
async fn marks_paths(
    sess: &Session,
    context: &LocalCompactionContext,
) -> Option<(PathBuf, Option<PathBuf>)> {
    sess.live_thread()?;
    let home = context.turn.config.codex_home.as_path();
    let own = marks_path(home, &sess.thread_id.to_string());
    let parent = sess
        .forked_from_thread_id()
        .await
        .map(|parent| marks_path(home, &parent.to_string()));
    Some((own, parent))
}

/// Installs the marked view and returns the released history tokens, or zero if unchanged.
async fn install_marks(
    sess: &Arc<Session>,
    context: &LocalCompactionContext,
    state: &mut LocalCompactionState,
    trigger: CleanupTrigger,
) -> CodexResult<usize> {
    // No model await between this snapshot and the persistence compare-and-swap.
    let history = sess.clone_history().await;
    let source = history.annotated_items();
    state.staged.retain_current(source);
    let (budget, _) = budget(sess, context).await;
    let before_tokens = visible_tokens(
        source,
        context.model_info().retains_prior_reasoning,
        item_tokens,
    );
    let replacement = cleaned(
        &state.staged,
        source,
        reasoning_budget(context, budget),
        item_tokens,
    )
    .map_err(invalid)?;
    let after_tokens = history_tokens(&replacement);
    let wanted = match trigger {
        CleanupTrigger::Savings => budget.useful(before_tokens, after_tokens),
        CleanupTrigger::Manual => after_tokens < before_tokens,
    };
    if !wanted
        || !sess
            .install_local_compaction(crate::session::LocalInstall {
                source: source.to_vec(),
                user_revision: history
                    .conversation_history_snapshot()
                    .user_message_revision(),
                generation: state.generation,
                replacement,
                fresh_baseline: None,
                message: CLEANUP_MESSAGE.to_string(),
                compaction_response_id: None,
                // Tool cleanup has a savings threshold, not a target occupancy.
                max_history_tokens: usize::MAX,
            })
            .await?
    {
        return Ok(0);
    }
    sess.recompute_token_usage(&context.turn).await;
    Ok(before_tokens - after_tokens)
}

const CLEANUP_MESSAGE: &str =
    "Local context cleanup; exact originals are available through recall_read_item.";

/// Budgets in local-estimate units, plus provider tokens per estimated token.
///
/// Local estimates (about 4 bytes per token) run well below provider counts for code-heavy
/// history, while the window and the hard limit are provider counts. Converting those two to
/// estimate units keeps every comparison in one unit.
async fn budget(sess: &Session, context: &LocalCompactionContext) -> (Budget, f64) {
    let config = &context.turn.config.local_compaction;
    let model = context.model_info();
    let window_tokens = model
        .resolved_context_window()
        .unwrap_or(128_000)
        .saturating_mul(model.effective_context_window_percent)
        / 100;
    let overhead = sess.request_overhead().await;
    let fixed_tokens = usize::try_from(
        overhead
            .base_instructions_tokens
            .saturating_add(overhead.tools_tokens),
    )
    .unwrap_or(0);
    // Ratio of the latest provider-reported context to the estimate of what it was sent.
    // Bounded so a stale or estimated report (for example right after a cleanup) falls back
    // to raw estimates.
    let estimated = visible_tokens(
        sess.clone_history().await.annotated_items(),
        model.retains_prior_reasoning,
        item_tokens,
    )
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
    if context.model_info().retains_prior_reasoning {
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
        context.model_info().slug.as_str(),
        context.turn.provider.info().name.as_str(),
    )
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
