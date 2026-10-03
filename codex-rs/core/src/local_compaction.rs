//! Private background marking and atomic installation at ordinary sampling boundaries.

mod marks;
mod request;
mod tool_cleanup;

pub use tool_cleanup::ToolCleanupOutcome;
pub use tool_cleanup::ToolCleanupStatus;
pub(crate) use tool_cleanup::apply_tool_cleanup;
pub(crate) use tool_cleanup::tool_cleanup_status;

use crate::compact::CompactedHistoryMetadata;
use crate::compact::InitialContextInjection;
use crate::compact::LocalCompactionContext;
use crate::context::ContextualUserFragment;
use crate::context::LocalCompactionFragment;
use crate::context_manager::estimate_item_token_count;
use crate::responses_metadata::CodexResponsesRequestKind;
use crate::responses_metadata::CompactionTurnMetadata;
use crate::session::session::Session;
use crate::session::step_context::StepContext;
use crate::session::turn_context::TurnContext;
use codex_analytics::CompactionImplementation;
use codex_analytics::CompactionPhase;
use codex_analytics::CompactionReason;
use codex_analytics::CompactionTrigger;
use codex_context_compaction::Budget;
use codex_context_compaction::StagedDecisions;
use codex_context_compaction::TierPlan;
use codex_context_compaction::eligible_results;
use codex_context_compaction::large_calls;
use codex_history::ResponseItemEnvelope;
use codex_model_provider::RemoteCompactionSupport;
use codex_protocol::error::CodexErr;
use codex_protocol::error::Result as CodexResult;
use codex_protocol::openai_models::ModelInfo;
use codex_protocol::protocol::EventMsg;
use codex_protocol::protocol::RawResponseCompletedEvent;
use codex_protocol::user_input::UserInput;
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
    Savings(Budget),
    Manual,
}

/// Whether tool marking and full compaction use the local pipeline for this model.
pub(crate) fn uses_local_route(turn: &TurnContext, model_info: &ModelInfo) -> bool {
    turn.config.local_compaction.force_local
        || crate::compaction_policy::remote_compaction_support(
            turn.provider.capabilities().remote_compaction,
            model_info,
        ) == RemoteCompactionSupport::Unsupported
}

pub(crate) async fn maybe_clean_history(
    sess: &Arc<Session>,
    step: &Arc<StepContext>,
) -> CodexResult<()> {
    let _boundary = sess.lock_local_compaction_boundary().await;
    let context = LocalCompactionContext::from_step(Arc::clone(step));
    let mut state = prepared_state(sess, &context).await;
    let result = clean_and_launch(sess, step, context, &mut state).await;
    // A concurrent full-window reset owns the new generation and discards this batch.
    sess.set_local_compaction_state(state).await;
    result
}

async fn clean_and_launch(
    sess: &Arc<Session>,
    step: &Arc<StepContext>,
    context: LocalCompactionContext,
    state: &mut LocalCompactionState,
) -> CodexResult<()> {
    let budget = budget(sess, &context).await;
    install_marks(sess, &context, state, CleanupTrigger::Savings(budget)).await?;
    let history = sess.clone_history().await;
    let source = history.annotated_items();
    state.staged.retain_current(source);
    if state.background.is_none()
        && state
            .retry_after
            .is_none_or(|retry| Instant::now() >= retry)
    {
        // A tool cleanup can reset body-after-prefix accounting; use the post-cleanup limit.
        let hard_limit = sess.local_compaction_hard_limit(&context).await;
        let eligible = eligible_results(source);
        // A result's weight includes the large call it is shortened together with.
        let calls = large_calls(source);
        let weight = |item: &ResponseItemEnvelope| {
            item_tokens(&item.item).saturating_add(
                item.item
                    .id()
                    .and_then(|id| calls.get(id.as_str()))
                    .map_or(0, |&call| item_tokens(&source[call].item)),
            )
        };
        let unmarked: Vec<_> = source
            .iter()
            .filter(|item| {
                !state.staged.contains(item)
                    && item
                        .item
                        .id()
                        .is_some_and(|id| eligible.iter().any(|eligible| eligible == id.as_str()))
            })
            .collect();
        let tokens: usize = unmarked.iter().map(|item| weight(*item)).sum();
        let config = &context.turn.config.local_compaction;
        let accumulated = unmarked.len() >= usize::from(config.mark_after_records)
            || tokens
                >= budget
                    .window_tokens
                    .saturating_mul(usize::from(config.mark_after_tokens_percent))
                    .div_ceil(100)
                    .max(1);
        let total = history_tokens(source);
        let pending = total.saturating_sub(history_tokens(
            &state.staged.apply(source).map_err(invalid)?,
        ));
        let upper =
            total.saturating_sub(history_tokens(&state.staged.optimistic_replacement(source)));
        if !unmarked.is_empty()
            && accumulated
            && budget.can_reach(
                pending,
                upper,
                total.saturating_add(budget.fixed_tokens),
                hard_limit,
            )
        {
            // Each batch costs one full-context request; spend it on the largest outputs.
            // The cap bounds the response so a full batch of shortens stays under its limit.
            let mut unmarked = unmarked;
            unmarked.sort_by_key(|item| std::cmp::Reverse(weight(*item)));
            let mut candidates: Vec<_> = unmarked
                .into_iter()
                .filter_map(|item| {
                    let mut candidate = request::Candidate::new(&item.item)?;
                    candidate.summarize_call = calls.contains_key(&candidate.id);
                    Some(candidate)
                })
                .take(64)
                .collect();
            let source = source.to_vec();
            let request = match request::classifier(
                &mut candidates,
                context.turn.config.compact_prompt.as_deref().unwrap_or(""),
            ) {
                Ok(request) => request,
                Err(error) => {
                    tracing::warn!(%error, "cannot prepare background local marking");
                    state.retry_after = Some(Instant::now() + Duration::from_secs(60));
                    return Ok(());
                }
            };
            let ids: Vec<_> = candidates
                .into_iter()
                .map(|candidate| candidate.id)
                .collect();
            let prompt = request::classifier_prompt(
                step,
                sess.get_prompt_base_instructions().await,
                &source,
                request,
            );
            let metadata = sess
                .responses_metadata(
                    &context.turn,
                    CodexResponsesRequestKind::Compaction(CompactionTurnMetadata::new(
                        CompactionTrigger::Auto,
                        CompactionReason::ContextLimit,
                        CompactionImplementation::Responses,
                        CompactionPhase::MidTurn,
                    )),
                )
                .await;
            let mut client = sess.services.model_client.new_session();
            if !sess
                .begin_local_compaction_usage(&context.turn.sub_id, state.generation)
                .await
            {
                return Ok(());
            }
            let weak = Arc::downgrade(sess);
            let completed = Arc::new(Mutex::new(MarkingResult::default()));
            let result_slot = Arc::clone(&completed);
            let task = tokio::spawn(async move {
                let result = async {
                    let response =
                        request::infer_json(&prompt, &context, &mut client, &metadata).await?;
                    if let Some(sess) = weak.upgrade() {
                        if let Some(rate_limits) = response.rate_limits {
                            sess.record_rate_limits_info(rate_limits).await;
                        }
                        sess.record_local_compaction_usage(
                            &context.turn,
                            &response.response_id,
                            response.usage.as_ref(),
                            AnalysisKind::Background,
                        )
                        .await?;
                    }
                    let decisions = StagedDecisions::parse_candidates(source, &ids, &response.json)
                        .map_err(invalid)?;
                    if let Some(path) = weak
                        .upgrade()
                        .and_then(|sess| marks::path(&sess, &context.turn.config))
                    {
                        marks::append(path, &decisions).await;
                    }
                    Ok(decisions)
                }
                .await;
                *result_slot.lock().expect("marking result lock") = MarkingResult {
                    value: Some(result),
                    finished: true,
                };
                if let Some(sess) = weak.upgrade() {
                    // This event is also the observable ready barrier. Preserve the foreground
                    // snapshot; this request may belong to a model used in an earlier turn.
                    sess.send_local_compaction_token_count(&context.turn).await;
                }
            });
            state.background = Some(Arc::new(BackgroundMarking {
                abort: task.abort_handle(),
                completed,
            }));
            state.retry_after = None;
        }
    }
    Ok(())
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
    let before_tokens = history_tokens(source);
    let replacement = state.staged.apply(source).map_err(invalid)?;
    let after_tokens = history_tokens(&replacement);
    let wanted = match trigger {
        CleanupTrigger::Savings(budget) => budget.useful(before_tokens, after_tokens),
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

pub(crate) async fn run_pipeline(
    sess: &Arc<Session>,
    context: &LocalCompactionContext,
    input: Vec<UserInput>,
    injection: InitialContextInjection,
    metadata: CompactionTurnMetadata,
) -> CodexResult<bool> {
    sess.cancel_local_compaction().await;
    let generation = sess.get_local_compaction_state().await.generation;
    let history = sess.clone_history().await;
    let source = history.annotated_items().to_vec();
    let revision = history
        .conversation_history_snapshot()
        .user_message_revision();
    let budget = budget(sess, context).await;
    let hard_limit = sess
        .local_compaction_hard_limit(context)
        .await
        .min(budget.window_tokens);
    // Leave space for continued work; the preferred ratio may be exceeded.
    let hard_history = hard_limit
        .saturating_sub(budget.fixed_tokens)
        .saturating_sub(1);
    let costs: Vec<_> = source.iter().map(|item| item_tokens(&item.item)).collect();
    let plan =
        TierPlan::new(&source, budget.history_target(), hard_history, &costs).map_err(invalid)?;
    let max_history = plan.result_budget_tokens.min(hard_history);
    let supplemental = input
        .into_iter()
        .filter_map(|item| match item {
            UserInput::Text { text, .. } => Some(text),
            _ => None,
        })
        .collect::<Vec<_>>()
        .join("\n");
    let mut observed_response = None;
    let fragments = if plan.l2.is_some() || plan.l3.is_some() {
        let request = request::summarizer(&plan, &supplemental)?;
        let metadata = sess
            .responses_metadata(
                &context.turn,
                CodexResponsesRequestKind::Compaction(metadata),
            )
            .await;
        let mut client = sess.services.model_client.new_session();
        let prompt = request::summarizer_prompt(
            context,
            sess.get_prompt_base_instructions().await,
            &source,
            request,
        )?;
        let mut response = request::infer_json(&prompt, context, &mut client, &metadata).await?;
        if let Some(rate_limits) = response.rate_limits.take() {
            sess.record_rate_limits_info(rate_limits).await;
        }
        sess.record_local_compaction_usage(
            &context.turn,
            &response.response_id,
            response.usage.as_ref(),
            AnalysisKind::Full,
        )
        .await?;
        let fragments = plan.parse(&response.json).map_err(invalid)?;
        observed_response = Some(response);
        fragments
    } else {
        Vec::new()
    };
    let mut replacement = plan.retained_prefix;
    for fragment in fragments {
        let mut envelope =
            ResponseItemEnvelope::new(ContextualUserFragment::into(LocalCompactionFragment {
                text: fragment.text,
                source: fragment.source.clone(),
            }));
        envelope.metadata.get_or_insert_default().local_compaction = Some(fragment.source);
        envelope.item.set_turn_id_if_missing(&context.turn.sub_id);
        replacement.push(envelope);
    }
    replacement.extend(plan.retained_tail);
    if history_tokens(&replacement) > max_history {
        return Err(invalid("tier result exceeds its safe history budget"));
    }
    if replacement == source {
        return Ok(false);
    }
    let baseline = match injection {
        InitialContextInjection::BeforeLastUserMessage { world_state, .. } => Some(world_state),
        InitialContextInjection::DoNotInject => None,
    };
    let installed = sess
        .install_local_compaction(
            &source,
            revision,
            generation,
            replacement,
            sess.reference_context_item().await,
            baseline,
            checkpoint(
                context,
                observed_response
                    .as_ref()
                    .map(|response| response.response_id.clone()),
            ),
            max_history,
        )
        .await?;
    if installed {
        if let Some(response) = observed_response {
            // Full compaction retains its existing completion event after validated install.
            // Usage was already charged privately; emitting this event must not charge twice.
            sess.send_event(
                &context.turn,
                EventMsg::RawResponseCompleted(RawResponseCompletedEvent {
                    response_id: response.response_id,
                    token_usage: response.usage,
                    usage_metadata: response.usage_metadata,
                }),
            )
            .await;
        }
        sess.recompute_token_usage(&context.turn).await;
    }
    Ok(installed)
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

async fn budget(sess: &Session, context: &LocalCompactionContext) -> Budget {
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
    Budget {
        window_tokens: usize::try_from(window_tokens).unwrap_or(0),
        fixed_tokens: approx_token_count(&base.text).saturating_add(tool_tokens),
        reclaim_percent: usize::from(config.reclaim_percent),
        compact_target_percent: usize::from(config.compact_target_percent),
    }
}

fn item_tokens(item: &codex_protocol::models::ResponseItem) -> usize {
    usize::try_from(estimate_item_token_count(item)).unwrap_or(0)
}

fn history_tokens(items: &[ResponseItemEnvelope]) -> usize {
    items.iter().map(|item| item_tokens(&item.item)).sum()
}

fn invalid(error: impl std::fmt::Display) -> CodexErr {
    CodexErr::InvalidRequest(format!("Local compaction failed: {error}"))
}
