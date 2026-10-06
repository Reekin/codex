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
use codex_context_compaction::MAX_ANALYSIS_BYTES;
use codex_context_compaction::MAX_CALL_SUMMARY_BYTES;
use codex_context_compaction::MAX_FRAGMENT_BYTES;
use codex_context_compaction::MAX_SUMMARY_BYTES;
use codex_context_compaction::StagedDecisions;
use codex_context_compaction::WindowPlan;
use codex_context_compaction::dropped_reasoning;
use codex_context_compaction::eligible_results;
use codex_context_compaction::large_calls;
use codex_context_compaction::user_turn;
use codex_history::LocalCompactionKind;
use codex_history::LocalCompactionSource;
use codex_history::ResponseItemEnvelope;
use codex_model_provider::RemoteCompactionSupport;
use codex_protocol::error::CodexErr;
use codex_protocol::error::Result as CodexResult;
use codex_protocol::models::ResponseItem;
use codex_protocol::openai_models::ModelInfo;
use codex_protocol::protocol::EventMsg;
use codex_protocol::protocol::RawResponseCompletedEvent;
use codex_protocol::user_input::UserInput;
use codex_utils_output_truncation::TruncationPolicy;
use codex_utils_output_truncation::approx_token_count;
use codex_utils_output_truncation::approx_tokens_from_byte_count;
use codex_utils_output_truncation::truncate_text;
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
    let (budget, _) = budget(sess, &context).await;
    install_marks(sess, &context, state, CleanupTrigger::Savings).await?;
    let history = sess.clone_history().await;
    let source = history.annotated_items();
    state.staged.retain_current(source);
    if state.background.is_none()
        && state
            .retry_after
            .is_none_or(|retry| Instant::now() >= retry)
    {
        let eligible = eligible_results(source);
        let turn = user_turn(source);
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
                !state.staged.decided(item, turn.as_deref())
                    && item
                        .item
                        .id()
                        .is_some_and(|id| eligible.iter().any(|eligible| eligible == id.as_str()))
            })
            .collect();
        let tokens: usize = unmarked.iter().map(|item| weight(*item)).sum();
        // Every batch resends the whole context, so only enough unmarked volume justifies one.
        let accumulated = tokens
            >= budget
                .window_tokens
                .saturating_mul(usize::from(
                    context
                        .turn
                        .config
                        .local_compaction
                        .mark_after_tokens_percent,
                ))
                .div_ceil(100)
                .max(1);
        // Marks are always useful: full compaction applies them to the window it keeps.
        if !unmarked.is_empty() && accumulated {
            // Each batch costs one full-context request; spend it on the largest outputs.
            // A replacement is never larger than its original, so small outputs cost little
            // response space and many fit in one batch; the request cap trims the rest.
            let mut unmarked = unmarked;
            unmarked.sort_by_key(|item| std::cmp::Reverse(weight(*item)));
            let mut response_bytes = 0_usize;
            let mut candidates: Vec<_> = unmarked
                .into_iter()
                .filter_map(|item| {
                    let mut candidate = request::Candidate::new(&item.item)?;
                    candidate.summarize_call = calls.contains_key(&candidate.id);
                    let worst = item_tokens(&item.item)
                        .saturating_mul(4)
                        .min(MAX_FRAGMENT_BYTES)
                        .saturating_add(candidate.id.len() + 64)
                        .saturating_add(if candidate.summarize_call {
                            MAX_CALL_SUMMARY_BYTES
                        } else {
                            0
                        });
                    Some((candidate, worst))
                })
                .take_while(|(_, worst)| {
                    response_bytes = response_bytes.saturating_add(*worst);
                    response_bytes <= MAX_ANALYSIS_BYTES / 2
                })
                .map(|(candidate, _)| candidate)
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
            let prompt = request::private_prompt(
                Some(step),
                sess.get_prompt_base_instructions().await,
                &source,
                &step.settings.model_info,
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
            let trace = inference_trace(sess, &context);
            let weak = Arc::downgrade(sess);
            let completed = Arc::new(Mutex::new(MarkingResult::default()));
            let result_slot = Arc::clone(&completed);
            let task = tokio::spawn(async move {
                let result = async {
                    let response =
                        request::infer_json(&prompt, &context, &mut client, &metadata, &trace)
                            .await?;
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

pub(crate) async fn run_pipeline(
    sess: &Arc<Session>,
    context: &LocalCompactionContext,
    input: Vec<UserInput>,
    injection: InitialContextInjection,
    metadata: CompactionTurnMetadata,
) -> CodexResult<bool> {
    let _boundary = sess.lock_local_compaction_boundary().await;
    // Validated marks, including a finished batch, clean the window that stays verbatim.
    let mut staged = prepared_state(sess, context).await.staged;
    sess.cancel_local_compaction().await;
    let generation = sess.get_local_compaction_state().await.generation;
    let history = sess.clone_history().await;
    let source = history.annotated_items().to_vec();
    let revision = history
        .conversation_history_snapshot()
        .user_message_revision();
    staged.retain_current(&source);
    let (budget, scale) = budget(sess, context).await;
    let hard_limit = to_estimate(sess.local_compaction_hard_limit(context).await, scale)
        .min(budget.window_tokens);
    // Leave space for continued work.
    let hard_history = hard_limit
        .saturating_sub(budget.fixed_tokens)
        .saturating_sub(1);
    // Tool cleanup keeps records in place, so plan on original indices; trimmed reasoning
    // costs nothing.
    let applied = staged.apply(&source).map_err(invalid)?;
    let mut costs: Vec<_> = applied.iter().map(|item| item_tokens(&item.item)).collect();
    let dropped = dropped_reasoning(&applied, reasoning_budget(context, budget), &costs);
    for (cost, dropped) in costs.iter_mut().zip(&dropped) {
        if *dropped {
            *cost = 0;
        }
    }
    // The summary and its wrapper are bounded in bytes, measured by the same estimator.
    let summary_tokens = usize::try_from(approx_tokens_from_byte_count(MAX_SUMMARY_BYTES + 512))
        .unwrap_or(usize::MAX);
    let plan = WindowPlan::new(
        &source,
        &costs,
        budget.history_target().min(hard_history),
        summary_tokens,
    )
    .map_err(invalid)?;
    let supplemental = input
        .into_iter()
        .filter_map(|item| match item {
            UserInput::Text { text, .. } => Some(text),
            _ => None,
        })
        .collect::<Vec<_>>()
        .join("\n");
    let guidance = if supplemental.is_empty() {
        context
            .turn
            .config
            .compact_prompt
            .clone()
            .unwrap_or_default()
    } else {
        supplemental
    };
    let mut observed_response = None;
    let summary = if let Some(range) = &plan.summarized {
        let request = request::summary(&guidance)?;
        let metadata = sess
            .responses_metadata(
                &context.turn,
                CodexResponsesRequestKind::Compaction(metadata),
            )
            .await;
        let mut client = sess.services.model_client.new_session();
        // Everything before the cut is a prefix of the ordinary request.
        let prompt = request::private_prompt(
            context.step.as_deref(),
            sess.get_prompt_base_instructions().await,
            &source[..plan.cut],
            &context.settings.model_info,
            request,
        );
        let trace = inference_trace(sess, context);
        let mut response =
            request::infer_json(&prompt, context, &mut client, &metadata, &trace).await?;
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
        let text = response.json.trim();
        if text.is_empty() {
            return Err(invalid("the summary is empty"));
        }
        let text = truncate_text(text, TruncationPolicy::Bytes(MAX_SUMMARY_BYTES));
        let summarized = LocalCompactionSource {
            first_item_id: range.first_item_id.clone(),
            last_item_id: range.last_item_id.clone(),
            kind: LocalCompactionKind::OldestOverview,
        };
        let mut envelope =
            ResponseItemEnvelope::new(ContextualUserFragment::into(LocalCompactionFragment {
                text,
                source: summarized.clone(),
            }));
        envelope.metadata.get_or_insert_default().local_compaction = Some(summarized);
        envelope.item.set_turn_id_if_missing(&context.turn.sub_id);
        observed_response = Some(response);
        Some(envelope)
    } else {
        None
    };
    let mut replacement: Vec<_> = plan
        .kept_instructions
        .iter()
        .map(|index| source[*index].clone())
        .collect();
    replacement.extend(summary);
    replacement.extend(plan.kept_input.map(|index| source[index].clone()));
    // The cleaned current window stays verbatim and is summarized by the next compaction.
    for (mut item, dropped) in applied.into_iter().zip(dropped).skip(plan.cut) {
        if !dropped {
            item.metadata.get_or_insert_default().previous_window = true;
            replacement.push(item);
        }
    }
    if history_tokens(&replacement) > hard_history {
        return Err(invalid("compacted history exceeds its safe budget"));
    }
    // Marking the window alone is not worth a checkpoint; the next compaction starts here.
    if replacement
        .iter()
        .map(|item| &item.item)
        .eq(source.iter().map(|item| &item.item))
    {
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
            hard_history,
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
