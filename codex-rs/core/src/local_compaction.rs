//! Private background marking and atomic installation at ordinary sampling boundaries.

mod request;

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
use codex_analytics::CompactionImplementation;
use codex_analytics::CompactionPhase;
use codex_analytics::CompactionReason;
use codex_analytics::CompactionTrigger;
use codex_context_compaction::Budget;
use codex_context_compaction::StagedDecisions;
use codex_context_compaction::TierPlan;
use codex_context_compaction::eligible_results;
use codex_history::ResponseItemEnvelope;
use codex_protocol::error::CodexErr;
use codex_protocol::error::Result as CodexResult;
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
    background: Option<Arc<BackgroundMarking>>,
    retry_after: Option<Instant>,
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
            ..Self::default()
        };
    }
}

pub(crate) async fn maybe_clean_history(
    sess: &Arc<Session>,
    step: &Arc<StepContext>,
) -> CodexResult<bool> {
    let context = LocalCompactionContext::from_step(Arc::clone(step));
    let budget = budget(sess, &context).await;
    let mut state = sess.get_local_compaction_state().await;
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
    // No model await between this snapshot and the persistence CAS.
    let history = sess.clone_history().await;
    let source = history.annotated_items();
    state.staged.retain_current(source);
    let before_tokens = history_tokens(source);
    let replacement = state.staged.apply(source).map_err(invalid)?;
    let after_tokens = history_tokens(&replacement);
    let mut installed = false;
    if budget.useful(before_tokens, after_tokens) {
        installed = sess
            .install_local_compaction(
                source,
                history
                    .conversation_history_snapshot()
                    .user_message_revision(),
                state.generation,
                replacement,
                sess.reference_context_item().await,
                None,
                checkpoint(&context, None),
                // Tool cleanup has a savings threshold, not a target occupancy.
                usize::MAX,
            )
            .await?;
        if installed {
            sess.recompute_token_usage(&context.turn).await;
        }
    }
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
        let tokens: usize = unmarked.iter().map(|item| item_tokens(&item.item)).sum();
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
            // Bound each request's IDs and response while leaving the rest unmarked.
            let ids: Vec<_> = unmarked
                .into_iter()
                .take(64)
                .filter_map(|item| item.item.id().map(ToString::to_string))
                .collect();
            let source = source.to_vec();
            let request = match request::classifier(
                &ids,
                context.turn.config.compact_prompt.as_deref().unwrap_or(""),
            ) {
                Ok(request) => request,
                Err(error) => {
                    tracing::warn!(%error, "cannot prepare background local marking");
                    state.retry_after = Some(Instant::now() + Duration::from_secs(60));
                    sess.set_local_compaction_state(state).await;
                    return Ok(installed);
                }
            };
            let base = sess.get_prompt_base_instructions().await;
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
                return Ok(installed);
            }
            let weak = Arc::downgrade(sess);
            let completed = Arc::new(Mutex::new(MarkingResult::default()));
            let result_slot = Arc::clone(&completed);
            let task = tokio::spawn(async move {
                let result = async {
                    let response = request::infer_json(
                        base,
                        &context,
                        &mut client,
                        &metadata,
                        &source,
                        request,
                    )
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
                    StagedDecisions::parse_candidates(source, &ids, &response.json).map_err(invalid)
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
    // A concurrent full-window reset owns the new generation and discards this batch.
    sess.set_local_compaction_state(state).await;
    Ok(installed)
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
        let mut response = request::infer_json(
            sess.get_prompt_base_instructions().await,
            context,
            &mut client,
            &metadata,
            &source,
            request,
        )
        .await?;
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
