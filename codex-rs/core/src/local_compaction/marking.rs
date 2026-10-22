//! Background marking. Every batch resends the whole context, so marking starts once enough
//! tool output is unmarked, then works through the backlog present at that point, oldest
//! first and one batch at a time; later output waits for the next start. A batch is sent
//! right after an ordinary request returns, whose prefix it extends, so the provider serves
//! that prefix from the prompt cache.

use std::collections::HashSet;
use std::sync::Arc;
use std::sync::Mutex;
use std::time::Duration;
use std::time::Instant;

use codex_analytics::CompactionImplementation;
use codex_analytics::CompactionPhase;
use codex_analytics::CompactionReason;
use codex_analytics::CompactionTrigger;
use codex_context_compaction::MAX_ANALYSIS_BYTES;
use codex_context_compaction::MAX_CALL_SUMMARY_BYTES;
use codex_context_compaction::MAX_FRAGMENT_BYTES;
use codex_context_compaction::StagedDecisions;
use codex_context_compaction::eligible_results;
use codex_context_compaction::large_calls;
use codex_context_compaction::user_turn;
use codex_history::ResponseItemEnvelope;
use codex_protocol::error::Result as CodexResult;

use super::AnalysisKind;
use super::BackgroundMarking;
use super::LocalCompactionState;
use super::MarkingResult;
use super::budget;
use super::inference_trace;
use super::invalid;
use super::item_tokens;
use super::marks;
use super::prepared_state;
use super::request;
use crate::compact::LocalCompactionContext;
use crate::responses_metadata::CodexResponsesRequestKind;
use crate::responses_metadata::CompactionTurnMetadata;
use crate::session::session::Session;
use crate::session::step_context::StepContext;

/// Called after an ordinary request returns; a batch already in flight is left to finish.
pub(crate) async fn start_marking(sess: &Arc<Session>, step: &Arc<StepContext>) -> CodexResult<()> {
    let _boundary = sess.lock_local_compaction_boundary().await;
    let context = LocalCompactionContext::from_step(Arc::clone(step));
    let mut state = prepared_state(sess, &context).await;
    let result = launch(sess, step, context, &mut state).await;
    // A concurrent full-window reset owns the new generation and discards this batch.
    sess.set_local_compaction_state(state).await;
    result
}

async fn launch(
    sess: &Arc<Session>,
    step: &Arc<StepContext>,
    context: LocalCompactionContext,
    state: &mut LocalCompactionState,
) -> CodexResult<()> {
    if state.background.is_some()
        || state
            .retry_after
            .is_some_and(|retry| Instant::now() < retry)
    {
        return Ok(());
    }
    let (budget, _) = budget(sess, &context).await;
    let history = sess.clone_history().await;
    let source = history.annotated_items();
    state.staged.retain_current(source);
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
    // History order is oldest first.
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
    let share = |percent: usize| {
        budget
            .window_tokens
            .saturating_mul(percent)
            .div_ceil(100)
            .max(1)
    };
    let start = share(usize::from(
        context
            .turn
            .config
            .local_compaction
            .mark_after_tokens_percent,
    ));
    let backlog: HashSet<String> = match state.backlog.take() {
        Some(backlog) => backlog,
        None if unmarked.iter().map(|item| weight(item)).sum::<usize>() >= start => unmarked
            .iter()
            .filter_map(|item| item.item.id().map(ToString::to_string))
            .collect(),
        None => return Ok(()),
    };
    let mut response_bytes = 0_usize;
    let mut batch_tokens = 0_usize;
    let mut full = false;
    let mut candidates = Vec::new();
    for item in unmarked.iter().filter(|item| {
        item.item
            .id()
            .is_some_and(|id| backlog.contains(id.as_str()))
    }) {
        let Some(mut candidate) = request::Candidate::new(&item.item) else {
            continue;
        };
        candidate.summarize_call = calls.contains_key(&candidate.id);
        // A replacement is never larger than its original, so small outputs cost little
        // response space and many fit in one batch.
        let worst = item_tokens(&item.item)
            .saturating_mul(4)
            .min(MAX_FRAGMENT_BYTES)
            .saturating_add(candidate.id.len() + 64)
            .saturating_add(if candidate.summarize_call {
                MAX_CALL_SUMMARY_BYTES
            } else {
                0
            });
        if response_bytes.saturating_add(worst) > MAX_ANALYSIS_BYTES / 2 {
            full = true;
            break;
        }
        response_bytes += worst;
        batch_tokens += weight(item);
        candidates.push(candidate);
    }
    let packed = candidates.len();
    let request = match request::classifier(
        &mut candidates,
        context.turn.config.compact_prompt.as_deref().unwrap_or(""),
    ) {
        Ok(request) => request,
        Err(error) => {
            tracing::warn!(%error, "cannot prepare background local marking");
            state.backlog = Some(backlog);
            state.retry_after = Some(Instant::now() + Duration::from_secs(60));
            return Ok(());
        }
    };
    full |= candidates.len() < packed;
    // A short remainder waits for the next backlog rather than paying for a request alone.
    if candidates.is_empty() || (!full && batch_tokens < start.min(share(1))) {
        return Ok(());
    }
    state.backlog = Some(backlog);
    let ids: Vec<_> = candidates
        .into_iter()
        .map(|candidate| candidate.id)
        .collect();
    let source = source.to_vec();
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
                request::infer_json(&prompt, &context, &mut client, &metadata, &trace).await?;
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
            let decisions =
                StagedDecisions::parse_candidates(source, &ids, &response.json).map_err(invalid)?;
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
    Ok(())
}
