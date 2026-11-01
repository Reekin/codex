//! Background marking. A batch is sent right after an ordinary request returns, whose prefix it
//! extends, so the provider serves that prefix from the prompt cache. Normal work never waits
//! for it; a later sampling boundary consumes the validated result.

use std::sync::Arc;
use std::sync::Mutex;
use std::time::Duration;
use std::time::Instant;

use codex_analytics::CompactionImplementation;
use codex_analytics::CompactionPhase;
use codex_analytics::CompactionReason;
use codex_analytics::CompactionTrigger;
use codex_context_compaction::Batch;
use codex_context_compaction::BatchLimits;
use codex_context_compaction::StagedDecisions;
use codex_context_compaction::append_marks;
use codex_context_compaction::next_batch;
use codex_protocol::error::Result as CodexResult;

use super::AnalysisKind;
use super::BackgroundMarking;
use super::LocalCompactionContext;
use super::LocalCompactionState;
use super::MarkingResult;
use super::budget;
use super::inference_trace;
use super::invalid;
use super::item_tokens;
use super::marks_paths;
use super::prepared_state;
use super::request;
use crate::responses_metadata::CompactionTurnMetadata;
use crate::session::session::Session;
use crate::session::step_context::StepContext;

/// Called after an ordinary request returns; a batch already in flight is left to finish.
pub(crate) async fn start_marking(sess: &Arc<Session>, step: &Arc<StepContext>) -> CodexResult<()> {
    let _boundary = sess.lock_local_compaction_boundary().await;
    let context = LocalCompactionContext::from_step(step);
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
    let source = history.annotated_items().to_vec();
    state.staged.retain_current(&source);
    let limits = BatchLimits {
        window_tokens: budget.window_tokens,
        start_percent: usize::from(
            context
                .turn
                .config
                .local_compaction
                .mark_after_tokens_percent,
        ),
    };
    let guidance = context
        .turn
        .config
        .compact_prompt
        .clone()
        .unwrap_or_default();
    let (ids, question) = match next_batch(
        &source,
        &state.staged,
        state.backlog.take(),
        limits,
        &guidance,
        item_tokens,
    ) {
        Batch::Idle => return Ok(()),
        Batch::Invalid { backlog, error } => {
            tracing::warn!(%error, "cannot prepare background local marking");
            state.backlog = Some(backlog);
            state.retry_after = Some(Instant::now() + Duration::from_secs(60));
            return Ok(());
        }
        Batch::Send {
            ids,
            request,
            backlog,
        } => {
            state.backlog = backlog;
            (ids, request)
        }
    };
    let prompt = request::private_prompt(sess, step, history, question).await;
    let metadata = sess
        .compaction_responses_metadata(
            &context.turn,
            CompactionTurnMetadata::new(
                CompactionTrigger::Auto,
                CompactionReason::ContextLimit,
                CompactionImplementation::Responses,
                CompactionPhase::MidTurn,
            ),
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
    let effort = request::request_effort(sess, step).await;
    let marks = marks_paths(sess, &context).await.map(|(own, _)| own);
    let weak = Arc::downgrade(sess);
    let step = Arc::clone(step);
    let completed = Arc::new(Mutex::new(MarkingResult::default()));
    let result_slot = Arc::clone(&completed);
    let task = tokio::spawn(async move {
        let result = async {
            let response =
                request::infer_text(&prompt, &step, effort, &mut client, &metadata, &trace).await?;
            if let Some(sess) = weak.upgrade() {
                if let Some(rate_limits) = response.rate_limits {
                    sess.record_rate_limits_info(rate_limits).await;
                }
                sess.record_local_compaction_usage(
                    &context.turn,
                    &context.settings,
                    &response.response_id,
                    response.usage.as_ref(),
                    AnalysisKind::Background,
                )
                .await?;
            }
            let decisions =
                StagedDecisions::parse_candidates(source, &ids, &response.text).map_err(invalid)?;
            if let Some(path) = marks.as_deref() {
                append_marks(path, &decisions).await;
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
