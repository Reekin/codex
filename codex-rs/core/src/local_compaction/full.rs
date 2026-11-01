//! Full compaction on the local route: a fresh instruction set, one handoff summary of what
//! leaves the window, and the cleaned current window kept verbatim.

use std::collections::HashSet;
use std::sync::Arc;

use codex_context_compaction::MAX_SUMMARY_BYTES;
use codex_context_compaction::Removal;
use codex_context_compaction::WindowPlan;
use codex_context_compaction::dropped_reasoning;
use codex_context_compaction::fold_tool_runs;
use codex_context_compaction::remove_parts;
use codex_context_compaction::same_beyond_rendered;
use codex_context_compaction::summary_record;
use codex_context_compaction::summary_request;
use codex_extension_api::ConversationHistorySnapshot as _;
use codex_history::ResponseItemEnvelope;
use codex_protocol::error::Result as CodexResult;
use codex_protocol::models::ResponseItem;
use codex_protocol::protocol::EventMsg;
use codex_protocol::protocol::RawResponseCompletedEvent;
use codex_utils_output_truncation::approx_tokens_from_byte_count;

use super::AnalysisKind;
use super::LocalCompactionContext;
use super::budget;
use super::history_tokens;
use super::inference_trace;
use super::invalid;
use super::item_tokens;
use super::prepared_state;
use super::reasoning_budget;
use super::request;
use super::to_estimate;
use crate::context::world_state::WorldState;
use crate::context::world_state::split_prefix_updates;
use crate::context_manager::updates::merge_world_state_updates;
use crate::responses_metadata::CompactionTurnMetadata;
use crate::session::LocalInstall;
use crate::session::session::Session;
use crate::session::step_context::StepContext;

const COMPACTION_MESSAGE: &str =
    "Local context compaction; exact originals are available through recall_read_item.";

/// `compactor` sends the summary request; `replacement` owns the new window: its instructions,
/// model window and kept reasoning. They differ when a previous model compacts for a new one.
///
/// Codex's instructions are rendered afresh from the current state and lead the new history,
/// replacing every earlier copy: records ahead of the kept window are dropped, and inside it
/// the parts of every kind rendered again are removed, as are the window's tool declarations.
/// Kinds are matched exactly, so notices that only carried a change of another kind stay where
/// they were. Returns whether a new history was installed.
pub(crate) async fn run_pipeline(
    sess: &Arc<Session>,
    compactor: &Arc<StepContext>,
    replacement: &Arc<StepContext>,
    world_state: &WorldState,
    metadata: CompactionTurnMetadata,
) -> CodexResult<bool> {
    let _boundary = sess.lock_local_compaction_boundary().await;
    let context = LocalCompactionContext::from_step(replacement);
    let (updates, baseline) = sess
        .build_initial_context_with_world_state(replacement, world_state)
        .await;
    let (prefix, updates) = split_prefix_updates(updates);
    let fresh: Vec<_> = prefix
        .into_iter()
        .chain(merge_world_state_updates(updates))
        .map(ResponseItemEnvelope::new)
        .collect();
    let fresh_kinds: HashSet<String> = fresh
        .iter()
        .filter_map(|item| match &item.item {
            ResponseItem::Message {
                internal_chat_message_metadata_passthrough: Some(metadata),
                ..
            } => metadata.content_item_kinds.as_ref(),
            _ => None,
        })
        .flatten()
        .map(|kind| kind.0.clone())
        .collect();
    // Validated marks, including a finished batch, clean the window that stays verbatim; marking
    // for the old window stops here.
    let mut staged = prepared_state(sess, &context).await.staged;
    sess.cancel_local_compaction().await;
    let generation = sess.local_compaction_state().await.generation;
    let history = sess.clone_history().await;
    let source = history.annotated_items().to_vec();
    let user_revision = history
        .conversation_history_snapshot()
        .user_message_revision();
    staged.retain_current(&source);
    let (budget, scale) = budget(sess, &context).await;
    let hard_limit = to_estimate(
        sess.local_compaction_hard_limit(&context.turn, &context.settings.model_info)
            .await,
        scale,
    )
    .min(budget.window_tokens);
    // Leave space for continued work.
    let hard_history = hard_limit
        .saturating_sub(budget.fixed_tokens)
        .saturating_sub(1);
    let mut applied = staged.apply(&source).map_err(invalid)?;
    let costs: Vec<_> = applied.iter().map(|item| item_tokens(&item.item)).collect();
    let mut dropped = dropped_reasoning(&applied, reasoning_budget(&context, budget), &costs);
    for (item, dropped) in applied.iter_mut().zip(&mut dropped) {
        if matches!(item.item, ResponseItem::AdditionalTools { .. })
            || remove_parts(&mut item.item, &fresh_kinds) == Removal::All
        {
            *dropped = true;
        }
    }
    // The plan works on the cleaned history, with low-value tool work folded into one line
    // per run.
    let working = fold_tool_runs(applied, dropped, &staged);
    let costs: Vec<_> = working.iter().map(|item| item_tokens(&item.item)).collect();
    // The summary and its wrapper are bounded in bytes, measured by the same estimator.
    let summary_tokens = usize::try_from(approx_tokens_from_byte_count(MAX_SUMMARY_BYTES + 512))
        .unwrap_or(usize::MAX);
    let plan = WindowPlan::new(
        &working,
        &costs,
        budget
            .history_target()
            .min(hard_history)
            .saturating_sub(history_tokens(&fresh)),
        summary_tokens,
    )
    .map_err(invalid)?;
    let mut observed_response = None;
    let summary = match &plan.summarized {
        Some(range) => {
            let guidance = context
                .turn
                .config
                .compact_prompt
                .clone()
                .unwrap_or_default();
            let question = summary_request(&guidance).map_err(invalid)?;
            let compactor_context = LocalCompactionContext::from_step(compactor);
            let responses_metadata = sess
                .compaction_responses_metadata(&compactor.turn, metadata)
                .await;
            let mut client = sess.services.model_client.new_session();
            // The whole history is the ordinary request's prefix, so it is served from the
            // prompt cache; a truncated prefix would miss it on providers that cache at request
            // boundaries.
            let prompt = request::private_prompt(sess, compactor, history, question).await;
            let trace = inference_trace(sess, &compactor_context);
            let effort = request::request_effort(sess, compactor).await;
            let mut response = request::infer_text(
                &prompt,
                compactor,
                effort,
                &mut client,
                &responses_metadata,
                &trace,
            )
            .await?;
            if let Some(rate_limits) = response.rate_limits.take() {
                sess.record_rate_limits_info(rate_limits).await;
            }
            sess.record_local_compaction_usage(
                &compactor.turn,
                &compactor.settings,
                &response.response_id,
                response.usage.as_ref(),
                AnalysisKind::Full,
            )
            .await?;
            let record =
                summary_record(&response.text, range, &compactor.turn.sub_id).map_err(invalid)?;
            observed_response = Some(response);
            Some(record)
        }
        None => None,
    };
    let fresh_len = fresh.len();
    let replacement_history = plan.replacement(fresh, working, summary);
    // Rendering the instructions again is not worth a checkpoint by itself: with nothing
    // summarized and every other record unchanged, the turn continues as it was, whatever the
    // budget says, and the next compaction starts here.
    if plan.summarized.is_none() && same_beyond_rendered(&replacement_history[fresh_len..], &source)
    {
        return Ok(false);
    }
    if history_tokens(&replacement_history) > hard_history {
        return Err(invalid("compacted history exceeds its safe budget"));
    }
    let installed = sess
        .install_local_compaction(LocalInstall {
            source,
            user_revision,
            generation,
            replacement: replacement_history,
            // The fresh instructions are the baseline later steps diff against.
            fresh_baseline: Some((replacement.to_turn_context_item(), baseline)),
            message: COMPACTION_MESSAGE.to_string(),
            compaction_response_id: observed_response
                .as_ref()
                .map(|response| response.response_id.clone()),
            max_history_tokens: hard_history,
        })
        .await?;
    if installed {
        if let Some(response) = observed_response {
            // Full compaction keeps its completion event, sent after validated installation.
            // Usage was already charged privately; this event does not charge it again.
            sess.send_event(
                &compactor.turn,
                EventMsg::RawResponseCompleted(RawResponseCompletedEvent {
                    response_id: response.response_id,
                    token_usage: response.usage,
                    usage_metadata: response.usage_metadata,
                }),
            )
            .await;
        }
        sess.recompute_token_usage(&replacement.turn).await;
    }
    Ok(installed)
}
