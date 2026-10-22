//! Full compaction: a fresh instruction set, the handoff summary, and the cleaned current window.

use std::collections::HashSet;
use std::sync::Arc;

use codex_context_compaction::MAX_SUMMARY_BYTES;
use codex_context_compaction::Removal;
use codex_context_compaction::ToolRun;
use codex_context_compaction::WindowPlan;
use codex_context_compaction::dropped_reasoning;
use codex_context_compaction::remove_parts;
use codex_context_compaction::rendered;
use codex_context_compaction::tool_runs;
use codex_history::LocalCompactionKind;
use codex_history::LocalCompactionSource;
use codex_history::ResponseItemEnvelope;
use codex_protocol::error::Result as CodexResult;
use codex_protocol::models::ResponseItem;
use codex_protocol::protocol::EventMsg;
use codex_protocol::protocol::RawResponseCompletedEvent;
use codex_protocol::user_input::UserInput;
use codex_utils_output_truncation::TruncationPolicy;
use codex_utils_output_truncation::approx_tokens_from_byte_count;
use codex_utils_output_truncation::truncate_text;

use super::AnalysisKind;
use super::budget;
use super::checkpoint;
use super::history_tokens;
use super::inference_trace;
use super::invalid;
use super::item_tokens;
use super::prepared_state;
use super::reasoning_budget;
use super::request;
use super::to_estimate;
use crate::compact::InitialContextInjection;
use crate::compact::LocalCompactionContext;
use crate::context::ContextualUserFragment;
use crate::context::LocalCompactionFragment;
use crate::responses_metadata::CodexResponsesRequestKind;
use crate::responses_metadata::CompactionTurnMetadata;
use crate::session::session::Session;

/// Codex's instructions are rendered afresh from the current state and lead the new history,
/// replacing every earlier copy: records ahead of the kept window are dropped, and inside it
/// the parts of every kind rendered again are removed. Kinds are matched exactly, so notices
/// that only carried a change of another kind stay where they were.
pub(crate) async fn run_pipeline(
    sess: &Arc<Session>,
    context: &LocalCompactionContext,
    input: Vec<UserInput>,
    injection: InitialContextInjection,
    metadata: CompactionTurnMetadata,
) -> CodexResult<bool> {
    let _boundary = sess.lock_local_compaction_boundary().await;
    let step = context
        .step
        .as_ref()
        .ok_or_else(|| invalid("full compaction needs a sampling step"))?;
    let world_state = match injection {
        InitialContextInjection::BeforeLastUserMessage { world_state, .. } => world_state,
        InitialContextInjection::DoNotInject => {
            Arc::new(sess.build_world_state_for_step(step).await?)
        }
    };
    let fresh: Vec<_> = sess
        .build_initial_context_with_world_state(&context.turn, &world_state)
        .await
        .into_iter()
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
    let mut applied = staged.apply(&source).map_err(invalid)?;
    let costs: Vec<_> = applied.iter().map(|item| item_tokens(&item.item)).collect();
    let mut dropped = dropped_reasoning(&applied, reasoning_budget(context, budget), &costs);
    for (item, dropped) in applied.iter_mut().zip(&mut dropped) {
        if remove_parts(&mut item.item, &fresh_kinds) == Removal::All {
            *dropped = true;
        }
    }
    // The plan works on the cleaned history, with low-value tool work folded into one line
    // per run.
    let runs = tool_runs(&applied, &staged, &dropped, 0);
    let folded: HashSet<usize> = runs.iter().flat_map(|run| run.indices.clone()).collect();
    let mut records: Vec<_> = runs
        .iter()
        .map(|run| (run.indices[0], tool_run_record(&applied, run)))
        .collect();
    records.reverse();
    let mut working = Vec::with_capacity(applied.len());
    for (index, (item, dropped)) in applied.into_iter().zip(dropped).enumerate() {
        if records.last().is_some_and(|(first, _)| *first == index)
            && let Some((_, record)) = records.pop()
        {
            working.push(record);
        }
        if !dropped && !folded.contains(&index) {
            working.push(item);
        }
    }
    let costs: Vec<_> = working.iter().map(|item| item_tokens(&item.item)).collect();
    let fresh_tokens = history_tokens(&fresh);
    // The summary and its wrapper are bounded in bytes, measured by the same estimator.
    let summary_tokens = usize::try_from(approx_tokens_from_byte_count(MAX_SUMMARY_BYTES + 512))
        .unwrap_or(usize::MAX);
    let plan = WindowPlan::new(
        &working,
        &costs,
        budget
            .history_target()
            .min(hard_history)
            .saturating_sub(fresh_tokens),
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
        // The whole history is the ordinary request's prefix, so it is served from the
        // prompt cache; a truncated prefix would miss it on providers that cache at request
        // boundaries.
        let prompt = request::private_prompt(
            context.step.as_deref(),
            sess.get_prompt_base_instructions().await,
            &source,
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
    let fresh_len = fresh.len();
    let mut replacement = fresh;
    replacement.extend(
        plan.kept_context
            .iter()
            .map(|index| working[*index].clone()),
    );
    replacement.extend(summary);
    replacement.extend(plan.kept_input.map(|index| working[index].clone()));
    // What stays of the current window is verbatim and summarized by the next compaction.
    for (mut item, removed) in working
        .into_iter()
        .zip(plan.removed)
        .skip(plan.window_start)
    {
        if !removed {
            item.metadata.get_or_insert_default().previous_window = true;
            replacement.push(item);
        }
    }
    // Rendering the instructions again is not worth a checkpoint by itself: with nothing
    // summarized and every other record unchanged, the turn continues as it was, whatever the
    // budget says, and the next compaction starts here.
    let others = |items: &[ResponseItemEnvelope]| {
        items
            .iter()
            .filter(|item| !rendered(&item.item))
            .map(|item| item.item.clone())
            .collect::<Vec<_>>()
    };
    if plan.summarized.is_none() && others(&replacement[fresh_len..]) == others(&source) {
        return Ok(false);
    }
    if history_tokens(&replacement) > hard_history {
        return Err(invalid("compacted history exceeds its safe budget"));
    }
    let installed = sess
        .install_local_compaction(
            &source,
            revision,
            generation,
            replacement,
            // The fresh instructions are the baseline later steps diff against.
            Some(context.turn.to_turn_context_item()),
            Some(world_state),
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

/// One line standing in for a run of folded tool records; originals stay in recall. It belongs
/// to the window its records belonged to.
fn tool_run_record(items: &[ResponseItemEnvelope], run: &ToolRun) -> ResponseItemEnvelope {
    let source = LocalCompactionSource {
        first_item_id: run.first_item_id.clone(),
        last_item_id: run.last_item_id.clone(),
        kind: LocalCompactionKind::ToolRun,
    };
    let mut record =
        ResponseItemEnvelope::new(ContextualUserFragment::into(LocalCompactionFragment {
            text: run.text.clone(),
            source: source.clone(),
        }));
    record.metadata.get_or_insert_default().local_compaction = Some(source);
    record.metadata.get_or_insert_default().previous_window = items[run.indices[0]]
        .metadata
        .as_ref()
        .is_some_and(|metadata| metadata.previous_window);
    if let Some(turn_id) = items[run.indices[0]].item.turn_id() {
        record.item.set_turn_id_if_missing(turn_id);
    }
    record
}
