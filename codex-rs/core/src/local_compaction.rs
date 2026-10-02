//! Runtime adapter for the local compaction policy. Analysis is private; installation is atomic.

mod request;

use std::collections::BTreeSet;
use std::sync::Arc;

use codex_analytics::CompactionPhase;
use codex_analytics::CompactionReason;
use codex_context_compaction::Budget;
use codex_context_compaction::StagedDecisions;
use codex_context_compaction::TierPlan;
use codex_context_compaction::eligible_results;
use codex_history::ResponseItemEnvelope;
use codex_protocol::error::CodexErr;
use codex_protocol::error::Result as CodexResult;
use codex_protocol::user_input::UserInput;
use codex_utils_output_truncation::approx_token_count;

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

#[derive(Clone, Debug, Default)]
pub(crate) struct LocalCompactionState {
    pub(crate) staged: Option<StagedDecisions>,
    pub(crate) user_message_revision: u64,
    processed_ids: BTreeSet<String>,
    last_attempt_tokens: usize,
}

pub(crate) async fn maybe_clean_history(
    sess: &Arc<Session>,
    step: &Arc<StepContext>,
) -> CodexResult<bool> {
    let context = LocalCompactionContext::from_step(Arc::clone(step));
    let history = sess.clone_history().await;
    let tokens = history_tokens(history.annotated_items());
    let budget = budget(sess, &context).await;
    let mandatory = mandatory_limit(&context, budget);
    if !budget.should_analyze(tokens) && tokens.saturating_add(budget.fixed_tokens) < mandatory {
        return Ok(false);
    }
    let state = sess.get_local_compaction_state().await;
    let revision = history
        .conversation_history_snapshot()
        .user_message_revision();
    let growth = budget
        .window_tokens
        .saturating_mul(budget.minimum_savings_percent)
        / 100;
    if state.user_message_revision == revision
        && state.last_attempt_tokens > 0
        && tokens.saturating_add(budget.fixed_tokens) < mandatory
        && tokens < state.last_attempt_tokens.saturating_add(growth.max(1))
    {
        return Ok(false);
    }
    let version = history.history_version();
    crate::compact::run_inline_auto_compact_task(
        Arc::clone(sess),
        Arc::clone(step),
        InitialContextInjection::DoNotInject,
        CompactionReason::ContextLimit,
        CompactionPhase::MidTurn,
    )
    .await?;
    Ok(sess.clone_history().await.history_version() != version)
}

pub(crate) async fn run_pipeline(
    sess: &Arc<Session>,
    context: &LocalCompactionContext,
    input: Vec<UserInput>,
    injection: InitialContextInjection,
    metadata: CompactionTurnMetadata,
) -> CodexResult<bool> {
    let history = sess.clone_history().await;
    let source = history.annotated_items().to_vec();
    let revision = history
        .conversation_history_snapshot()
        .user_message_revision();
    let before_tokens = history_tokens(&source);
    let budget = budget(sess, context).await;
    let mandatory = mandatory_limit(context, budget);
    let mut state = sess.get_local_compaction_state().await;
    if state.user_message_revision != revision {
        state = LocalCompactionState {
            user_message_revision: revision,
            ..Default::default()
        };
    }
    let supplemental = input
        .into_iter()
        .filter_map(|item| match item {
            UserInput::Text { text, .. } => Some(text),
            _ => None,
        })
        .collect::<Vec<_>>()
        .join("\n");
    let responses_metadata = sess
        .responses_metadata(
            &context.turn,
            CodexResponsesRequestKind::Compaction(metadata),
        )
        .await;
    let mut client = sess.services.model_client.new_session();
    let mut response_id = None;
    let mut replacement = source.clone();
    if let Some(staged) = &state.staged {
        if staged.is_current(&source) {
            replacement = staged.apply(&source).map_err(invalid)?;
        } else {
            state.staged = None;
        }
    }
    let eligible = eligible_results(&source);
    let mut candidates: Vec<_> = eligible
        .iter()
        .filter(|id| !state.processed_ids.contains(*id))
        .cloned()
        .collect();
    // The automatic entrypoint already enforces meaningful growth before a retry.
    // Include prior keeps even when newly completed outputs also need classification.
    if before_tokens > budget.history_target() {
        if let Some(staged) = &state.staged {
            candidates.extend(staged.kept_ids());
        } else if candidates.is_empty() {
            candidates = eligible;
        }
    }
    for batch in candidates.chunks(64) {
        let request = request::classifier(batch, &supplemental)?;
        let (json, id) = request::infer_json(
            sess,
            context,
            &mut client,
            &responses_metadata,
            &source,
            request,
        )
        .await?;
        let staged =
            StagedDecisions::parse_candidates(source.clone(), batch, &json).map_err(invalid)?;
        if let Some(previous) = &mut state.staged {
            previous.merge(staged).map_err(invalid)?;
        } else {
            state.staged = Some(staged);
        }
        if let Some(staged) = &state.staged {
            replacement = staged.apply(&source).map_err(invalid)?;
        }
        state.processed_ids.extend(batch.iter().cloned());
        state.last_attempt_tokens = before_tokens;
        // Speculative decisions have no corresponding history/rollout event.
        sess.set_local_compaction_state(state.clone()).await;
        response_id = Some(id);
        if history_tokens(&replacement) <= budget.history_target() {
            break;
        }
    }
    let cleaned_tokens = history_tokens(&replacement);
    if cleaned_tokens > budget.history_target() {
        let costs: Vec<_> = replacement
            .iter()
            .map(|item| item_tokens(&item.item))
            .collect();
        if let Some(plan) = TierPlan::new(&replacement, budget.history_target(), &costs) {
            if plan.max_fragment_bytes == 0 {
                state.last_attempt_tokens = before_tokens;
                sess.set_local_compaction_state(state).await;
                return insufficient_room(budget, before_tokens, mandatory);
            }
            let request = request::summarizer(&plan, &supplemental)?;
            let (json, id) = request::infer_json(
                sess,
                context,
                &mut client,
                &responses_metadata,
                &replacement,
                request,
            )
            .await?;
            let fragments = plan.parse(&json).map_err(invalid)?;
            let mut tiered = plan.retained_prefix;
            for fragment in fragments {
                let mut envelope = ResponseItemEnvelope::new(ContextualUserFragment::into(
                    LocalCompactionFragment {
                        text: fragment.text,
                        source: fragment.source.clone(),
                    },
                ));
                envelope.metadata.get_or_insert_default().local_compaction = Some(fragment.source);
                envelope.item.set_turn_id_if_missing(&context.turn.sub_id);
                tiered.push(envelope);
            }
            tiered.extend(plan.retained_tail);
            replacement = tiered;
            response_id = Some(id);
        }
    }
    let after_tokens = history_tokens(&replacement);
    state.last_attempt_tokens = before_tokens;
    if after_tokens > budget.history_target() {
        // Protected work is indivisible. Never truncate it or install a partial tier response.
        sess.set_local_compaction_state(state).await;
        return insufficient_room(budget, before_tokens, mandatory);
    }
    if !budget.useful(before_tokens, after_tokens) {
        sess.set_local_compaction_state(state).await;
        return insufficient_room(budget, before_tokens, mandatory);
    }
    let baseline = match injection {
        InitialContextInjection::BeforeLastUserMessage { world_state, .. } => Some(world_state),
        InitialContextInjection::DoNotInject => None,
    };
    let surviving_ids: BTreeSet<_> = replacement
        .iter()
        .filter_map(|item| item.item.id().map(ToString::to_string))
        .collect();
    state.processed_ids.retain(|id| surviving_ids.contains(id));
    let installed = sess
        .install_local_compaction(
            &source,
            revision,
            replacement,
            sess.reference_context_item().await,
            baseline,
            CompactedHistoryMetadata {
                message:
                    "Local context cleanup; exact originals are available through recall_read_item."
                        .to_string(),
                // The CAS installer allocates real window IDs only after validating the source.
                window_number: 0,
                window_ids: crate::state::AutoCompactWindowIds::new_initial(),
                compaction_response_id: response_id,
                compaction_model_hash: context.settings.model_info.comp_hash.clone(),
            },
            budget.history_target(),
        )
        .await?;
    if installed {
        state.staged = None;
        state.last_attempt_tokens = after_tokens;
        state.user_message_revision = sess
            .conversation_history_snapshot()
            .await
            .user_message_revision();
        sess.set_local_compaction_state(state).await;
        sess.recompute_token_usage(&context.turn).await;
    }
    Ok(installed)
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
        trigger_percent: usize::from(config.trigger_percent),
        target_percent: usize::from(config.target_percent),
        minimum_savings_percent: usize::from(config.minimum_savings_percent),
    }
}

fn item_tokens(item: &codex_protocol::models::ResponseItem) -> usize {
    usize::try_from(estimate_item_token_count(item)).unwrap_or(0)
}

fn mandatory_limit(context: &LocalCompactionContext, budget: Budget) -> usize {
    context
        .settings
        .model_info
        .auto_compact_token_limit()
        .and_then(|tokens| usize::try_from(tokens).ok())
        .unwrap_or(budget.window_tokens)
        .min(budget.window_tokens)
}

fn insufficient_room(
    budget: Budget,
    history_tokens: usize,
    mandatory_tokens: usize,
) -> CodexResult<bool> {
    if history_tokens.saturating_add(budget.fixed_tokens) >= mandatory_tokens {
        Err(invalid(
            "local compaction cannot reclaim enough room at the mandatory input limit",
        ))
    } else {
        Ok(false)
    }
}

fn history_tokens(items: &[ResponseItemEnvelope]) -> usize {
    items.iter().map(|item| item_tokens(&item.item)).sum()
}

fn invalid(error: impl std::fmt::Display) -> CodexErr {
    CodexErr::InvalidRequest(format!("Local compaction failed: {error}"))
}

#[cfg(test)]
#[path = "local_compaction_tests.rs"]
mod tests;
