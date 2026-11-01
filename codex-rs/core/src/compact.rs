use std::collections::HashSet;
use std::sync::Arc;
use std::time::Instant;

use crate::context::CompactionSummary;
use crate::context::ContextualUserFragment;
use crate::context::world_state::WorldState;
use crate::context::world_state::WorldStateSnapshot;
use crate::context::world_state::split_prefix_updates;
use crate::context_manager::updates::merge_world_state_updates;
use crate::hook_runtime::PostCompactHookOutcome;
use crate::hook_runtime::PreCompactHookOutcome;
use crate::hook_runtime::run_post_compact_hooks;
use crate::hook_runtime::run_pre_compact_hooks;
use crate::responses_metadata::CompactionTurnMetadata;
use crate::session::session::Session;
use crate::session::step_context::StepContext;
use crate::session::turn_context::TurnContext;
use crate::state::AutoCompactWindowIds;
use codex_analytics::CodexCompactionEvent;
use codex_analytics::CompactionImplementation;
use codex_analytics::CompactionPhase;
use codex_analytics::CompactionReason;
use codex_analytics::CompactionStatus;
use codex_analytics::CompactionStrategy;
use codex_analytics::CompactionTrigger;
use codex_analytics::now_unix_seconds;
use codex_context_fragments::AnnotatedContent;
use codex_context_fragments::set_annotated_content;
use codex_context_fragments::truncate_annotated_text;
use codex_history::CodexHarnessMetadata;
use codex_history::ResponseItemEnvelope;
use codex_protocol::ResponseItemId;
use codex_protocol::error::CodexErr;
use codex_protocol::error::CodexErrorDetails;
use codex_protocol::error::Result as CodexResult;
use codex_protocol::items::ContextCompactionItem;
use codex_protocol::items::TurnItem;
use codex_protocol::models::AgentMessageInputContent;
use codex_protocol::models::ContentItem;
use codex_protocol::models::ContentItemKind;
use codex_protocol::models::ResponseItem;
use codex_protocol::protocol::EventMsg;
use codex_utils_output_truncation::approx_token_count;

pub use codex_prompts::SUMMARY_PREFIX;
const COMPACT_USER_MESSAGE_MAX_TOKENS: usize = 20_000;

/// Thread-local policy that preserves inherited context without changing model or skill budgets.
/// Insert into `StartThreadOptions::thread_extension_init` to disable automatic compaction.
pub struct AutoCompactionDisabled;

/// Metadata for a new compaction checkpoint, kept separate from its replacement history.
///
/// `Session::replace_compacted_history` assigns missing item IDs before constructing the persisted
/// `CompactedItem`, ensuring the live and persisted histories remain identical.
pub(crate) struct CompactedHistoryMetadata {
    pub(crate) input_goal_ids: HashSet<ResponseItemId>,
    pub(crate) message: String,
    pub(crate) window_number: u64,
    pub(crate) window_ids: AutoCompactWindowIds,
    pub(crate) compaction_response_id: Option<String>,
    pub(crate) compaction_model_hash: Option<String>,
    pub(crate) reviewer_compaction_hash: Option<String>,
}

/// Renders the window prefix and ordinary context with their comparison baseline.
pub(crate) async fn build_compaction_replacement_history(
    sess: &Session,
    step_context: &StepContext,
    world_state: &WorldState,
    compacted_history: Vec<ResponseItemEnvelope>,
) -> (Vec<ResponseItemEnvelope>, WorldStateSnapshot) {
    let (updates, snapshot) = sess
        .build_initial_context_with_world_state(step_context, world_state)
        .await;
    let (prefix, context) = split_prefix_updates(updates);
    let context = merge_world_state_updates(context);
    (
        assemble_compaction_history(compacted_history, prefix, context),
        snapshot,
    )
}

fn assemble_compaction_history(
    compacted_history: Vec<ResponseItemEnvelope>,
    prefix: Vec<ResponseItem>,
    context: Vec<ResponseItem>,
) -> Vec<ResponseItemEnvelope> {
    let history = insert_initial_context_before_last_real_user_or_summary(
        compacted_history,
        context.into_iter().map(ResponseItemEnvelope::new).collect(),
    );
    prefix
        .into_iter()
        .map(ResponseItemEnvelope::new)
        .chain(history)
        .collect()
}

/// Automatic compaction on the local route. Returns whether a new history was installed.
pub(crate) async fn run_inline_auto_compact_task(
    sess: Arc<Session>,
    step_context: Arc<StepContext>,
    replacement_step_context: Arc<StepContext>,
    world_state: Arc<WorldState>,
    reason: CompactionReason,
    phase: CompactionPhase,
) -> CodexResult<bool> {
    run_compact_task_inner(
        sess,
        step_context,
        replacement_step_context,
        world_state,
        CompactionTurnMetadata::new(
            CompactionTrigger::Auto,
            reason,
            CompactionImplementation::Responses,
            phase,
        ),
    )
    .await
}

pub(crate) async fn run_compact_task(
    sess: Arc<Session>,
    step_context: Arc<StepContext>,
    world_state: Arc<WorldState>,
) -> CodexResult<()> {
    run_compact_task_inner(
        sess.clone(),
        Arc::clone(&step_context),
        step_context,
        world_state,
        CompactionTurnMetadata::new(
            CompactionTrigger::Manual,
            CompactionReason::UserRequested,
            CompactionImplementation::Responses,
            CompactionPhase::StandaloneTurn,
        ),
    )
    .await?;
    Ok(())
}

async fn run_compact_task_inner(
    sess: Arc<Session>,
    step_context: Arc<StepContext>,
    replacement_step_context: Arc<StepContext>,
    world_state: Arc<WorldState>,
    compaction_metadata: CompactionTurnMetadata,
) -> CodexResult<bool> {
    let turn_context = Arc::clone(&step_context.turn);
    let trigger = compaction_metadata.trigger();
    let reason = compaction_metadata.reason();
    let phase = compaction_metadata.phase();
    let attempt = CompactionAnalyticsAttempt::begin(
        sess.as_ref(),
        turn_context.as_ref(),
        trigger,
        reason,
        CompactionImplementation::Responses,
        phase,
    )
    .await;
    let pre_compact_outcome = run_pre_compact_hooks(&sess, &turn_context, trigger).await;
    match pre_compact_outcome {
        PreCompactHookOutcome::Continue => {}
        PreCompactHookOutcome::Stopped => {
            let error = CodexErr::TurnAborted;
            attempt
                .track(
                    sess.as_ref(),
                    CompactionStatus::Interrupted,
                    Some(&error),
                    CompactionAnalyticsDetails::default(),
                )
                .await;
            return Err(error);
        }
    }
    let result = run_compact_task_inner_impl(
        Arc::clone(&sess),
        step_context,
        replacement_step_context,
        world_state,
        compaction_metadata,
    )
    .await;
    let status = compaction_status_from_result(&result);
    let codex_error = result.as_ref().err();
    if result.is_ok() {
        let post_compact_outcome = run_post_compact_hooks(&sess, &turn_context, trigger).await;
        if let PostCompactHookOutcome::Stopped = post_compact_outcome {
            attempt
                .track(
                    sess.as_ref(),
                    status,
                    codex_error,
                    CompactionAnalyticsDetails::default(),
                )
                .await;
            return Err(CodexErr::TurnAborted);
        }
    }
    attempt
        .track(
            sess.as_ref(),
            status,
            codex_error,
            CompactionAnalyticsDetails::default(),
        )
        .await;
    if let Err(err) = &result
        && !matches!(phase, CompactionPhase::PostTurn)
        && !matches!(
            err.details(),
            CodexErrorDetails::Interrupted | CodexErrorDetails::TurnAborted
        )
    {
        sess.track_turn_codex_error(turn_context.as_ref(), err);
        // Pre-turn failures are reported after preserving the incoming prompt.
        if !matches!(phase, CompactionPhase::PreTurn) {
            let event = EventMsg::Error(err.to_error_event(/*message_prefix*/ None));
            sess.send_event(&turn_context, event).await;
        }
    }
    result
}

/// Local full compaction: see `crate::local_compaction::run_pipeline`.
async fn run_compact_task_inner_impl(
    sess: Arc<Session>,
    step_context: Arc<StepContext>,
    replacement_step_context: Arc<StepContext>,
    world_state: Arc<WorldState>,
    compaction_metadata: CompactionTurnMetadata,
) -> CodexResult<bool> {
    let turn_context = Arc::clone(&step_context.turn);
    let compaction_item = TurnItem::ContextCompaction(ContextCompactionItem::new());
    sess.emit_turn_item_started(&turn_context, &compaction_item)
        .await;
    let installed = Box::pin(crate::local_compaction::run_pipeline(
        &sess,
        &step_context,
        &replacement_step_context,
        &world_state,
        compaction_metadata,
    ))
    .await?;
    sess.emit_turn_item_completed(&turn_context, compaction_item)
        .await;
    Ok(installed)
}

pub(crate) struct CompactionAnalyticsAttempt {
    thread_id: String,
    turn_id: String,
    trigger: CompactionTrigger,
    reason: CompactionReason,
    implementation: CompactionImplementation,
    phase: CompactionPhase,
    active_context_tokens_before: i64,
    started_at: u64,
    start_instant: Instant,
}

#[derive(Clone, Copy, Default)]
pub(crate) struct CompactionAnalyticsDetails {
    pub(crate) active_context_tokens_before: Option<i64>,
    pub(crate) retained_image_count: Option<usize>,
    pub(crate) compaction_summary_tokens: Option<i64>,
    pub(crate) cached_input_tokens: Option<i64>,
    pub(crate) cache_write_input_tokens: Option<i64>,
}

impl CompactionAnalyticsAttempt {
    pub(crate) async fn begin(
        sess: &Session,
        turn_context: &TurnContext,
        trigger: CompactionTrigger,
        reason: CompactionReason,
        implementation: CompactionImplementation,
        phase: CompactionPhase,
    ) -> Self {
        let active_context_tokens_before = sess.get_total_token_usage().await;
        Self {
            thread_id: sess.thread_id.to_string(),
            turn_id: turn_context.sub_id.clone(),
            trigger,
            reason,
            implementation,
            phase,
            active_context_tokens_before,
            started_at: now_unix_seconds(),
            start_instant: Instant::now(),
        }
    }

    pub(crate) async fn track(
        self,
        sess: &Session,
        status: CompactionStatus,
        codex_error: Option<&CodexErr>,
        details: CompactionAnalyticsDetails,
    ) {
        let CompactionAnalyticsDetails {
            active_context_tokens_before,
            retained_image_count,
            compaction_summary_tokens,
            cached_input_tokens,
            cache_write_input_tokens,
        } = details;
        let active_context_tokens_before =
            active_context_tokens_before.unwrap_or(self.active_context_tokens_before);
        let active_context_tokens_after = sess.get_total_token_usage().await;
        sess.services
            .analytics_events_client
            .track_compaction(CodexCompactionEvent {
                thread_id: self.thread_id,
                turn_id: self.turn_id,
                trigger: self.trigger,
                reason: self.reason,
                implementation: self.implementation,
                phase: self.phase,
                strategy: CompactionStrategy::Memento,
                status,
                codex_error_kind: codex_error.map(Into::into),
                codex_error_http_status_code: codex_error
                    .and_then(CodexErr::http_status_code_value),
                usage_limit_window_minutes: codex_error.and_then(|error| match error.details() {
                    CodexErrorDetails::UsageLimitReached(error) => error.limit_window_minutes,
                    _ => None,
                }),
                active_context_tokens_before,
                active_context_tokens_after,
                retained_image_count,
                compaction_summary_tokens,
                cached_input_tokens,
                cache_write_input_tokens,
                started_at: self.started_at,
                completed_at: now_unix_seconds(),
                duration_ms: Some(
                    u64::try_from(self.start_instant.elapsed().as_millis()).unwrap_or(u64::MAX),
                ),
            });
    }
}

pub(crate) fn compaction_status_from_result<T>(result: &CodexResult<T>) -> CompactionStatus {
    match result {
        Ok(_) => CompactionStatus::Completed,
        Err(err)
            if matches!(
                err.details(),
                CodexErrorDetails::Interrupted | CodexErrorDetails::TurnAborted
            ) =>
        {
            CompactionStatus::Interrupted
        }
        Err(_) => CompactionStatus::Failed,
    }
}

pub fn content_items_to_text(content: &[ContentItem]) -> Option<String> {
    let mut pieces = Vec::new();
    for item in content {
        match item {
            ContentItem::InputText { text } | ContentItem::OutputText { text, .. } => {
                if !text.is_empty() {
                    pieces.push(text.as_str());
                }
            }
            ContentItem::InputImage { .. } | ContentItem::InputAudio { .. } => {}
        }
    }
    if pieces.is_empty() {
        None
    } else {
        Some(pieces.join("\n"))
    }
}

#[derive(Clone, Debug, PartialEq)]
pub(crate) struct CompactedUserMessage<'a> {
    // Flattened text is only for budgeting and recognizing summaries. The
    // output is built from original parts, never reconstructed from this text.
    // Borrow from the history snapshot until selected output is materialized.
    message: String,
    original: &'a ResponseItem,
    harness_metadata: Option<&'a CodexHarnessMetadata>,
}

#[cfg(test)]
pub(crate) fn collect_user_messages(items: &[ResponseItem]) -> Vec<CompactedUserMessage<'_>> {
    items
        .iter()
        .filter_map(|item| compacted_user_message(item, /*harness_metadata*/ None))
        .collect()
}

pub(crate) fn collect_annotated_user_messages(
    items: &[ResponseItemEnvelope],
) -> Vec<CompactedUserMessage<'_>> {
    items
        .iter()
        .filter_map(|envelope| compacted_user_message(&envelope.item, envelope.metadata.as_ref()))
        .collect()
}

fn compacted_user_message<'a>(
    item: &'a ResponseItem,
    harness_metadata: Option<&'a CodexHarnessMetadata>,
) -> Option<CompactedUserMessage<'a>> {
    let Some(TurnItem::UserMessage(user)) = crate::event_mapping::parse_turn_item(item) else {
        return None;
    };
    let message = user.message();
    if is_summary_message(&message) {
        return None;
    }
    Some(CompactedUserMessage {
        message,
        original: item,
        harness_metadata,
    })
}

pub(crate) fn is_summary_message(message: &str) -> bool {
    message.starts_with(format!("{SUMMARY_PREFIX}\n").as_str())
}

pub(crate) fn is_real_user_or_agent_message(item: &ResponseItem) -> bool {
    if let ResponseItem::AgentMessage { content, .. } = item {
        return !matches!(
            content.first(),
            Some(AgentMessageInputContent::InputText { text })
                if text.starts_with("Message Type: FINAL_ANSWER\n")
        );
    }
    matches!(item, ResponseItem::Message { role, .. } if role == "user")
        && matches!(
            crate::event_mapping::parse_turn_item(item),
            Some(TurnItem::UserMessage(user)) if !is_summary_message(&user.message())
        )
}

/// Inserts canonical initial context into compacted replacement history at the
/// model-expected boundary.
///
/// Placement rules:
/// - Prefer immediately before the last real user, agent message, or retained standalone output.
/// - If no real user messages remain, insert before the compaction summary so
///   the summary stays last.
/// - If there are no user messages, insert before the last compaction item so
///   that item remains last (remote compaction may return only compaction items).
/// - If there are no user messages or compaction items, append the context.
pub(crate) fn insert_initial_context_before_last_real_user_or_summary(
    mut compacted_history: Vec<ResponseItemEnvelope>,
    initial_context: Vec<ResponseItemEnvelope>,
) -> Vec<ResponseItemEnvelope> {
    let mut last_user_or_summary_index = None;
    let mut last_real_user_index = None;
    for (i, item) in compacted_history.iter().enumerate().rev() {
        if crate::compact_retained_tool_outputs::is_retained_tool_output(item)
            || is_real_user_or_agent_message(&item.item)
        {
            last_real_user_index = Some(i);
            break;
        }
        // Compaction summaries are encoded as user messages, so track both:
        // the last real user message (preferred insertion point) and the last
        // user-message-like item (fallback summary insertion point).
        if matches!(
            crate::event_mapping::parse_turn_item(&item.item),
            Some(TurnItem::UserMessage(_))
        ) {
            last_user_or_summary_index.get_or_insert(i);
        }
    }
    let last_compaction_index = compacted_history
        .iter()
        .enumerate()
        .rev()
        .find_map(|(i, item)| {
            matches!(
                &item.item,
                ResponseItem::Compaction { .. } | ResponseItem::ContextCompaction { .. }
            )
            .then_some(i)
        });
    let insertion_index = last_real_user_index
        .or(last_user_or_summary_index)
        .or(last_compaction_index);

    // Re-inject canonical context from the current session since we stripped it
    // from the pre-compaction history. Prefer placing it before the last real
    // user message; if there is no real user message left, place it before the
    // summary or compaction item so the compaction item remains last.
    if let Some(insertion_index) = insertion_index {
        compacted_history.splice(insertion_index..insertion_index, initial_context);
    } else {
        compacted_history.extend(initial_context);
    }

    compacted_history
}

pub(crate) fn build_compacted_history(
    initial_context: Vec<ResponseItemEnvelope>,
    user_messages: &[CompactedUserMessage<'_>],
    summary_text: &str,
) -> Vec<ResponseItemEnvelope> {
    build_compacted_history_with_limit(
        initial_context,
        user_messages,
        summary_text,
        COMPACT_USER_MESSAGE_MAX_TOKENS,
    )
}

fn build_compacted_history_with_limit(
    mut history: Vec<ResponseItemEnvelope>,
    user_messages: &[CompactedUserMessage<'_>],
    summary_text: &str,
    max_tokens: usize,
) -> Vec<ResponseItemEnvelope> {
    let mut selected_messages = Vec::new();
    if max_tokens > 0 {
        let mut remaining = max_tokens;
        for message in user_messages.iter().rev() {
            if remaining == 0 {
                break;
            }
            let tokens = approx_token_count(&message.message);
            let ResponseItem::Message {
                id,
                content,
                internal_chat_message_metadata_passthrough,
                ..
            } = message.original
            else {
                continue;
            };
            let mut harness_metadata = message.harness_metadata.cloned();
            let preserve_content = tokens <= remaining
                && content
                    .iter()
                    .all(|part| matches!(part, ContentItem::InputText { .. }));
            let mut item = ResponseItem::Message {
                status: None,
                encrypted_content: None,
                id: id.clone(),
                role: "user".to_owned(),
                content: if preserve_content {
                    content.clone()
                } else {
                    Vec::new()
                },
                phase: None,
                internal_chat_message_metadata_passthrough:
                    internal_chat_message_metadata_passthrough.clone(),
            };
            if !preserve_content {
                // Select text and its annotations by the same source index.
                // Do not clone discarded images/audio or infer their text's source.
                let metadata = internal_chat_message_metadata_passthrough.as_ref();
                let mut parts: Vec<_> = crate::event_mapping::user_input_parts(content)
                    .filter_map(|(index, part)| {
                        let ContentItem::InputText { text } = part else {
                            return None;
                        };
                        let kind = metadata
                            .and_then(|metadata| metadata.content_item_kinds.as_ref())
                            .and_then(|kinds| kinds.get(index))
                            .cloned()
                            .unwrap_or_else(|| ContentItemKind("unknown".to_string()));
                        let source = metadata
                            .and_then(|metadata| metadata.content_item_metadata.as_ref())
                            .and_then(|sources| sources.get(index))
                            .cloned()
                            .unwrap_or_default();
                        Some(AnnotatedContent::text(text, kind, source))
                    })
                    .collect();
                if parts.is_empty() {
                    // Preserve the existing empty-text fallback for media-only input.
                    parts.push(AnnotatedContent::new(
                        ContentItem::InputText {
                            text: String::new(),
                        },
                        ContentItemKind("unknown".to_string()),
                    ));
                }
                set_annotated_content(&mut item, truncate_annotated_text(parts, remaining));
            }
            if tokens > remaining
                && let Some(metadata) = &mut harness_metadata
            {
                metadata.mark_retained_sources_incomplete();
            }
            selected_messages.push(ResponseItemEnvelope {
                item,
                metadata: harness_metadata,
            });
            if tokens > remaining {
                break;
            }
            remaining = remaining.saturating_sub(tokens);
        }
        selected_messages.reverse();
    }
    history.extend(selected_messages);

    let summary_text = if summary_text.is_empty() {
        "(no summary available)".to_string()
    } else {
        summary_text.to_string()
    };

    history.push(ResponseItemEnvelope::new(ContextualUserFragment::into(
        CompactionSummary::new(summary_text),
    )));

    history
}

#[cfg(test)]
#[path = "compact_tests.rs"]
mod tests;
