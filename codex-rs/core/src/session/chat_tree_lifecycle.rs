use std::sync::Arc;
use std::time::Instant;

use crate::chat_tree::ChatTreeError;
use crate::client_common::Prompt;
use crate::client_common::ResponseEvent;
use crate::responses_metadata::CodexResponsesRequestKind;
use crate::session::session::Session;
use crate::session::turn_context::TurnContext;
use codex_features::Feature;
use codex_protocol::config_types::ReasoningSummary as ReasoningSummaryConfig;
use codex_protocol::models::BaseInstructions;
use codex_protocol::models::ContentItem;
use codex_protocol::models::ResponseItem;
use codex_protocol::protocol::ChatTreeNodeStatus;
use codex_protocol::protocol::Event;
use codex_protocol::protocol::EventMsg;
use codex_protocol::protocol::RolloutItem;
use codex_protocol::protocol::TurnAbortReason;
use codex_protocol::user_input::UserInput;
use codex_rollout_trace::InferenceTraceContext;
use futures::StreamExt;
use tokio_util::sync::CancellationToken;
use tracing::trace;
use tracing::warn;

const CHAT_TREE_SUMMARY_SYSTEM_INSTRUCTIONS: &str =
    "You generate a concise summary label for one completed assistant turn.";

pub(crate) struct ChatTreeSummaryJob {
    last_user_message: Option<String>,
    last_agent_message: Option<String>,
    cancellation_token: CancellationToken,
}

fn summarize_for_chat_tree(message: Option<&str>, fallback: &str) -> String {
    let trimmed = message
        .map(str::trim)
        .filter(|text| !text.is_empty())
        .unwrap_or(fallback);
    let line = trimmed
        .lines()
        .find(|line| !line.trim().is_empty())
        .map(str::trim)
        .unwrap_or(fallback);
    let line = line
        .strip_prefix('"')
        .and_then(|line| line.strip_suffix('"'))
        .unwrap_or(line)
        .trim();
    if line.chars().count() > 96 {
        let mut summary = line.chars().take(93).collect::<String>();
        summary.push_str("...");
        summary
    } else {
        line.to_string()
    }
}

fn chat_tree_summary_request(user_message: &str, assistant_message: &str) -> String {
    format!(
        "Summarize this turn for a chat tree node.\n\
         Requirements:\n\
         - single line\n\
         - at most 96 characters\n\
         - no markdown\n\
         - no surrounding quotes\n\
         - describe user intent + assistant outcome\n\n\
         User message:\n\
         {user_message}\n\n\
         Assistant message:\n\
         {assistant_message}"
    )
}

fn abort_reason_summary(reason: &TurnAbortReason) -> &'static str {
    match reason {
        TurnAbortReason::Interrupted | TurnAbortReason::BudgetLimited => "turn interrupted",
        TurnAbortReason::Replaced => "turn replaced",
        TurnAbortReason::ReviewEnded => "turn review ended",
    }
}

fn chat_tree_status_from_abort_reason(reason: &TurnAbortReason) -> ChatTreeNodeStatus {
    match reason {
        TurnAbortReason::Interrupted | TurnAbortReason::BudgetLimited => {
            ChatTreeNodeStatus::Interrupted
        }
        TurnAbortReason::Replaced => ChatTreeNodeStatus::Replaced,
        TurnAbortReason::ReviewEnded => ChatTreeNodeStatus::ReviewEnded,
    }
}

fn is_root_chat_tree_turn(turn_context: &TurnContext) -> bool {
    !turn_context.session_source.is_non_root_agent()
}

impl Session {
    pub(crate) async fn persist_rollout_items_durable(
        &self,
        items: &[RolloutItem],
    ) -> std::io::Result<()> {
        if let Some(live_thread) = self.live_thread() {
            live_thread
                .append_items(items)
                .await
                .map_err(std::io::Error::other)?;
            live_thread.flush().await.map_err(std::io::Error::other)?;
        }
        Ok(())
    }

    pub(crate) async fn start_chat_tree_node(
        &self,
        turn_context: &TurnContext,
    ) -> Result<(), ChatTreeError> {
        let Some(started) = ({
            let mut state = self.state.lock().await;
            let parent_history = state.clone_history();
            state
                .chat_tree
                .start_node(turn_context.sub_id.clone(), parent_history)
        }) else {
            return Ok(());
        };
        let msg = EventMsg::ChatTreeNodeStarted(Box::new(started.event.clone().into()));
        if let Err(err) = self
            .persist_rollout_items_durable(&[RolloutItem::EventMsg(msg.clone())])
            .await
        {
            let mut state = self.state.lock().await;
            state.chat_tree.rollback_node_start(&started);
            return Err(ChatTreeError::Persistence(err.to_string()));
        }
        self.services
            .rollout_thread_trace
            .record_codex_turn_event(&turn_context.sub_id, &msg);
        self.services
            .rollout_thread_trace
            .record_tool_call_event(turn_context.sub_id.clone(), &msg);
        self.services
            .rollout_thread_trace
            .record_protocol_event(&msg);
        self.deliver_event_raw(Event {
            id: turn_context.sub_id.clone(),
            msg,
        })
        .await;
        Ok(())
    }

    pub(crate) async fn finalize_chat_tree_node(
        &self,
        turn_context: &TurnContext,
        status: ChatTreeNodeStatus,
    ) -> bool {
        let Some(finalization) = ({
            let mut state = self.state.lock().await;
            let history_snapshot = state.clone_history();
            state
                .chat_tree
                .finalize_node(&turn_context.sub_id, status, history_snapshot)
        }) else {
            return false;
        };
        let msg = EventMsg::ChatTreeNodeFinalized(Box::new(finalization.event.clone().into()));
        if let Err(err) = self
            .persist_rollout_items_durable(&[RolloutItem::EventMsg(msg.clone())])
            .await
        {
            let mut state = self.state.lock().await;
            state.chat_tree.rollback_node_finalization(&finalization);
            warn!(
                "failed to persist chat tree node finalization for turn {}: {err}",
                turn_context.sub_id
            );
            return false;
        }
        self.services
            .rollout_thread_trace
            .record_codex_turn_event(&turn_context.sub_id, &msg);
        self.services
            .rollout_thread_trace
            .record_tool_call_event(turn_context.sub_id.clone(), &msg);
        self.services
            .rollout_thread_trace
            .record_protocol_event(&msg);
        self.deliver_event_raw(Event {
            id: turn_context.sub_id.clone(),
            msg,
        })
        .await;
        true
    }

    pub(crate) async fn update_chat_tree_node_summary(
        &self,
        turn_context: &TurnContext,
        node_id: &str,
        summary: String,
    ) {
        let _ = self
            .update_chat_tree_node_summary_inner(turn_context, node_id, summary, None)
            .await;
    }

    pub(crate) async fn update_chat_tree_node_summary_unless_cancelled(
        &self,
        turn_context: &TurnContext,
        node_id: &str,
        summary: String,
        cancellation_token: &CancellationToken,
    ) -> bool {
        self.update_chat_tree_node_summary_inner(
            turn_context,
            node_id,
            summary,
            Some(cancellation_token),
        )
        .await
    }

    async fn update_chat_tree_node_summary_inner(
        &self,
        turn_context: &TurnContext,
        node_id: &str,
        summary: String,
        cancellation_token: Option<&CancellationToken>,
    ) -> bool {
        let Some(update) = ({
            let mut state = if let Some(cancellation_token) = cancellation_token {
                tokio::select! {
                    _ = cancellation_token.cancelled() => return false,
                    state = self.state.lock() => state,
                }
            } else {
                self.state.lock().await
            };
            if cancellation_token.is_some_and(CancellationToken::is_cancelled) {
                return false;
            }
            state
                .chat_tree
                .update_node_summary(node_id, Some(summary.clone()))
        }) else {
            return false;
        };
        let msg = EventMsg::ChatTreeNodeSummaryUpdated(Box::new(update.event.clone().into()));
        if let Err(err) = self
            .persist_rollout_items_durable(&[RolloutItem::EventMsg(msg.clone())])
            .await
        {
            let mut state = self.state.lock().await;
            state.chat_tree.rollback_node_summary_update(&update);
            warn!("failed to persist chat tree node summary for turn {node_id}: {err}");
            return false;
        }
        self.services
            .rollout_thread_trace
            .record_codex_turn_event(node_id, &msg);
        self.services
            .rollout_thread_trace
            .record_tool_call_event(node_id.to_string(), &msg);
        self.services
            .rollout_thread_trace
            .record_protocol_event(&msg);
        self.deliver_event_raw(Event {
            id: turn_context.sub_id.clone(),
            msg,
        })
        .await;
        true
    }

    pub async fn set_current_chat_tree_node(
        &self,
        node_id: &str,
        expected_revision: Option<u64>,
    ) -> Result<(), ChatTreeError> {
        let selection = {
            let mut state = self.state.lock().await;
            state
                .chat_tree
                .set_current_node(node_id, expected_revision)?
        };

        let event = Event {
            id: node_id.to_string(),
            msg: EventMsg::ChatTreeCurrentNodeChanged(Box::new(selection.event.clone().into())),
        };
        if let Err(err) = self
            .persist_rollout_items_durable(&[RolloutItem::EventMsg(event.msg.clone())])
            .await
        {
            let mut state = self.state.lock().await;
            state.chat_tree.rollback_current_node_change(&selection);
            return Err(ChatTreeError::Persistence(err.to_string()));
        }

        self.replace_history(
            selection.history.raw_items().to_vec(),
            selection.history.reference_context_item(),
        )
        .await;
        self.services
            .rollout_thread_trace
            .record_protocol_event(&event.msg);
        self.deliver_event_raw(event).await;
        Ok(())
    }

    pub(crate) async fn chat_tree_projection(
        &self,
    ) -> crate::chat_tree::ChatTreeProjectionSnapshot {
        let state = self.state.lock().await;
        state.chat_tree.projection()
    }

    pub(crate) async fn start_chat_tree_node_for_turn(
        &self,
        turn_context: &TurnContext,
        input: &[UserInput],
    ) {
        if !is_root_chat_tree_turn(turn_context) {
            return;
        }
        turn_context.capture_chat_tree_summary_user_message(input);
        if let Err(err) = self.start_chat_tree_node(turn_context).await {
            warn!(
                turn_id = turn_context.sub_id.as_str(),
                error = ?err,
                "failed to start chat tree node"
            );
        }
    }

    pub(crate) async fn complete_chat_tree_node_before_turn_complete(
        &self,
        turn_context: &TurnContext,
        last_agent_message: Option<String>,
    ) -> Option<ChatTreeSummaryJob> {
        if !is_root_chat_tree_turn(turn_context) {
            return None;
        }
        let last_agent_message_for_summary =
            last_agent_message.filter(|message| !message.trim().is_empty());
        let finalized_chat_tree_node = self
            .finalize_chat_tree_node(turn_context, ChatTreeNodeStatus::Completed)
            .await;
        if !turn_context
            .config
            .features
            .enabled(Feature::ChatTreeSummary)
        {
            return None;
        }
        if last_agent_message_for_summary.is_none() || !finalized_chat_tree_node {
            return None;
        }
        Some(ChatTreeSummaryJob {
            last_user_message: turn_context.chat_tree_summary_user_message(),
            last_agent_message: last_agent_message_for_summary,
            cancellation_token: self
                .register_chat_tree_summary_job(&turn_context.sub_id)
                .await,
        })
    }

    pub(crate) async fn spawn_chat_tree_summary_after_turn_complete(
        self: &Arc<Self>,
        turn_context: Arc<TurnContext>,
        summary_job: Option<ChatTreeSummaryJob>,
    ) {
        let Some(summary_job) = summary_job else {
            return;
        };
        if summary_job.cancellation_token.is_cancelled() {
            self.finish_chat_tree_summary_job(&turn_context.sub_id)
                .await;
            return;
        }
        self.spawn_chat_tree_summary_job(
            turn_context,
            summary_job.last_user_message,
            summary_job.last_agent_message,
            summary_job.cancellation_token,
        );
    }

    pub(crate) async fn abort_chat_tree_node(
        &self,
        turn_context: &TurnContext,
        reason: &TurnAbortReason,
    ) {
        if !is_root_chat_tree_turn(turn_context) {
            return;
        }
        let chat_tree_status = chat_tree_status_from_abort_reason(reason);
        let chat_tree_summary = summarize_for_chat_tree(None, abort_reason_summary(reason));
        if self
            .finalize_chat_tree_node(turn_context, chat_tree_status)
            .await
        {
            self.update_chat_tree_node_summary(
                turn_context,
                &turn_context.sub_id,
                chat_tree_summary,
            )
            .await;
        }
    }

    async fn register_chat_tree_summary_job(&self, node_id: &str) -> CancellationToken {
        let token = CancellationToken::new();
        let mut jobs = self.chat_tree_summary_jobs.lock().await;
        if let Some(previous_token) = jobs.insert(node_id.to_string(), token.clone()) {
            previous_token.cancel();
        }
        token
    }

    #[cfg(test)]
    pub(crate) async fn register_chat_tree_summary_job_for_test(
        &self,
        node_id: &str,
    ) -> CancellationToken {
        self.register_chat_tree_summary_job(node_id).await
    }

    async fn finish_chat_tree_summary_job(&self, node_id: &str) {
        let mut jobs = self.chat_tree_summary_jobs.lock().await;
        jobs.remove(node_id);
        self.chat_tree_summary_jobs_changed.notify_waiters();
    }

    #[cfg(test)]
    pub(crate) async fn finish_chat_tree_summary_job_for_test(&self, node_id: &str) {
        self.finish_chat_tree_summary_job(node_id).await;
    }

    pub(crate) async fn cancel_all_chat_tree_summary_jobs(&self) {
        loop {
            let notified = self.chat_tree_summary_jobs_changed.notified();
            let tokens = {
                let jobs = self.chat_tree_summary_jobs.lock().await;
                if jobs.is_empty() {
                    return;
                }
                jobs.values().cloned().collect::<Vec<_>>()
            };
            for token in tokens {
                token.cancel();
            }
            notified.await;
        }
    }

    fn spawn_chat_tree_summary_job(
        self: &Arc<Self>,
        turn_context: Arc<TurnContext>,
        last_user_message: Option<String>,
        last_agent_message: Option<String>,
        cancellation_token: CancellationToken,
    ) {
        let sess = Arc::clone(self);
        tokio::spawn(async move {
            sess.run_async_chat_tree_summary_job(
                turn_context,
                last_user_message,
                last_agent_message,
                cancellation_token,
            )
            .await;
        });
    }

    async fn run_async_chat_tree_summary_job(
        self: Arc<Self>,
        turn_context: Arc<TurnContext>,
        last_user_message: Option<String>,
        last_agent_message: Option<String>,
        cancellation_token: CancellationToken,
    ) {
        let node_id = turn_context.sub_id.clone();
        async {
            let user_message = last_user_message.as_deref().unwrap_or("(none)");
            let assistant_message = last_agent_message.as_deref().unwrap_or("(none)");
            let request_payload = chat_tree_summary_request(user_message, assistant_message);
            let prompt = Prompt {
                input: vec![ResponseItem::Message {
                    id: None,
                    role: "user".to_string(),
                    content: vec![ContentItem::InputText {
                        text: request_payload,
                    }],
                    phase: None,
                    internal_chat_message_metadata_passthrough: None,
                }],
                tools: Vec::new(),
                parallel_tool_calls: false,
                base_instructions: BaseInstructions {
                    text: CHAT_TREE_SUMMARY_SYSTEM_INSTRUCTIONS.to_string(),
                },
                output_schema: None,
                output_schema_strict: true,
            };
            let mut client_session = self.services.model_client.new_session();
            let responses_metadata = turn_context.turn_metadata_state.to_responses_metadata(
                self.installation_id.clone(),
                node_id.to_string(),
                CodexResponsesRequestKind::ChatTreeSummary,
            );
            let inference_trace_context = InferenceTraceContext::disabled();
            let started_at = Instant::now();
            let mut stream = match tokio::select! {
                _ = cancellation_token.cancelled() => {
                    trace!(
                        node_id = node_id.as_str(),
                        "chat tree summary request cancelled before start"
                    );
                    return;
                }
                stream = client_session.stream(
                    &prompt,
                    &turn_context.model_info,
                    &turn_context.session_telemetry,
                    None,
                    ReasoningSummaryConfig::None,
                    turn_context.config.service_tier.clone(),
                    &responses_metadata,
                    &inference_trace_context,
                ) => stream,
            } {
                Ok(stream) => stream,
                Err(err) => {
                    warn!(
                        node_id = node_id.as_str(),
                        error = %err,
                        "chat tree summary request failed to start"
                    );
                    return;
                }
            };
            let mut response_from_item: Option<String> = None;
            let mut response_from_deltas = String::new();
            let mut saw_completed = false;
            loop {
                let Some(event_result) = (tokio::select! {
                    _ = cancellation_token.cancelled() => {
                        trace!(
                            node_id = node_id.as_str(),
                            "chat tree summary request cancelled during stream"
                        );
                        return;
                    }
                    event_result = stream.next() => event_result,
                }) else {
                    break;
                };
                let event = match event_result {
                    Ok(event) => event,
                    Err(err) => {
                        warn!(
                            node_id = node_id.as_str(),
                            error = %err,
                            "chat tree summary stream failed"
                        );
                        return;
                    }
                };
                match event {
                    ResponseEvent::OutputTextDelta(delta) => response_from_deltas.push_str(&delta),
                    ResponseEvent::OutputItemDone(item) => {
                        if let ResponseItem::Message { role, content, .. } = item
                            && role == "assistant"
                        {
                            response_from_item = crate::compact::content_items_to_text(&content);
                        }
                    }
                    ResponseEvent::Completed { .. } => {
                        saw_completed = true;
                        break;
                    }
                    _ => {}
                }
            }
            if !saw_completed || cancellation_token.is_cancelled() {
                return;
            }
            let response_payload = response_from_item.or_else(|| {
                let trimmed = response_from_deltas.trim();
                (!trimmed.is_empty()).then(|| trimmed.to_string())
            });
            let Some(response_payload) = response_payload else {
                warn!(
                    node_id = node_id.as_str(),
                    "chat tree summary response had no assistant text"
                );
                return;
            };
            let summary = summarize_for_chat_tree(Some(response_payload.as_str()), "");
            if summary.is_empty() {
                warn!(
                    node_id = node_id.as_str(),
                    "chat tree summary response normalized to empty summary"
                );
                return;
            }
            if cancellation_token.is_cancelled() {
                return;
            }
            let updated = self
                .update_chat_tree_node_summary_unless_cancelled(
                    turn_context.as_ref(),
                    &node_id,
                    summary,
                    &cancellation_token,
                )
                .await;
            if !updated {
                return;
            }
            trace!(
                node_id = node_id.as_str(),
                elapsed_ms = started_at.elapsed().as_millis(),
                "chat tree summary updated"
            );
        }
        .await;
        self.finish_chat_tree_summary_job(&node_id).await;
    }
}

#[cfg(test)]
mod tests {
    use super::chat_tree_summary_request;
    use super::summarize_for_chat_tree;
    use pretty_assertions::assert_eq;

    #[test]
    fn summarize_for_chat_tree_truncates_to_96_chars_with_ellipsis() {
        let summary = summarize_for_chat_tree(
            Some(
                "abcdefghijklmnopqrstuvwxyzabcdefghijklmnopqrstuvwxyzabcdefghijklmnopqrstuvwxyzabcdefghijklmnopqrst",
            ),
            "fallback",
        );

        assert_eq!(summary.chars().count(), 96);
        assert!(summary.ends_with("..."));
        assert_eq!(
            summary,
            "abcdefghijklmnopqrstuvwxyzabcdefghijklmnopqrstuvwxyzabcdefghijklmnopqrstuvwxyzabcdefghijklmno..."
        );
    }

    #[test]
    fn summarize_for_chat_tree_keeps_exactly_96_chars_without_ellipsis() {
        let input = "abcdefghijklmnopqrstuvwxyzabcdefghijklmnopqrstuvwxyzabcdefghijklmnopqrstuvwxyzabcdefghijklmnopqr";
        let summary = summarize_for_chat_tree(Some(input), "fallback");

        assert_eq!(summary.chars().count(), 96);
        assert_eq!(summary, input);
    }

    #[test]
    fn chat_tree_summary_request_preserves_placeholder_like_user_text() {
        let request = chat_tree_summary_request(
            "literal user text with {assistant_message}",
            "assistant outcome",
        );

        assert!(request.contains(
            "User message:\nliteral user text with {assistant_message}\n\nAssistant message:\nassistant outcome"
        ));
    }
}
