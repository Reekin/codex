mod compact;
mod ghost_snapshot;
mod regular;
mod review;
mod undo;
mod user_shell;

use std::sync::Arc;
use std::time::Duration;
use std::time::Instant;

use async_trait::async_trait;
use futures::StreamExt;
use tokio::select;
use tokio::sync::Notify;
use tokio_util::sync::CancellationToken;
use tokio_util::task::AbortOnDropHandle;
use tracing::Instrument;
use tracing::info;
use tracing::info_span;
use tracing::trace;
use tracing::warn;

use crate::AuthManager;
use crate::Prompt;
use crate::client_common::ResponseEvent;
use crate::codex::Session;
use crate::codex::TurnContext;
use crate::contextual_user_message::TURN_ABORTED_OPEN_TAG;
use crate::event_mapping::parse_turn_item;
use crate::models_manager::manager::ModelsManager;
use crate::protocol::ChatTreeNodeUpdatedEvent;
use crate::protocol::EventMsg;
use crate::protocol::TokenUsage;
use crate::protocol::TurnAbortReason;
use crate::protocol::TurnAbortedEvent;
use crate::protocol::TurnCompleteEvent;
use crate::state::ActiveTurn;
use crate::state::RunningTask;
use crate::state::TaskKind;
use codex_otel::metrics::names::TURN_E2E_DURATION_METRIC;
use codex_otel::metrics::names::TURN_TOKEN_USAGE_METRIC;
use codex_otel::metrics::names::TURN_TOOL_CALL_METRIC;
use codex_protocol::config_types::ReasoningSummary as ReasoningSummaryConfig;
use codex_protocol::items::TurnItem;
use codex_protocol::models::BaseInstructions;
use codex_protocol::models::ContentItem;
use codex_protocol::models::ResponseInputItem;
use codex_protocol::models::ResponseItem;
use codex_protocol::protocol::RolloutItem;
use codex_protocol::user_input::UserInput;

use crate::features::Feature;
pub(crate) use compact::CompactTask;
pub(crate) use ghost_snapshot::GhostSnapshotTask;
pub(crate) use regular::RegularTask;
pub(crate) use review::ReviewTask;
pub(crate) use undo::UndoTask;
pub(crate) use user_shell::UserShellCommandMode;
pub(crate) use user_shell::UserShellCommandTask;
pub(crate) use user_shell::execute_user_shell_command;

const GRACEFULL_INTERRUPTION_TIMEOUT_MS: u64 = 100;
const TURN_ABORTED_INTERRUPTED_GUIDANCE: &str = "The user interrupted the previous turn on purpose. Any running unified exec processes were terminated. If any tools/commands were aborted, they may have partially executed; verify current state before retrying.";
const CHAT_TREE_SUMMARY_SYSTEM_INSTRUCTIONS: &str =
    "You generate a concise summary label for one completed assistant turn.";
const CHAT_TREE_SUMMARY_REQUEST_TEMPLATE: &str = "Summarize this turn for a chat tree node.\nRequirements:\n- single line\n- at most 96 characters\n- no markdown\n- no surrounding quotes\n- describe user intent + assistant outcome\n\nUser message:\n{user_message}\n\nAssistant message:\n{assistant_message}";

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
    let mut summary = line.chars().take(96).collect::<String>();
    if line.chars().count() > 96 {
        summary.push_str("...");
    }
    summary
}

fn abort_reason_summary(reason: &TurnAbortReason) -> &'static str {
    match reason {
        TurnAbortReason::Interrupted => "turn interrupted",
        TurnAbortReason::Replaced => "turn replaced",
        TurnAbortReason::ReviewEnded => "turn review ended",
    }
}

/// Thin wrapper that exposes the parts of [`Session`] task runners need.
#[derive(Clone)]
pub(crate) struct SessionTaskContext {
    session: Arc<Session>,
}

impl SessionTaskContext {
    pub(crate) fn new(session: Arc<Session>) -> Self {
        Self { session }
    }

    pub(crate) fn clone_session(&self) -> Arc<Session> {
        Arc::clone(&self.session)
    }

    pub(crate) fn auth_manager(&self) -> Arc<AuthManager> {
        Arc::clone(&self.session.services.auth_manager)
    }

    pub(crate) fn models_manager(&self) -> Arc<ModelsManager> {
        Arc::clone(&self.session.services.models_manager)
    }
}

/// Async task that drives a [`Session`] turn.
///
/// Implementations encapsulate a specific Codex workflow (regular chat,
/// reviews, ghost snapshots, etc.). Each task instance is owned by a
/// [`Session`] and executed on a background Tokio task. The trait is
/// intentionally small: implementers identify themselves via
/// [`SessionTask::kind`], perform their work in [`SessionTask::run`], and may
/// release resources in [`SessionTask::abort`].
#[async_trait]
pub(crate) trait SessionTask: Send + Sync + 'static {
    /// Describes the type of work the task performs so the session can
    /// surface it in telemetry and UI.
    fn kind(&self) -> TaskKind;

    /// Returns the tracing name for a spawned task span.
    fn span_name(&self) -> &'static str;

    /// Executes the task until completion or cancellation.
    ///
    /// Implementations typically stream protocol events using `session` and
    /// `ctx`, returning an optional final agent message when finished. The
    /// provided `cancellation_token` is cancelled when the session requests an
    /// abort; implementers should watch for it and terminate quickly once it
    /// fires. Returning [`Some`] yields a final message that
    /// [`Session::on_task_finished`] will emit to the client.
    async fn run(
        self: Arc<Self>,
        session: Arc<SessionTaskContext>,
        ctx: Arc<TurnContext>,
        input: Vec<UserInput>,
        cancellation_token: CancellationToken,
    ) -> Option<String>;

    /// Gives the task a chance to perform cleanup after an abort.
    ///
    /// The default implementation is a no-op; override this if additional
    /// teardown or notifications are required once
    /// [`Session::abort_all_tasks`] cancels the task.
    async fn abort(&self, session: Arc<SessionTaskContext>, ctx: Arc<TurnContext>) {
        let _ = (session, ctx);
    }
}

impl Session {
    pub async fn spawn_task<T: SessionTask>(
        self: &Arc<Self>,
        turn_context: Arc<TurnContext>,
        input: Vec<UserInput>,
        task: T,
    ) {
        self.abort_all_tasks(TurnAbortReason::Replaced).await;
        self.clear_connector_selection().await;

        let task: Arc<dyn SessionTask> = Arc::new(task);
        let task_kind = task.kind();
        let span_name = task.span_name();
        let started_at = Instant::now();
        turn_context
            .turn_timing_state
            .mark_turn_started(started_at)
            .await;
        let token_usage_at_turn_start = self.total_token_usage().await.unwrap_or_default();

        let cancellation_token = CancellationToken::new();
        let done = Arc::new(Notify::new());

        let timer = turn_context
            .session_telemetry
            .start_timer(TURN_E2E_DURATION_METRIC, &[])
            .ok();

        let done_clone = Arc::clone(&done);
        let handle = {
            let session_ctx = Arc::new(SessionTaskContext::new(Arc::clone(self)));
            let ctx = Arc::clone(&turn_context);
            let task_for_run = Arc::clone(&task);
            let task_cancellation_token = cancellation_token.child_token();
            // Task-owned turn spans keep a core-owned span open for the
            // full task lifecycle after the submission dispatch span ends.
            let task_span = info_span!(
                "turn",
                otel.name = span_name,
                thread.id = %self.conversation_id,
                turn.id = %turn_context.sub_id,
                model = %turn_context.model_info.slug,
            );
            tokio::spawn(
                async move {
                    let ctx_for_finish = Arc::clone(&ctx);
                    let last_agent_message = task_for_run
                        .run(
                            Arc::clone(&session_ctx),
                            ctx,
                            input,
                            task_cancellation_token.child_token(),
                        )
                        .await;
                    let sess = session_ctx.clone_session();
                    sess.flush_rollout().await;
                    if !task_cancellation_token.is_cancelled() {
                        // Emit completion uniformly from spawn site so all tasks share the same lifecycle.
                        sess.on_task_finished(Arc::clone(&ctx_for_finish), last_agent_message)
                            .await;
                    }
                    done_clone.notify_waiters();
                }
                .instrument(task_span),
            )
        };

        let running_task = RunningTask {
            done,
            handle: Arc::new(AbortOnDropHandle::new(handle)),
            kind: task_kind,
            task,
            cancellation_token,
            turn_context: Arc::clone(&turn_context),
            _timer: timer,
        };
        self.register_new_active_task(running_task, token_usage_at_turn_start)
            .await;
    }

    pub async fn abort_all_tasks(self: &Arc<Self>, reason: TurnAbortReason) {
        if let Some(mut active_turn) = self.take_active_turn().await {
            for task in active_turn.drain_tasks() {
                self.handle_task_abort(task, reason.clone()).await;
            }
            // Let interrupted tasks observe cancellation before dropping pending approvals, or an
            // in-flight approval wait can surface as a model-visible rejection before TurnAborted.
            active_turn.clear_pending().await;
        }
        if reason == TurnAbortReason::Interrupted {
            self.close_unified_exec_processes().await;
        }
    }

    pub async fn on_task_finished(
        self: &Arc<Self>,
        turn_context: Arc<TurnContext>,
        last_agent_message: Option<String>,
    ) {
        turn_context
            .turn_metadata_state
            .cancel_git_enrichment_task();

        let mut active = self.active_turn.lock().await;
        let mut pending_input = Vec::<ResponseInputItem>::new();
        let mut should_clear_active_turn = false;
        let mut token_usage_at_turn_start = None;
        let mut turn_tool_calls = 0_u64;
        if let Some(at) = active.as_mut()
            && at.remove_task(&turn_context.sub_id)
        {
            let mut ts = at.turn_state.lock().await;
            pending_input = ts.take_pending_input();
            turn_tool_calls = ts.tool_calls;
            token_usage_at_turn_start = Some(ts.token_usage_at_turn_start.clone());
            should_clear_active_turn = true;
        }
        if should_clear_active_turn {
            *active = None;
        }
        drop(active);
        if !pending_input.is_empty() {
            let pending_response_items = pending_input
                .into_iter()
                .map(ResponseItem::from)
                .collect::<Vec<_>>();
            for response_item in pending_response_items {
                if let Some(TurnItem::UserMessage(user_message)) = parse_turn_item(&response_item) {
                    // Keep leftover user input on the same persistence + lifecycle path as the
                    // normal pre-sampling drain. This helper records the response item once, then
                    // emits ItemStarted/UserMessage and ItemCompleted/UserMessage for clients.
                    self.record_user_prompt_and_emit_turn_item(
                        turn_context.as_ref(),
                        &user_message.content,
                        response_item,
                    )
                    .await;
                } else {
                    self.record_conversation_items(
                        turn_context.as_ref(),
                        std::slice::from_ref(&response_item),
                    )
                    .await;
                }
            }
        }
        // Emit token usage metrics.
        if let Some(token_usage_at_turn_start) = token_usage_at_turn_start {
            // TODO(jif): drop this
            let tmp_mem = (
                "tmp_mem_enabled",
                if self.enabled(Feature::MemoryTool) {
                    "true"
                } else {
                    "false"
                },
            );
            self.services.session_telemetry.histogram(
                TURN_TOOL_CALL_METRIC,
                i64::try_from(turn_tool_calls).unwrap_or(i64::MAX),
                &[tmp_mem],
            );
            let total_token_usage = self.total_token_usage().await.unwrap_or_default();
            let turn_token_usage = crate::protocol::TokenUsage {
                input_tokens: (total_token_usage.input_tokens
                    - token_usage_at_turn_start.input_tokens)
                    .max(0),
                cached_input_tokens: (total_token_usage.cached_input_tokens
                    - token_usage_at_turn_start.cached_input_tokens)
                    .max(0),
                output_tokens: (total_token_usage.output_tokens
                    - token_usage_at_turn_start.output_tokens)
                    .max(0),
                reasoning_output_tokens: (total_token_usage.reasoning_output_tokens
                    - token_usage_at_turn_start.reasoning_output_tokens)
                    .max(0),
                total_tokens: (total_token_usage.total_tokens
                    - token_usage_at_turn_start.total_tokens)
                    .max(0),
            };
            self.services.session_telemetry.histogram(
                TURN_TOKEN_USAGE_METRIC,
                turn_token_usage.total_tokens,
                &[("token_type", "total"), tmp_mem],
            );
            self.services.session_telemetry.histogram(
                TURN_TOKEN_USAGE_METRIC,
                turn_token_usage.input_tokens,
                &[("token_type", "input"), tmp_mem],
            );
            self.services.session_telemetry.histogram(
                TURN_TOKEN_USAGE_METRIC,
                turn_token_usage.cached_input(),
                &[("token_type", "cached_input"), tmp_mem],
            );
            self.services.session_telemetry.histogram(
                TURN_TOKEN_USAGE_METRIC,
                turn_token_usage.output_tokens,
                &[("token_type", "output"), tmp_mem],
            );
            self.services.session_telemetry.histogram(
                TURN_TOKEN_USAGE_METRIC,
                turn_token_usage.reasoning_output_tokens,
                &[("token_type", "reasoning_output"), tmp_mem],
            );
        }
        let last_user_message_for_summary = self
            .clone_history()
            .await
            .raw_items()
            .iter()
            .rev()
            .find_map(|item| {
                if let ResponseItem::Message { role, content, .. } = item
                    && role == "user"
                {
                    return crate::compact::content_items_to_text(content);
                }
                None
            });
        let last_agent_message_for_summary = last_agent_message.clone();
        let chat_tree = self
            .finalize_chat_tree_node(&turn_context.sub_id, None)
            .await;
        let has_chat_tree = chat_tree.is_some();
        let event = EventMsg::TurnComplete(TurnCompleteEvent {
            turn_id: turn_context.sub_id.clone(),
            last_agent_message,
            chat_tree,
        });
        self.send_event(turn_context.as_ref(), event).await;
        if has_chat_tree {
            info!(
                turn_id = turn_context.sub_id.as_str(),
                "spawning async chat tree summary request after TurnComplete"
            );
            let sess = Arc::clone(self);
            let turn_context = Arc::clone(&turn_context);
            tokio::spawn(async move {
                sess.run_async_chat_tree_summary_job(
                    turn_context,
                    last_user_message_for_summary,
                    last_agent_message_for_summary,
                )
                .await;
            });
        } else {
            warn!(
                turn_id = turn_context.sub_id.as_str(),
                "skipping async chat tree summary because node was missing"
            );
        }
    }

    async fn run_async_chat_tree_summary_job(
        self: &Arc<Self>,
        turn_context: Arc<TurnContext>,
        last_user_message: Option<String>,
        last_agent_message: Option<String>,
    ) {
        let node_id = turn_context.sub_id.clone();
        let user_message = last_user_message.as_deref().unwrap_or("(none)");
        let assistant_message = last_agent_message.as_deref().unwrap_or("(none)");
        let request_payload = CHAT_TREE_SUMMARY_REQUEST_TEMPLATE
            .replace("{user_message}", user_message)
            .replace("{assistant_message}", assistant_message);
        let prompt = Prompt {
            input: vec![ResponseItem::Message {
                id: None,
                role: "user".to_string(),
                content: vec![ContentItem::InputText {
                    text: request_payload.clone(),
                }],
                end_turn: None,
                phase: None,
            }],
            tools: vec![],
            parallel_tool_calls: false,
            base_instructions: BaseInstructions {
                text: CHAT_TREE_SUMMARY_SYSTEM_INSTRUCTIONS.to_string(),
            },
            personality: None,
            output_schema: None,
        };
        let mut client_session = self.services.model_client.new_session();
        let turn_metadata_header = turn_context.turn_metadata_state.current_header_value();
        let started_at = Instant::now();
        info!(
            turn_id = node_id.as_str(),
            node_id = node_id.as_str(),
            model = turn_context.model_info.slug.as_str(),
            request_payload_len = request_payload.len(),
            turn_metadata_header = turn_metadata_header.as_deref().unwrap_or(""),
            "chat tree summary async request sending"
        );
        let mut stream = match client_session
            .stream(
                &prompt,
                &turn_context.model_info,
                &turn_context.session_telemetry,
                None,
                ReasoningSummaryConfig::None,
                turn_context.config.service_tier,
                turn_metadata_header.as_deref(),
            )
            .await
        {
            Ok(stream) => stream,
            Err(err) => {
                warn!(
                    turn_id = node_id.as_str(),
                    node_id = node_id.as_str(),
                    error = %err,
                    "chat tree summary async request failed to start"
                );
                return;
            }
        };
        let mut response_from_item: Option<String> = None;
        let mut response_from_deltas = String::new();
        let mut saw_completed = false;
        while let Some(event_result) = stream.next().await {
            let event = match event_result {
                Ok(event) => event,
                Err(err) => {
                    warn!(
                        turn_id = node_id.as_str(),
                        node_id = node_id.as_str(),
                        error = %err,
                        "chat tree summary async request stream error"
                    );
                    return;
                }
            };
            trace!(
                turn_id = node_id.as_str(),
                node_id = node_id.as_str(),
                response_event = ?event,
                "chat tree summary async response event received"
            );
            match event {
                ResponseEvent::OutputTextDelta(delta) => {
                    response_from_deltas.push_str(&delta);
                }
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
        if !saw_completed {
            warn!(
                turn_id = node_id.as_str(),
                node_id = node_id.as_str(),
                "chat tree summary async stream ended before completed"
            );
            return;
        }
        let response_payload = response_from_item.or_else(|| {
            let trimmed = response_from_deltas.trim();
            if trimmed.is_empty() {
                None
            } else {
                Some(trimmed.to_string())
            }
        });
        let Some(response_payload) = response_payload else {
            warn!(
                turn_id = node_id.as_str(),
                node_id = node_id.as_str(),
                "chat tree summary async response had no assistant text"
            );
            return;
        };
        info!(
            turn_id = node_id.as_str(),
            node_id = node_id.as_str(),
            response_payload_len = response_payload.len(),
            "chat tree summary async response finalized"
        );
        let summary = summarize_for_chat_tree(Some(response_payload.as_str()), "");
        if summary.is_empty() {
            warn!(
                turn_id = node_id.as_str(),
                node_id = node_id.as_str(),
                "chat tree summary async response normalized to empty summary"
            );
            return;
        }
        let elapsed_ms = started_at.elapsed().as_millis();
        let chat_tree = self
            .finalize_chat_tree_node(&node_id, Some(summary.clone()))
            .await;
        let Some(chat_tree) = chat_tree else {
            warn!(
                turn_id = node_id.as_str(),
                node_id = node_id.as_str(),
                elapsed_ms,
                "chat tree summary async could not finalize node"
            );
            return;
        };
        info!(
            turn_id = node_id.as_str(),
            node_id = node_id.as_str(),
            elapsed_ms,
            summary_len = summary.len(),
            "chat tree summary async node finalized"
        );
        self.send_event(
            turn_context.as_ref(),
            EventMsg::ChatTreeNodeUpdated(ChatTreeNodeUpdatedEvent { chat_tree }),
        )
        .await;
        info!(
            turn_id = node_id.as_str(),
            node_id = node_id.as_str(),
            elapsed_ms = started_at.elapsed().as_millis(),
            "chat tree summary async update event sent"
        );
    }

    async fn register_new_active_task(
        &self,
        task: RunningTask,
        token_usage_at_turn_start: TokenUsage,
    ) {
        let mut active = self.active_turn.lock().await;
        let mut turn = ActiveTurn::default();
        let mut turn_state = turn.turn_state.lock().await;
        turn_state.token_usage_at_turn_start = token_usage_at_turn_start;
        drop(turn_state);
        turn.add_task(task);
        *active = Some(turn);
    }

    async fn take_active_turn(&self) -> Option<ActiveTurn> {
        let mut active = self.active_turn.lock().await;
        active.take()
    }

    pub(crate) async fn close_unified_exec_processes(&self) {
        self.services
            .unified_exec_manager
            .terminate_all_processes()
            .await;
    }

    async fn handle_task_abort(self: &Arc<Self>, task: RunningTask, reason: TurnAbortReason) {
        let sub_id = task.turn_context.sub_id.clone();
        if task.cancellation_token.is_cancelled() {
            return;
        }

        trace!(task_kind = ?task.kind, sub_id, "aborting running task");
        task.cancellation_token.cancel();
        task.turn_context
            .turn_metadata_state
            .cancel_git_enrichment_task();
        let session_task = task.task;

        select! {
            _ = task.done.notified() => {
            },
            _ = tokio::time::sleep(Duration::from_millis(GRACEFULL_INTERRUPTION_TIMEOUT_MS)) => {
                warn!("task {sub_id} didn't complete gracefully after {}ms", GRACEFULL_INTERRUPTION_TIMEOUT_MS);
            }
        }

        task.handle.abort();

        let session_ctx = Arc::new(SessionTaskContext::new(Arc::clone(self)));
        session_task
            .abort(session_ctx, Arc::clone(&task.turn_context))
            .await;

        if reason == TurnAbortReason::Interrupted {
            let marker = ResponseItem::Message {
                id: None,
                role: "user".to_string(),
                content: vec![ContentItem::InputText {
                    text: format!(
                        "{TURN_ABORTED_OPEN_TAG}\n{TURN_ABORTED_INTERRUPTED_GUIDANCE}\n</turn_aborted>"
                    ),
                }],
                end_turn: None,
                phase: None,
            };
            self.record_into_history(std::slice::from_ref(&marker), task.turn_context.as_ref())
                .await;
            self.persist_rollout_items(&[RolloutItem::ResponseItem(marker)])
                .await;
            // Ensure the marker is durably visible before emitting TurnAborted: some clients
            // synchronously re-read the rollout on receipt of the abort event.
            self.flush_rollout().await;
        }

        let chat_tree_summary = summarize_for_chat_tree(None, abort_reason_summary(&reason));
        let chat_tree = self
            .finalize_chat_tree_node(&task.turn_context.sub_id, Some(chat_tree_summary))
            .await;

        let event = EventMsg::TurnAborted(TurnAbortedEvent {
            turn_id: Some(task.turn_context.sub_id.clone()),
            reason,
            chat_tree,
        });
        self.send_event(task.turn_context.as_ref(), event).await;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use pretty_assertions::assert_eq;

    #[test]
    fn summarize_for_chat_tree_limits_length() {
        let long = "a".repeat(120);

        let summary = summarize_for_chat_tree(Some(long.as_str()), "fallback");

        assert_eq!(summary.len(), 99);
        assert_eq!(summary.chars().skip(96).collect::<String>(), "...");
    }

    #[test]
    fn summarize_for_chat_tree_allows_empty_without_fallback() {
        let summary = summarize_for_chat_tree(None, "");

        assert_eq!(summary, "");
    }
}
