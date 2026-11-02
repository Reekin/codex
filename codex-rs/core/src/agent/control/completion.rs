//! Delivers terminal child results and completion activity to the agent tree.
//!
//! Sessions capture terminal state; the controller owns routing and queue-only delivery.
//! Delivery remains best effort; metrics distinguish queue acceptance from delivery failure.

use super::LocalAgentControl;
use crate::TurnStartOptions;
use crate::agent::api::AgentTurnOutcome;
use crate::agent_communication::AgentCommunicationContext;
use crate::session_prefix::format_guardian_interruption_message;
use crate::session_prefix::format_inter_agent_completion_message;
use codex_protocol::AgentPath;
use codex_protocol::items::SubAgentActivityItem;
use codex_protocol::protocol::AgentStatus;
use codex_protocol::protocol::CodexErrorInfo;
use codex_protocol::protocol::InterAgentCommunication;
use codex_protocol::protocol::MultiAgentVersion;
use codex_protocol::protocol::SessionSource;
use codex_protocol::protocol::SubAgentActivityKind;
use codex_protocol::protocol::SubAgentSource;
use codex_rollout_trace::AgentResultTracePayload;
use codex_rollout_trace::ThreadTraceContext;
use tracing::debug;

impl LocalAgentControl {
    /// Routes a captured terminal outcome without retaining the child's live turn context.
    pub(crate) async fn notify_parent_of_terminal_turn(
        &self,
        outcome: AgentTurnOutcome,
        trace: &ThreadTraceContext,
    ) {
        match outcome.multi_agent_version {
            MultiAgentVersion::V2 => {}
            MultiAgentVersion::V1 => {
                return self.notify_legacy_parent_of_terminal_turn(outcome).await;
            }
            MultiAgentVersion::Disabled => return,
        }
        let SessionSource::SubAgent(SubAgentSource::ThreadSpawn {
            parent_thread_id,
            agent_path: Some(child_agent_path),
            ..
        }) = &outcome.source
        else {
            return;
        };
        let parent_thread_id = *parent_thread_id;
        let status = outcome.status;
        let Some(parent_agent_path) = child_agent_path
            .as_str()
            .rsplit_once('/')
            .and_then(|(parent, _)| AgentPath::try_from(parent).ok())
        else {
            return;
        };

        if matches!(status, AgentStatus::Completed(_))
            && let Some(parent_turn_id) = outcome.parent_turn_id
        {
            let initiating_thread_id = match outcome.initiating_agent_path.as_ref() {
                Some(initiating_agent_path) if initiating_agent_path != &parent_agent_path => {
                    self.runtime.resolve_agent_reference(
                        outcome.thread_id,
                        &outcome.source,
                        initiating_agent_path.as_str(),
                    )
                    .await
                    .inspect_err(|err| {
                        debug!(
                            "failed to resolve completed activity initiator {initiating_agent_path}: {err}"
                        );
                    })
                    .ok()
                }
                _ => Some(parent_thread_id),
            };
            if let Some(initiating_thread_id) = initiating_thread_id
                && let Err(err) = self
                    .emit_sub_agent_activity(
                        initiating_thread_id,
                        parent_turn_id,
                        SubAgentActivityItem {
                            model: None,
                            reasoning_effort: None,
                            id: format!("subagent-completed-{}", outcome.turn_id),
                            kind: SubAgentActivityKind::Completed,
                            agent_thread_id: outcome.thread_id,
                            agent_path: child_agent_path.clone(),
                        },
                    )
                    .await
            {
                debug!(
                    "failed to emit completed activity to initiating thread {initiating_thread_id}: {err}"
                );
            }
        }

        let message = match (&status, outcome.error_info) {
            (AgentStatus::Errored(error), Some(CodexErrorInfo::TooManyDenials)) => {
                Some(format_guardian_interruption_message(
                    parent_agent_path.clone(),
                    child_agent_path.clone(),
                    error,
                ))
            }
            _ => format_inter_agent_completion_message(
                parent_agent_path.clone(),
                child_agent_path.clone(),
                &status,
            ),
        };
        let Some(message) = message else {
            return;
        };
        // `communication` owns the message. Keep a second copy only when the
        // recorder will actually need it after parent delivery succeeds.
        let trace_message = trace.is_enabled().then(|| message.clone());
        let communication = InterAgentCommunication::new(
            child_agent_path.clone(),
            parent_agent_path,
            Vec::new(),
            message,
            /*trigger_turn*/ false,
        );
        let context = AgentCommunicationContext::subagent_result(outcome.thread_id, &status);
        let delivery = self
            .send_inter_agent_communication(
                parent_thread_id,
                communication,
                context,
                TurnStartOptions::default(),
            )
            .await;
        if let Some(metrics) = codex_otel::global() {
            let _ = metrics.counter(
                "codex.multi_agent.result_delivery",
                /*inc*/ 1,
                &[(
                    "outcome",
                    if delivery.is_ok() { "queued" } else { "failed" },
                )],
            );
        }
        if let Err(err) = delivery {
            debug!("failed to notify parent thread {parent_thread_id}: {err}");
            return;
        }
        if let Some(message) = trace_message {
            trace.record_agent_result_interaction(
                outcome.turn_id.as_str(),
                parent_thread_id,
                &AgentResultTracePayload {
                    child_agent_path: child_agent_path.as_str(),
                    message: &message,
                    status: &status,
                },
            );
        }
    }

    /// Queues a MAv1 child's terminal status for its loaded parent, including follow-up turns.
    /// Sessions report MAv1 turns here only when the parent runs with subagent continuation.
    async fn notify_legacy_parent_of_terminal_turn(&self, outcome: AgentTurnOutcome) {
        let SessionSource::SubAgent(SubAgentSource::ThreadSpawn {
            parent_thread_id,
            agent_path,
            ..
        }) = &outcome.source
        else {
            return;
        };
        let Ok(state) = self.runtime.upgrade() else {
            return;
        };
        let Ok(parent_thread) = state.get_thread(*parent_thread_id).await else {
            debug!("failed to notify unloaded parent thread {parent_thread_id}");
            return;
        };
        let child_reference = agent_path
            .as_ref()
            .map(ToString::to_string)
            .unwrap_or_else(|| outcome.thread_id.to_string());
        let resumes_parent = crate::agent_communication::status_resumes_parent(&outcome.status);
        let notification =
            super::legacy_subagent_notification(&parent_thread, &child_reference, outcome.status)
                .await;
        if resumes_parent {
            parent_thread
                .deliver_subagent_notification(notification)
                .await;
        } else {
            parent_thread
                .session
                .inject_no_new_turn(vec![notification], /*current_turn_context*/ None)
                .await;
        }
    }
}
