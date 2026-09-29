//! Select one response before exposing either candidate to history or tool execution.

use super::*;
use crate::responses_retry::STREAM_IDLE_TIMEOUT_INCREMENT;
use codex_protocol::models::MessagePhase;
use std::future::Future;

const MAX_BUFFERED_EVENTS: usize = 65_536;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Winner {
    Primary,
    Backup,
}

impl ModelClientSession {
    /// Buffer candidate output until a complete tool call or answer selects its stream.
    /// A delayed backup uses a separate connection and the same prompt and turn identity.
    #[allow(clippy::too_many_arguments)]
    pub async fn stream(
        &mut self,
        prompt: &Prompt,
        model_info: &ModelInfo,
        session_telemetry: &SessionTelemetry,
        effort: Option<ReasoningEffortConfig>,
        summary: ReasoningSummaryConfig,
        service_tier: Option<String>,
        responses_metadata: &CodexResponsesMetadata,
        inference_trace: &InferenceTraceContext,
    ) -> Result<ResponseStream> {
        self.hedge_started = false;
        let hedge_after = self.stream_idle_timeout;
        if !self.hedge_allowed {
            return self
                .stream_once(
                    prompt,
                    model_info,
                    session_telemetry,
                    effort,
                    summary,
                    service_tier,
                    responses_metadata,
                    inference_trace,
                    hedge_after,
                )
                .await;
        }

        let backup_wait = hedge_after.saturating_add(STREAM_IDLE_TIMEOUT_INCREMENT);
        let mut backup = ModelClientSession {
            client: self.client.clone(),
            websocket_session: WebsocketSession::default(),
            stream_idle_timeout: backup_wait,
            hedge_allowed: false,
            hedge_started: false,
            cache_websocket_on_drop: false,
            turn_state: Arc::clone(&self.turn_state),
        };
        let mut hedge_started = false;
        let result = select_response(
            self.stream_once(
                prompt,
                model_info,
                session_telemetry,
                effort.clone(),
                summary,
                service_tier.clone(),
                responses_metadata,
                inference_trace,
                hedge_after.saturating_add(backup_wait),
            ),
            backup.stream_once(
                prompt,
                model_info,
                session_telemetry,
                effort,
                summary,
                service_tier,
                responses_metadata,
                inference_trace,
                backup_wait,
            ),
            hedge_after,
            backup_wait,
            &mut hedge_started,
        )
        .await;
        self.hedge_started = hedge_started;
        let (winner, stream) = result?;
        if winner == Winner::Backup {
            // Transfer the entire incremental-request state, not only the socket. The loser
            // cannot publish its connection or response ID into the shared connection cache.
            std::mem::swap(&mut self.websocket_session, &mut backup.websocket_session);
        }
        Ok(stream)
    }
}

async fn select_response<P, B>(
    primary: P,
    backup: B,
    hedge_after: Duration,
    backup_wait: Duration,
    hedge_started: &mut bool,
) -> Result<(Winner, ResponseStream)>
where
    P: Future<Output = Result<ResponseStream>>,
    B: Future<Output = Result<ResponseStream>>,
{
    let primary = ready_response(primary);
    tokio::pin!(primary);
    tokio::select! {
        result = &mut primary => return result.map(|stream| (Winner::Primary, stream)),
        _ = tokio::time::sleep(hedge_after) => {}
    }

    let backup = async {
        *hedge_started = true;
        tracing::warn!(
            ?hedge_after,
            "starting a concurrent response request; retaining the original stream"
        );
        ready_response(backup).await
    };
    tokio::pin!(backup);
    let race = async {
        tokio::select! {
            result = &mut primary => match result {
                Ok(stream) => Ok((Winner::Primary, stream)),
                Err(_) => backup.await.map(|stream| (Winner::Backup, stream)),
            },
            result = &mut backup => match result {
                Ok(stream) => Ok((Winner::Backup, stream)),
                Err(_) => primary.await.map(|stream| (Winner::Primary, stream)),
            },
        }
    };
    // Bound the whole selection window even if a stalled provider sends only heartbeats or
    // reasoning. Dropping these futures also drops both streams and any in-progress connection.
    tokio::time::timeout(backup_wait, race)
        .await
        .map_err(|_| CodexErr::Stream("timed out waiting for a complete response item".into()))?
}

async fn ready_response(
    request: impl Future<Output = Result<ResponseStream>>,
) -> Result<ResponseStream> {
    let mut stream = request.await?;
    let mut buffered = std::collections::VecDeque::new();
    while let Some(event) = stream.next().await {
        let event = event?;
        let ready = match &event {
            ResponseEvent::OutputItemDone(item) => match item {
                ResponseItem::Message {
                    role,
                    content,
                    phase,
                    ..
                } => {
                    role == "assistant"
                        && !content.is_empty()
                        && *phase != Some(MessagePhase::Commentary)
                }
                ResponseItem::AgentMessage { .. }
                | ResponseItem::LocalShellCall { .. }
                | ResponseItem::FunctionCall { .. }
                | ResponseItem::CustomToolCall { .. }
                | ResponseItem::ToolSearchCall { .. }
                | ResponseItem::WebSearchCall { .. }
                | ResponseItem::ImageGenerationCall { .. }
                | ResponseItem::Compaction { .. }
                | ResponseItem::ContextCompaction { .. } => true,
                ResponseItem::AdditionalTools { .. }
                | ResponseItem::Reasoning { .. }
                | ResponseItem::FunctionCallOutput { .. }
                | ResponseItem::CustomToolCallOutput { .. }
                | ResponseItem::ToolSearchOutput { .. }
                | ResponseItem::ConfigurationUpdate { .. }
                | ResponseItem::CompactionTrigger { .. }
                | ResponseItem::Other => false,
            },
            // Preserve legal terminal responses without an ordinary message, including the
            // existing empty-final-answer recovery handled by the caller.
            ResponseEvent::Completed { .. } => true,
            ResponseEvent::Created { .. }
            | ResponseEvent::SafetyBuffering(_)
            | ResponseEvent::OutputItemAdded(_)
            | ResponseEvent::ServerModel(_)
            | ResponseEvent::ModelVerifications(_)
            | ResponseEvent::TurnModerationMetadata(_)
            | ResponseEvent::ServerReasoningIncluded(_)
            | ResponseEvent::OutputTextDelta(_)
            | ResponseEvent::ToolCallInputDelta { .. }
            | ResponseEvent::ReasoningSummaryDelta { .. }
            | ResponseEvent::ReasoningSummaryDone { .. }
            | ResponseEvent::ReasoningContentDelta { .. }
            | ResponseEvent::ReasoningSummaryPartAdded { .. }
            | ResponseEvent::RateLimits(_)
            | ResponseEvent::ModelsEtag(_) => false,
        };
        buffered.push_back(event);
        if ready {
            stream.buffered = buffered;
            return Ok(stream);
        }
        if buffered.len() >= MAX_BUFFERED_EVENTS {
            return Err(CodexErr::Stream(
                "response candidate buffer limit exceeded".into(),
            ));
        }
    }
    Err(CodexErr::Stream(
        "stream closed before a complete response item".into(),
    ))
}

#[cfg(test)]
#[path = "hedged_stream_tests.rs"]
mod tests;
