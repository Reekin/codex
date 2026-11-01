//! Race a delayed replacement against a slow response without delaying a healthy one.
//!
//! The original request streams to the caller unchanged until it produces a complete tool call,
//! answer, or compaction result, or until its waiting interval elapses. Only then does a
//! replacement start. From that point both candidates are buffered until one becomes ready, and
//! only the selected candidate's events continue to the caller.

use super::*;
use crate::responses_retry::STREAM_IDLE_TIMEOUT_INCREMENT;
use codex_api::ResponseInterrupt;
use codex_protocol::models::MessagePhase;
use futures::future::BoxFuture;
use std::collections::VecDeque;

const MAX_BUFFERED_EVENTS: usize = 65_536;

/// A replacement request that has not been sent yet; polling it sends the request.
type ReplacementRequest = BoxFuture<'static, Result<(ResponseStream, WebsocketSession)>>;

/// Shared between one request's stream driver and the session that issued the request.
#[derive(Default)]
pub(super) struct HedgeState {
    started: AtomicBool,
    /// Connection and continuation state of a replacement that was selected.
    selected_replacement: StdMutex<Option<WebsocketSession>>,
}

impl ModelClientSession {
    #[allow(clippy::too_many_arguments)]
    /// Streams a single model request within the current turn.
    ///
    /// The caller is responsible for passing per-turn settings explicitly (model selection,
    /// reasoning settings, telemetry context, and turn metadata). When `stream_retry_timeout` is
    /// enabled and the caller's retry state allows it, a request that produces no complete tool
    /// call or answer within the waiting interval gets a concurrent replacement on a separate
    /// connection with the same prompt and turn identity. Events stream through unchanged until
    /// that replacement starts. Otherwise this sends one request with the provider's timeout.
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
        self.adopt_selected_replacement();
        let idle_timeout = self.stream_idle_timeout;
        // Requests are boxed: inline request futures would otherwise be embedded in every
        // caller's state machine and overflow the turn task's stack.
        let Some(hedge_after) = idle_timeout.filter(|_| self.hedge_allowed) else {
            return Box::pin(self.stream_once(
                prompt,
                model_info,
                session_telemetry,
                effort,
                summary,
                service_tier,
                responses_metadata,
                inference_trace,
                idle_timeout,
            ))
            .await;
        };

        let hedge_at = tokio::time::Instant::now().checked_add(hedge_after);
        let selection_window = hedge_after.saturating_add(STREAM_IDLE_TIMEOUT_INCREMENT);
        let replacement = self.replacement_request(
            prompt,
            model_info,
            session_telemetry,
            effort.clone(),
            summary,
            service_tier.clone(),
            responses_metadata,
            inference_trace,
            selection_window,
        );
        let original = Box::pin(self.stream_once(
            prompt,
            model_info,
            session_telemetry,
            effort,
            summary,
            service_tier,
            responses_metadata,
            inference_trace,
            // Keep the original eligible for the replacement's whole waiting interval.
            Some(hedge_after.saturating_add(selection_window)),
        ))
        .await?;
        let state = Arc::new(HedgeState::default());
        self.hedge = Some(Arc::clone(&state));
        Ok(hedged_stream(
            original,
            hedge_at,
            replacement,
            selection_window,
            state,
        ))
    }

    /// Reports whether the latest request started a replacement, and adopts the replacement's
    /// connection state if it was selected.
    pub(super) fn adopt_selected_replacement(&mut self) -> bool {
        let Some(hedge) = self.hedge.take() else {
            return false;
        };
        let selected = hedge
            .selected_replacement
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .take();
        if let Some(websocket_session) = selected {
            // Transfer the entire incremental-request state, not only the socket. The original's
            // state is discarded rather than published to the shared connection cache.
            self.websocket_session = websocket_session;
        }
        hedge.started.load(Ordering::Relaxed)
    }

    /// Prepares the replacement without sending it. It owns its inputs because it may start
    /// after `stream` has returned.
    #[allow(clippy::too_many_arguments)]
    fn replacement_request(
        &self,
        prompt: &Prompt,
        model_info: &ModelInfo,
        session_telemetry: &SessionTelemetry,
        effort: Option<ReasoningEffortConfig>,
        summary: ReasoningSummaryConfig,
        service_tier: Option<String>,
        responses_metadata: &CodexResponsesMetadata,
        inference_trace: &InferenceTraceContext,
        idle_timeout: Duration,
    ) -> ReplacementRequest {
        let mut replacement = ModelClientSession {
            client: self.client.clone(),
            // A separate connection and continuation state under the same auth owner, so the
            // replacement neither shares the original socket nor resets the turn's routing state.
            websocket_session: WebsocketSession {
                auth_owner_generation: self.websocket_session.auth_owner_generation,
                ..Default::default()
            },
            stream_idle_timeout: Some(idle_timeout),
            hedge_allowed: false,
            hedge: None,
            cache_websocket_on_drop: false,
            turn_state: Arc::clone(&self.turn_state),
        };
        let prompt = prompt.clone();
        let model_info = model_info.clone();
        let session_telemetry = session_telemetry.clone();
        let responses_metadata = responses_metadata.clone();
        let inference_trace = inference_trace.clone();
        Box::pin(async move {
            let stream = Box::pin(replacement.stream_once(
                &prompt,
                &model_info,
                &session_telemetry,
                effort,
                summary,
                service_tier,
                &responses_metadata,
                &inference_trace,
                Some(idle_timeout),
            ))
            .await?;
            Ok::<_, CodexErr>((stream, std::mem::take(&mut replacement.websocket_session)))
        })
    }
}

/// Returns a stream fed by a driver task that owns both candidates.
fn hedged_stream(
    original: ResponseStream,
    hedge_at: Option<tokio::time::Instant>,
    replacement: ReplacementRequest,
    selection_window: Duration,
    state: Arc<HedgeState>,
) -> ResponseStream {
    let (tx_event, rx_event) = mpsc::channel(RESPONSE_STREAM_CHANNEL_CAPACITY);
    let (tx_interrupt, rx_interrupt) = oneshot::channel();
    let consumer_dropped = CancellationToken::new();
    let stream = ResponseStream {
        buffered: VecDeque::new(),
        routing_affinity: original.routing_affinity.clone(),
        rx_event,
        interrupt: original.interrupt.is_some().then_some(tx_interrupt),
        consumer_dropped: consumer_dropped.clone(),
    };
    let driver = Driver {
        tx_event,
        interrupt: Some(rx_interrupt),
        consumer_dropped,
    };
    tokio::spawn(driver.run(original, hedge_at, replacement, selection_window, state));
    stream
}

struct Driver {
    tx_event: mpsc::Sender<Result<ResponseEvent>>,
    interrupt: Option<oneshot::Receiver<ResponseInterrupt>>,
    consumer_dropped: CancellationToken,
}

enum Forwarded {
    Interrupt(Option<ResponseInterrupt>),
    Event(Option<Result<ResponseEvent>>),
}

enum Candidate {
    Original,
    Replacement,
}

enum Selection {
    Interrupt(Option<ResponseInterrupt>),
    Buffered(Candidate, Result<bool>),
    Connected(Result<(ResponseStream, WebsocketSession)>),
    Expired,
}

impl Driver {
    async fn run(
        mut self,
        mut original: ResponseStream,
        hedge_at: Option<tokio::time::Instant>,
        replacement: ReplacementRequest,
        selection_window: Duration,
        state: Arc<HedgeState>,
    ) {
        if !self.forward(&mut original, hedge_at).await {
            return;
        }
        state.started.store(true, Ordering::Relaxed);
        tracing::warn!(
            ?selection_window,
            "starting a concurrent response request; retaining the original stream"
        );
        if let Some(mut winner) = self
            .select(original, replacement, selection_window, &state)
            .await
        {
            self.forward(&mut winner, /*hedge_at*/ None).await;
        }
    }

    /// Forwards `stream` to the consumer. Returns `true` only when `hedge_at` elapses before the
    /// stream produces a complete result; otherwise returns `false` once the stream or the
    /// consumer ends.
    async fn forward(
        &mut self,
        stream: &mut ResponseStream,
        mut hedge_at: Option<tokio::time::Instant>,
    ) -> bool {
        loop {
            let step = tokio::select! {
                biased;
                () = self.consumer_dropped.cancelled() => return false,
                interrupt = recv_interrupt(&mut self.interrupt) => Forwarded::Interrupt(interrupt),
                event = stream.next() => Forwarded::Event(event),
                () = sleep_until(hedge_at) => return true,
            };
            match step {
                Forwarded::Interrupt(None) => {}
                Forwarded::Interrupt(Some(interrupt)) => {
                    // The consumer is winding this response down; a replacement would not help.
                    hedge_at = None;
                    if let Some(sender) = stream.interrupt.take() {
                        let _ = sender.send(interrupt);
                    }
                }
                Forwarded::Event(None) => return false,
                Forwarded::Event(Some(event)) => {
                    // A complete result or a failure settles the request without a replacement.
                    if !matches!(&event, Ok(event) if !is_complete(event)) {
                        hedge_at = None;
                    }
                    if self.tx_event.send(event).await.is_err() {
                        return false;
                    }
                }
            }
        }
    }

    /// Buffers both candidates until one produces a complete result, and returns it with its
    /// buffered events. Returning drops the other candidate, which cancels it.
    async fn select(
        &mut self,
        original: ResponseStream,
        replacement_request: ReplacementRequest,
        selection_window: Duration,
        state: &HedgeState,
    ) -> Option<ResponseStream> {
        let mut original = Some(original);
        let mut replacement_request = Some(replacement_request);
        let mut replacement = None;
        let mut replacement_session = None;
        let mut pending_interrupt = None;
        // Bound the whole selection window even if a stalled provider sends only heartbeats or
        // reasoning.
        let deadline = tokio::time::sleep(selection_window);
        tokio::pin!(deadline);
        loop {
            let step = tokio::select! {
                biased;
                () = self.consumer_dropped.cancelled() => return None,
                interrupt = recv_interrupt(&mut self.interrupt) => Selection::Interrupt(interrupt),
                ready = buffer_next(original.as_mut()) => {
                    Selection::Buffered(Candidate::Original, ready)
                }
                ready = buffer_next(replacement.as_mut()) => {
                    Selection::Buffered(Candidate::Replacement, ready)
                }
                result = connect(&mut replacement_request) => Selection::Connected(result),
                () = &mut deadline => Selection::Expired,
            };
            let error = match step {
                Selection::Interrupt(None) | Selection::Buffered(_, Ok(false)) => continue,
                Selection::Interrupt(Some(interrupt)) => {
                    // The consumer is winding this response down: stop the race and keep the
                    // original, whose earlier events it has already seen.
                    if let Some(mut original) = original.take() {
                        if let Some(sender) = original.interrupt.take() {
                            let _ = sender.send(interrupt);
                        }
                        return Some(original);
                    }
                    pending_interrupt = Some(interrupt);
                    continue;
                }
                Selection::Buffered(Candidate::Original, Ok(true)) => return original,
                Selection::Buffered(Candidate::Replacement, Ok(true)) => {
                    *state
                        .selected_replacement
                        .lock()
                        .unwrap_or_else(std::sync::PoisonError::into_inner) =
                        replacement_session.take();
                    let mut winner = replacement?;
                    if let Some(interrupt) = pending_interrupt
                        && let Some(sender) = winner.interrupt.take()
                    {
                        let _ = sender.send(interrupt);
                    }
                    return Some(winner);
                }
                Selection::Connected(Ok((stream, session))) => {
                    replacement = Some(stream);
                    replacement_session = Some(session);
                    continue;
                }
                Selection::Buffered(Candidate::Original, Err(error)) => {
                    original = None;
                    error
                }
                Selection::Buffered(Candidate::Replacement, Err(error))
                | Selection::Connected(Err(error)) => {
                    replacement = None;
                    error
                }
                Selection::Expired => {
                    let error =
                        CodexErr::Stream("timed out waiting for a complete response item".into());
                    let _ = self.tx_event.send(Err(error)).await;
                    return None;
                }
            };
            // A failed candidate leaves the other eligible; report only when neither remains.
            if original.is_some() || replacement.is_some() || replacement_request.is_some() {
                continue;
            }
            let _ = self.tx_event.send(Err(error)).await;
            return None;
        }
    }
}

async fn recv_interrupt(
    receiver: &mut Option<oneshot::Receiver<ResponseInterrupt>>,
) -> Option<ResponseInterrupt> {
    let Some(pending) = receiver else {
        return std::future::pending().await;
    };
    let interrupt = pending.await.ok();
    *receiver = None;
    interrupt
}

async fn sleep_until(deadline: Option<tokio::time::Instant>) {
    match deadline {
        Some(deadline) => tokio::time::sleep_until(deadline).await,
        None => std::future::pending().await,
    }
}

async fn connect(
    request: &mut Option<ReplacementRequest>,
) -> Result<(ResponseStream, WebsocketSession)> {
    let Some(pending) = request else {
        return std::future::pending().await;
    };
    let result = pending.await;
    *request = None;
    result
}

/// Buffers the candidate's next event and reports whether it completes a usable result.
async fn buffer_next(candidate: Option<&mut ResponseStream>) -> Result<bool> {
    let Some(stream) = candidate else {
        return std::future::pending().await;
    };
    // Read the transport directly: `buffered` holds this candidate's undelivered events.
    match stream.rx_event.recv().await {
        Some(Ok(event)) => {
            let ready = is_complete(&event);
            stream.buffered.push_back(event);
            if !ready && stream.buffered.len() >= MAX_BUFFERED_EVENTS {
                return Err(CodexErr::Stream(
                    "response candidate buffer limit exceeded".into(),
                ));
            }
            Ok(ready)
        }
        Some(Err(error)) => Err(error),
        None => Err(CodexErr::Stream(
            "stream closed before a complete response item".into(),
        )),
    }
}

/// Whether `event` selects its stream: a complete tool call, final answer, or compaction result,
/// or a terminal response so the caller's existing empty-final-answer recovery still applies.
fn is_complete(event: &ResponseEvent) -> bool {
    match event {
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
        ResponseEvent::Completed { .. } => true,
        ResponseEvent::Created { .. }
        | ResponseEvent::SafetyBuffering(_)
        | ResponseEvent::OutputItemAdded(_)
        | ResponseEvent::ServerModel(_)
        | ResponseEvent::ModelVerifications(_)
        | ResponseEvent::TurnModerationMetadata(_)
        | ResponseEvent::GuardianAssessment(_)
        | ResponseEvent::ServerReasoningIncluded(_)
        | ResponseEvent::OutputTextDelta(_)
        | ResponseEvent::ToolCallInputDelta { .. }
        | ResponseEvent::ReasoningSummaryDelta { .. }
        | ResponseEvent::ReasoningSummaryDone { .. }
        | ResponseEvent::ReasoningContentDelta { .. }
        | ResponseEvent::ReasoningSummaryPartAdded { .. }
        | ResponseEvent::RateLimits(_)
        | ResponseEvent::ModelsEtag(_) => false,
    }
}

#[cfg(test)]
#[path = "client_hedged_stream_tests.rs"]
mod tests;
