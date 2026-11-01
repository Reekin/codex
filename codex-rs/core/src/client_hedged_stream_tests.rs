use super::HedgeState;
use super::ReplacementRequest;
use super::hedged_stream;
use crate::client::WebsocketSession;
use crate::client_common::ResponseEvent;
use crate::client_common::ResponseStream;
use codex_protocol::error::CodexErr;
use codex_protocol::error::CodexErrorDetails;
use codex_protocol::error::Result;
use futures::StreamExt;
use pretty_assertions::assert_eq;
use std::sync::Arc;
use std::sync::atomic::AtomicBool;
use std::sync::atomic::Ordering;
use std::time::Duration;
use tokio::sync::mpsc;
use tokio::time::Instant;
use tokio_util::sync::CancellationToken;

const HEDGE_AFTER: Duration = Duration::from_secs(5);
const SELECTION_WINDOW: Duration = Duration::from_secs(10);

fn delayed_stream(
    delay: Duration,
    event: Result<ResponseEvent>,
    cancelled: CancellationToken,
) -> ResponseStream {
    let (tx, rx_event) = mpsc::channel(4);
    let task_cancelled = cancelled.clone();
    tokio::spawn(async move {
        tokio::select! {
            _ = task_cancelled.cancelled() => return,
            _ = tokio::time::sleep(delay) => {}
        }
        let _ = tx.send(event).await;
        task_cancelled.cancelled().await;
    });
    ResponseStream {
        buffered: Default::default(),
        routing_affinity: None,
        rx_event,
        interrupt: None,
        consumer_dropped: cancelled,
    }
}

/// A replacement whose stream is created, and whose delay starts, only once it is sent.
fn replacement(
    delay: Duration,
    event: Result<ResponseEvent>,
    cancelled: CancellationToken,
    sent: Arc<AtomicBool>,
) -> ReplacementRequest {
    Box::pin(async move {
        sent.store(true, Ordering::Relaxed);
        Ok::<_, CodexErr>((
            delayed_stream(delay, event, cancelled),
            WebsocketSession::default(),
        ))
    })
}

fn start(
    original: ResponseStream,
    replacement: ReplacementRequest,
) -> (ResponseStream, Arc<HedgeState>) {
    let state = Arc::new(HedgeState::default());
    let stream = hedged_stream(
        original,
        Some(Instant::now() + HEDGE_AFTER),
        replacement,
        SELECTION_WINDOW,
        Arc::clone(&state),
    );
    (stream, state)
}

fn completed(id: &str) -> ResponseEvent {
    ResponseEvent::Completed {
        response_id: id.into(),
        token_usage: None,
        usage_metadata: None,
        end_turn: None,
    }
}

async fn wait_cancelled(token: &CancellationToken) {
    tokio::time::timeout(Duration::from_secs(1), token.cancelled())
        .await
        .expect("candidate should be cancelled");
}

#[tokio::test(start_paused = true)]
async fn events_stream_through_before_the_waiting_interval() {
    let started_at = Instant::now();
    let sent = Arc::new(AtomicBool::new(false));
    let (mut stream, state) = start(
        delayed_stream(
            Duration::ZERO,
            Ok(ResponseEvent::OutputTextDelta("partial".into())),
            CancellationToken::new(),
        ),
        replacement(
            Duration::ZERO,
            Ok(completed("replacement")),
            CancellationToken::new(),
            Arc::clone(&sent),
        ),
    );

    let event = stream.next().await;
    assert!(matches!(event, Some(Ok(ResponseEvent::OutputTextDelta(_)))));
    assert!(started_at.elapsed() < HEDGE_AFTER);
    assert!(!state.started.load(Ordering::Relaxed));
    assert!(!sent.load(Ordering::Relaxed));
}

#[test_case::test_case(7, 4, "original"; "original")]
#[test_case::test_case(12, 2, "replacement"; "replacement")]
#[test_case::test_case(1, 2, "original"; "no replacement")]
#[tokio::test(start_paused = true)]
async fn first_complete_response_wins_and_cancels_the_loser(
    original_seconds: u64,
    replacement_seconds: u64,
    expected: &str,
) {
    let original_cancelled = CancellationToken::new();
    let replacement_cancelled = CancellationToken::new();
    let sent = Arc::new(AtomicBool::new(false));
    let (mut stream, state) = start(
        delayed_stream(
            Duration::from_secs(original_seconds),
            Ok(completed("original")),
            original_cancelled.clone(),
        ),
        replacement(
            Duration::from_secs(replacement_seconds),
            Ok(completed("replacement")),
            replacement_cancelled.clone(),
            Arc::clone(&sent),
        ),
    );

    let Some(Ok(ResponseEvent::Completed { response_id, .. })) = stream.next().await else {
        panic!("expected the selected response to complete");
    };
    assert_eq!(response_id, expected);
    let started = state.started.load(Ordering::Relaxed);
    assert_eq!(started, Duration::from_secs(original_seconds) > HEDGE_AFTER);
    assert_eq!(sent.load(Ordering::Relaxed), started);
    let (selected, rejected) = if expected == "replacement" {
        (&replacement_cancelled, &original_cancelled)
    } else {
        (&original_cancelled, &replacement_cancelled)
    };
    assert_eq!(rejected.is_cancelled(), started);
    assert!(!selected.is_cancelled());
    assert_eq!(
        state
            .selected_replacement
            .lock()
            .expect("selection lock")
            .is_some(),
        expected == "replacement"
    );
    drop(stream);
    wait_cancelled(selected).await;
}

#[tokio::test(start_paused = true)]
async fn failed_replacement_does_not_discard_the_live_original() {
    let (mut stream, state) = start(
        delayed_stream(
            Duration::from_secs(7),
            Ok(completed("original")),
            CancellationToken::new(),
        ),
        Box::pin(async {
            Err::<(ResponseStream, WebsocketSession), _>(CodexErr::Stream(
                "replacement failed".into(),
            ))
        }),
    );

    let Some(Ok(ResponseEvent::Completed { response_id, .. })) = stream.next().await else {
        panic!("expected the original to complete");
    };
    assert_eq!(response_id, "original");
    assert!(state.started.load(Ordering::Relaxed));
}

#[tokio::test(start_paused = true)]
async fn dropping_the_stream_during_selection_cancels_both_requests() {
    let original_cancelled = CancellationToken::new();
    let replacement_cancelled = CancellationToken::new();
    let (stream, state) = start(
        delayed_stream(
            Duration::from_secs(20),
            Ok(completed("original")),
            original_cancelled.clone(),
        ),
        replacement(
            Duration::from_secs(20),
            Ok(completed("replacement")),
            replacement_cancelled.clone(),
            Arc::new(AtomicBool::new(false)),
        ),
    );

    tokio::time::sleep(HEDGE_AFTER + Duration::from_secs(1)).await;
    assert!(state.started.load(Ordering::Relaxed));
    drop(stream);
    wait_cancelled(&original_cancelled).await;
    wait_cancelled(&replacement_cancelled).await;
}

#[tokio::test(start_paused = true)]
async fn selection_window_bounds_two_stalled_requests() {
    let started_at = Instant::now();
    let (mut stream, _state) = start(
        delayed_stream(
            Duration::from_secs(60),
            Ok(completed("original")),
            CancellationToken::new(),
        ),
        replacement(
            Duration::from_secs(60),
            Ok(completed("replacement")),
            CancellationToken::new(),
            Arc::new(AtomicBool::new(false)),
        ),
    );

    let Some(Err(error)) = stream.next().await else {
        panic!("expected the selection window to end with a stream error");
    };
    assert!(matches!(error.details(), CodexErrorDetails::Stream(_)));
    let elapsed = started_at.elapsed();
    assert!(elapsed >= HEDGE_AFTER + SELECTION_WINDOW);
    assert!(elapsed < HEDGE_AFTER + SELECTION_WINDOW + Duration::from_secs(1));
}
