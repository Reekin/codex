use super::Winner;
use super::select_response;
use crate::client_common::ResponseEvent;
use crate::client_common::ResponseStream;
use codex_protocol::error::CodexErr;
use codex_protocol::error::Result;
use futures::StreamExt;
use pretty_assertions::assert_eq;
use std::time::Duration;
use tokio::sync::mpsc;
use tokio_util::sync::CancellationToken;

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
        rx_event,
        consumer_dropped: cancelled,
    }
}

fn completed(id: &str) -> ResponseEvent {
    ResponseEvent::Completed {
        response_id: id.into(),
        token_usage: None,
        usage_metadata: None,
        end_turn: None,
    }
}

#[rstest::rstest]
#[case::primary(7, 4, Winner::Primary)]
#[case::backup(12, 2, Winner::Backup)]
#[case::no_hedge(1, 2, Winner::Primary)]
#[tokio::test(start_paused = true)]
async fn first_complete_response_wins_and_cancels_the_loser(
    #[case] primary_seconds: u64,
    #[case] backup_seconds: u64,
    #[case] expected: Winner,
) {
    let primary_cancelled = CancellationToken::new();
    let backup_cancelled = CancellationToken::new();
    let mut started = false;
    let (winner, mut stream) = select_response(
        async {
            Ok(delayed_stream(
                Duration::from_secs(primary_seconds),
                Ok(completed("primary")),
                primary_cancelled.clone(),
            ))
        },
        async {
            Ok(delayed_stream(
                Duration::from_secs(backup_seconds),
                Ok(completed("backup")),
                backup_cancelled.clone(),
            ))
        },
        Duration::from_secs(5),
        Duration::from_secs(10),
        &mut started,
    )
    .await
    .unwrap();
    assert_eq!(winner, expected);
    assert_eq!(started, primary_seconds > 5);
    let (selected, rejected) = match winner {
        Winner::Primary => (&primary_cancelled, &backup_cancelled),
        Winner::Backup => (&backup_cancelled, &primary_cancelled),
    };
    assert_eq!(rejected.is_cancelled(), started);
    assert!(!selected.is_cancelled());
    assert!(matches!(
        stream.next().await,
        Some(Ok(ResponseEvent::Completed { .. }))
    ));
    drop(stream);
    assert!(selected.is_cancelled());
}

#[tokio::test(start_paused = true)]
async fn failed_backup_does_not_discard_the_live_original() {
    let mut started = false;
    let (winner, _) = select_response(
        async {
            Ok(delayed_stream(
                Duration::from_secs(7),
                Ok(completed("primary")),
                CancellationToken::new(),
            ))
        },
        async { Err(CodexErr::Stream("backup failed".into())) },
        Duration::from_secs(5),
        Duration::from_secs(10),
        &mut started,
    )
    .await
    .unwrap();
    assert_eq!(winner, Winner::Primary);
    assert!(started);
}

#[tokio::test(start_paused = true)]
async fn cancelling_selection_drops_both_requests() {
    let primary_cancelled = CancellationToken::new();
    let backup_cancelled = CancellationToken::new();
    let mut started = false;
    let result = tokio::time::timeout(
        Duration::from_secs(6),
        select_response(
            async {
                Ok(delayed_stream(
                    Duration::from_secs(20),
                    Ok(completed("primary")),
                    primary_cancelled.clone(),
                ))
            },
            async {
                Ok(delayed_stream(
                    Duration::from_secs(20),
                    Ok(completed("backup")),
                    backup_cancelled.clone(),
                ))
            },
            Duration::from_secs(5),
            Duration::from_secs(10),
            &mut started,
        ),
    )
    .await;
    assert!(result.is_err());
    assert!(primary_cancelled.is_cancelled());
    assert!(backup_cancelled.is_cancelled());
}
