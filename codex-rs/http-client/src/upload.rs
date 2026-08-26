use crate::TransportError;
use bytes::Bytes;
use futures::stream;
use std::future::pending;
use std::io;
use std::sync::Arc;
use std::sync::atomic::AtomicUsize;
use std::sync::atomic::Ordering;
use std::time::Duration;
use tokio::time::Instant;
use tokio::time::sleep;

const UPLOAD_OBSERVATION_THRESHOLD_BYTES: usize = 1024 * 1024;
const UPLOAD_CHUNK_BYTES: usize = 64 * 1024;
const UPLOAD_GRACE_PERIOD: Duration = Duration::from_secs(10);
const UPLOAD_SAMPLE_PERIOD: Duration = Duration::from_secs(5);
const MIN_UPLOAD_BYTES_PER_SECOND: usize = 256 * 1024;
const MAX_PROJECTED_REMAINING: Duration = Duration::from_secs(30);

#[derive(Clone, Copy, Debug)]
pub(crate) struct UploadPolicy {
    pub(crate) body_threshold: usize,
    pub(crate) chunk_size: usize,
    pub(crate) grace_period: Duration,
    pub(crate) sample_period: Duration,
    pub(crate) min_bytes_per_second: usize,
    pub(crate) max_projected_remaining: Duration,
}

impl Default for UploadPolicy {
    fn default() -> Self {
        Self {
            body_threshold: UPLOAD_OBSERVATION_THRESHOLD_BYTES,
            chunk_size: UPLOAD_CHUNK_BYTES,
            grace_period: UPLOAD_GRACE_PERIOD,
            sample_period: UPLOAD_SAMPLE_PERIOD,
            min_bytes_per_second: MIN_UPLOAD_BYTES_PER_SECOND,
            max_projected_remaining: MAX_PROJECTED_REMAINING,
        }
    }
}

#[derive(Debug)]
pub(crate) struct UploadMonitor {
    total_bytes: usize,
    submitted_bytes: Arc<AtomicUsize>,
    started_at: Instant,
    policy: UploadPolicy,
}

pub(crate) fn monitored_body(
    body: Bytes,
    policy: UploadPolicy,
) -> (reqwest::Body, Option<UploadMonitor>) {
    if body.len() < policy.body_threshold {
        return (reqwest::Body::from(body), None);
    }

    let total_bytes = body.len();
    let submitted_bytes = Arc::new(AtomicUsize::new(0));
    let stream_progress = Arc::clone(&submitted_bytes);
    let chunk_size = policy.chunk_size.max(1);
    let stream = stream::unfold((body, 0usize), move |(body, offset)| {
        let stream_progress = Arc::clone(&stream_progress);
        async move {
            if offset >= body.len() {
                return None;
            }
            let end = offset.saturating_add(chunk_size).min(body.len());
            let chunk = body.slice(offset..end);
            stream_progress.store(end, Ordering::Release);
            Some((Ok::<Bytes, io::Error>(chunk), (body, end)))
        }
    });

    (
        reqwest::Body::wrap_stream(stream),
        Some(UploadMonitor {
            total_bytes,
            submitted_bytes,
            started_at: Instant::now(),
            policy,
        }),
    )
}

impl UploadMonitor {
    pub(crate) async fn wait_for_stall(&self) -> TransportError {
        sleep(self.policy.grace_period).await;
        let mut previous_bytes = 0usize;
        let mut previous_at = self.started_at;

        loop {
            let now = Instant::now();
            let submitted_bytes = self.submitted_bytes.load(Ordering::Acquire);
            if submitted_bytes >= self.total_bytes {
                pending::<()>().await;
            }

            let elapsed = now.duration_since(previous_at);
            let advanced = submitted_bytes.saturating_sub(previous_bytes);
            let bytes_per_second = rate_per_second(advanced, elapsed);
            let remaining_bytes = self.total_bytes.saturating_sub(submitted_bytes);
            let estimated_remaining = projected_duration(remaining_bytes, bytes_per_second);
            if upload_is_stalled(bytes_per_second, estimated_remaining, self.policy) {
                return TransportError::SlowUpload {
                    total_bytes: self.total_bytes,
                    submitted_bytes,
                    elapsed: now.duration_since(self.started_at),
                    bytes_per_second,
                    estimated_remaining,
                };
            }

            previous_bytes = submitted_bytes;
            previous_at = now;
            sleep(self.policy.sample_period).await;
        }
    }
}

fn upload_is_stalled(
    bytes_per_second: u64,
    estimated_remaining: Duration,
    policy: UploadPolicy,
) -> bool {
    bytes_per_second < policy.min_bytes_per_second as u64
        && estimated_remaining > policy.max_projected_remaining
}

fn projected_duration(bytes: usize, bytes_per_second: u64) -> Duration {
    if bytes_per_second == 0 {
        return Duration::MAX;
    }
    let seconds = (bytes as u64).saturating_add(bytes_per_second - 1) / bytes_per_second;
    Duration::from_secs(seconds)
}

fn rate_per_second(bytes: usize, elapsed: Duration) -> u64 {
    let nanos = elapsed.as_nanos();
    if nanos == 0 {
        return u64::MAX;
    }
    ((bytes as u128).saturating_mul(1_000_000_000) / nanos)
        .try_into()
        .unwrap_or(u64::MAX)
}

#[cfg(test)]
#[path = "upload_tests.rs"]
mod tests;
