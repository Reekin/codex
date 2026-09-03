//! Errors returned by the shared Codex HTTP transport.

use crate::client::HttpError;
use http::HeaderMap;
use http::StatusCode;
use std::time::Duration;
use thiserror::Error;

#[derive(Debug, Error)]
pub enum TransportError {
    #[error("http {status}: {body:?}")]
    Http {
        status: StatusCode,
        url: Option<String>,
        headers: Option<HeaderMap>,
        body: Option<String>,
    },
    #[error("retry limit reached")]
    RetryLimit,
    #[error("timeout")]
    Timeout,
    #[error("connection failed: {0}")]
    Connection(#[source] HttpError),
    #[error("network error: {0}")]
    Network(String),
    #[error(
        "large request upload stalled after {elapsed:?}: submitted {submitted_bytes} of {total_bytes} bytes at {bytes_per_second} B/s with {estimated_remaining:?} remaining"
    )]
    SlowUpload {
        total_bytes: usize,
        submitted_bytes: usize,
        elapsed: Duration,
        bytes_per_second: u64,
        estimated_remaining: Duration,
    },
    #[error("request build error: {0}")]
    Build(String),
}

#[derive(Debug, Error)]
pub enum StreamError {
    #[error("stream failed: {0}")]
    Stream(String),
    #[error("timeout")]
    Timeout,
}
