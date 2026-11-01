//! Shared HTTP transport for fixed clients and route-aware pools, with optional response limits.
//!
//! Each request's limit applies to observed body bytes, including unsuccessful responses;
//! declared content lengths are only an additional early-rejection check.
//!
//! Large encoded JSON bodies are streamed under an upload monitor. A stalled attempt is dropped
//! before completion and replaced by an attempt on a fresh connection, within a request-scoped
//! budget; the attempt after the budget is exhausted runs unmonitored to completion.

use crate::HttpResponse;
use crate::RouteAwareClientPool;
use crate::RouteAwareRequestError;
use crate::client::HttpClient;
use crate::client::HttpClientBackend;
use crate::error::TransportError;
use crate::request::Request;
use crate::request::RequestBody;
use crate::request::Response;
use crate::request_draft::RequestDraft;
use crate::retry_after::RetryAfter;
use crate::upload::UploadMonitor;
use crate::upload::UploadPolicy;
use crate::upload::monitored_body;
use bytes::Bytes;
use bytes::BytesMut;
use futures::StreamExt;
use futures::stream::BoxStream;
use http::HeaderMap;
use http::HeaderValue;
use http::Method;
use http::StatusCode;
use tracing::Level;
use tracing::enabled;
use tracing::trace;
use tracing::warn;

pub type ByteStream = BoxStream<'static, Result<Bytes, TransportError>>;

pub struct StreamResponse {
    pub status: StatusCode,
    pub headers: HeaderMap,
    pub bytes: ByteStream,
}

pub trait HttpTransport: Send + Sync {
    fn execute(
        &self,
        req: Request,
    ) -> impl std::future::Future<Output = Result<Response, TransportError>> + Send;
    fn stream(
        &self,
        req: Request,
    ) -> impl std::future::Future<Output = Result<StreamResponse, TransportError>> + Send;
}

#[derive(Clone, Debug)]
pub struct ReqwestTransport {
    client: TransportClient,
    upload_policy: UploadPolicy,
}

#[derive(Clone, Debug)]
enum TransportClient {
    Fixed(HttpClient),
    RouteAware(Box<RouteAwareClientPool>),
}

impl ReqwestTransport {
    pub fn new(client: reqwest::Client) -> Self {
        Self::from_http_client(HttpClient::new(client))
    }

    pub fn from_http_client(client: HttpClient) -> Self {
        Self {
            client: TransportClient::Fixed(client),
            upload_policy: UploadPolicy::default(),
        }
    }

    /// Uses the pool to resolve the route for every request and redirect URL.
    pub fn from_route_aware_client_pool(client: RouteAwareClientPool) -> Self {
        Self {
            client: TransportClient::RouteAware(Box::new(client)),
            upload_policy: UploadPolicy::default(),
        }
    }

    #[cfg(test)]
    fn with_upload_policy(mut self, upload_policy: UploadPolicy) -> Self {
        self.upload_policy = upload_policy;
        self
    }

    async fn send(&self, req: Request) -> Result<HttpResponse, TransportError> {
        loop {
            let (request, upload_monitor) = self.draft(&req)?;
            let Some(upload_monitor) = upload_monitor else {
                return self.send_draft(request).await;
            };
            // Completing the select drops the incomplete attempt, so its connection cannot
            // return to the pool before the retry starts.
            let stall = tokio::select! {
                result = self.send_draft(request) => return result,
                stall = upload_monitor.wait_for_stall() => stall,
            };
            let attempt = req.record_slow_upload_retry();
            warn!(
                attempt,
                max_attempts = self.upload_policy.max_fresh_connection_retries + 1,
                total_bytes = stall.total_bytes,
                submitted_bytes = stall.submitted_bytes,
                elapsed_ms = stall.elapsed.as_millis(),
                bytes_per_second = stall.bytes_per_second,
                estimated_remaining_ms = stall.estimated_remaining.as_millis(),
                "large request upload stalled; retrying on a fresh connection"
            );
        }
    }

    /// Builds one attempt, monitoring large encoded JSON uploads while the retry budget lasts.
    fn draft(
        &self,
        req: &Request,
    ) -> Result<(RequestDraft, Option<UploadMonitor>), TransportError> {
        let mut prepared = req.prepare_body_for_send().map_err(TransportError::Build)?;

        let Request {
            method,
            url,
            headers: _,
            body,
            compression: _,
            timeout,
            response_body_limit_bytes: _,
            slow_upload_retries: _,
        } = req;
        let monitor_upload = matches!(body, Some(RequestBody::EncodedJson(_)))
            && req.slow_upload_retries() < self.upload_policy.max_fresh_connection_retries;

        let method = Method::from_bytes(method.as_str().as_bytes()).unwrap_or(Method::GET);
        let mut request = RequestDraft::new(method, url.as_str())
            .map_err(|error| Self::map_error(error.into()))?;
        *request.request.timeout_mut() = *timeout;
        let mut upload_monitor = None;
        *request.request.body_mut() = match prepared.body {
            Some(body) if monitor_upload => {
                let body_len = body.len();
                let (body, monitor) = monitored_body(body, self.upload_policy);
                if monitor.is_some() {
                    // A streamed body has no inherent length; keep the upload framed by length.
                    prepared
                        .headers
                        .insert(http::header::CONTENT_LENGTH, HeaderValue::from(body_len));
                }
                upload_monitor = monitor;
                Some(body)
            }
            body => body.map(Into::into),
        };
        request.extend_headers(prepared.headers);
        Ok((request, upload_monitor))
    }

    async fn send_draft(&self, request: RequestDraft) -> Result<HttpResponse, TransportError> {
        match &self.client {
            TransportClient::Fixed(client) => match &client.backend {
                HttpClientBackend::Routed(pool) => pool
                    .send(request)
                    .await
                    .map_err(Self::map_route_aware_error),
                HttpClientBackend::Direct(client) => {
                    let request = request
                        .build(client)
                        .map_err(|error| Self::map_error(error.into()))?;
                    client
                        .execute(request)
                        .await
                        .map(HttpResponse::from)
                        .map_err(|error| Self::map_error(error.into()))
                }
            },
            TransportClient::RouteAware(client) => client
                .send(request)
                .await
                .map_err(Self::map_route_aware_error),
        }
    }

    fn map_error(err: crate::HttpError) -> TransportError {
        let err = err.without_url();
        if let crate::HttpError::Policy(denied) = err {
            TransportError::Policy(denied)
        } else if err.is_connect() {
            TransportError::Connection(err.without_url())
        } else if err.is_timeout() {
            TransportError::Timeout
        } else {
            TransportError::Network(err.to_string())
        }
    }

    fn map_route_aware_error(error: RouteAwareRequestError) -> TransportError {
        match error.without_url() {
            RouteAwareRequestError::Policy(denied) => TransportError::Policy(denied),
            RouteAwareRequestError::Request(error) => Self::map_error(error.into()),
            RouteAwareRequestError::Timeout => TransportError::Timeout,
            RouteAwareRequestError::Route(error) => TransportError::Build(error.to_string()),
            RouteAwareRequestError::Build(error) => TransportError::Build(error),
            error @ (RouteAwareRequestError::UnsupportedRedirectScheme(_)
            | RouteAwareRequestError::TooManyRedirects) => {
                TransportError::Network(error.to_string())
            }
        }
    }

    fn trace_request(&self, req: &Request) {
        let request_logging_enabled = match &self.client {
            TransportClient::Fixed(client) => client.request_logging_enabled(),
            TransportClient::RouteAware(client) => client.request_logging_enabled(),
        };
        if request_logging_enabled && enabled!(Level::TRACE) {
            trace!(
                "{} to {}: {}",
                req.method,
                req.url,
                request_body_for_trace(req)
            );
        }
    }
}

fn request_body_for_trace(req: &Request) -> String {
    match req.body.as_ref() {
        Some(RequestBody::Json(body)) => body.to_string(),
        Some(RequestBody::EncodedJson(body)) => {
            String::from_utf8_lossy(body.trace_bytes()).into_owned()
        }
        Some(RequestBody::Raw(body)) => format!("<raw body: {} bytes>", body.len()),
        None => String::new(),
    }
}

impl HttpTransport for ReqwestTransport {
    async fn execute(&self, req: Request) -> Result<Response, TransportError> {
        self.trace_request(&req);

        let url = req.url.clone();
        let response_body_limit_bytes = req.response_body_limit_bytes;
        let resp = self.send(req).await?;
        let status = resp.status();
        let headers = resp.headers().clone();
        let retry_after = RetryAfter::from_headers(&headers);
        let bytes = match response_body_limit_bytes {
            Some(max_bytes) => bounded_response_bytes(resp, max_bytes).await,
            None => resp.bytes().await.map_err(Self::map_error),
        };
        if !status.is_success() {
            let body = match bytes {
                Ok(bytes) => String::from_utf8(bytes.to_vec()).ok(),
                Err(
                    error @ (TransportError::ResponseTooLarge { .. } | TransportError::Policy(_)),
                ) => return Err(error),
                // Keep bounded diagnostic-body failures from hiding HTTP auth/retry status.
                Err(_) if response_body_limit_bytes.is_some() => None,
                // TODO(anp): Preserve retry advice on body failures without changing their
                // network/timeout classification to HTTP status handling.
                Err(error) => return Err(error),
            };
            return Err(TransportError::Http {
                status,
                url: Some(url),
                headers: Some(headers),
                body,
                retry_after,
            });
        }
        Ok(Response {
            status,
            headers,
            body: bytes?,
        })
    }

    async fn stream(&self, req: Request) -> Result<StreamResponse, TransportError> {
        self.trace_request(&req);

        let url = req.url.clone();
        let response_body_limit_bytes = req.response_body_limit_bytes;
        let resp = self.send(req).await?;
        let status = resp.status();
        let headers = resp.headers().clone();
        if !status.is_success() {
            let retry_after = RetryAfter::from_headers(&headers);
            let body = match response_body_limit_bytes {
                Some(max_bytes) => match bounded_response_bytes(resp, max_bytes).await {
                    Ok(bytes) => {
                        // Reuse the unbounded path's charset, BOM and replacement decoding,
                        // but only after the body has passed the byte limit.
                        let mut buffered = http::Response::new(bytes);
                        *buffered.headers_mut() = headers.clone();
                        reqwest::Response::from(buffered).text().await.ok()
                    }
                    Err(
                        error @ (TransportError::ResponseTooLarge { .. }
                        | TransportError::Policy(_)),
                    ) => return Err(error),
                    // A failed diagnostic body must not hide HTTP auth or retry semantics.
                    Err(_) => None,
                },
                None => match resp.text().await {
                    Err(crate::HttpError::Policy(denied)) => return Err(denied.into()),
                    body => body.ok(),
                },
            };
            return Err(TransportError::Http {
                status,
                url: Some(url),
                headers: Some(headers),
                body,
                retry_after,
            });
        }
        let bytes = match response_body_limit_bytes {
            Some(max_bytes) => bounded_response_stream(resp, max_bytes)?,
            None => Box::pin(
                resp.bytes_stream()
                    .map(|result| result.map_err(Self::map_error)),
            ),
        };
        Ok(StreamResponse {
            status,
            headers,
            bytes,
        })
    }
}

fn bounded_response_stream(
    response: HttpResponse,
    max_bytes: usize,
) -> Result<ByteStream, TransportError> {
    if response
        .content_length()
        .is_some_and(|length| length > max_bytes as u64)
    {
        return Err(TransportError::ResponseTooLarge { max_bytes });
    }

    let stream = futures::stream::try_unfold(
        (response, max_bytes),
        move |(mut response, remaining)| async move {
            match response
                .chunk()
                .await
                .map_err(|error| ReqwestTransport::map_error(error.without_url()))?
            {
                Some(chunk) if chunk.len() <= remaining => {
                    let remaining = remaining - chunk.len();
                    Ok(Some((chunk, (response, remaining))))
                }
                Some(_) => Err(TransportError::ResponseTooLarge { max_bytes }),
                None => Ok(None),
            }
        },
    );
    Ok(Box::pin(stream))
}

async fn bounded_response_bytes(
    response: HttpResponse,
    max_bytes: usize,
) -> Result<Bytes, TransportError> {
    let mut stream = bounded_response_stream(response, max_bytes)?;
    let mut body = BytesMut::new();
    while let Some(chunk) = stream.next().await {
        body.extend_from_slice(&chunk?);
    }
    Ok(body.freeze())
}

#[cfg(test)]
#[path = "transport_tests.rs"]
mod tests;

#[cfg(test)]
#[path = "transport_limit_tests.rs"]
mod limit_tests;
