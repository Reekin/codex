use crate::client::HttpClient;
use crate::client::RequestBuilder;
use crate::error::TransportError;
use crate::request::Request;
use crate::request::RequestBody;
use crate::request::Response;
use crate::upload::UploadMonitor;
use crate::upload::UploadPolicy;
use crate::upload::monitored_body;
use bytes::Bytes;
use futures::StreamExt;
use futures::stream::BoxStream;
use http::HeaderMap;
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
    client: HttpClient,
    upload_policy: UploadPolicy,
}

impl ReqwestTransport {
    pub fn new(client: reqwest::Client) -> Self {
        Self {
            client: HttpClient::new(client),
            upload_policy: UploadPolicy::default(),
        }
    }

    pub fn from_http_client(client: HttpClient) -> Self {
        Self {
            client,
            upload_policy: UploadPolicy::default(),
        }
    }

    #[cfg(test)]
    fn with_upload_policy(mut self, upload_policy: UploadPolicy) -> Self {
        self.upload_policy = upload_policy;
        self
    }

    fn build(&self, req: Request) -> Result<BuiltRequest, TransportError> {
        let monitor_upload = matches!(req.body.as_ref(), Some(RequestBody::EncodedJson(_)))
            && !req.slow_upload_retry_claimed();
        let prepared = req.prepare_body_for_send().map_err(TransportError::Build)?;

        let Request {
            method,
            url,
            headers: _,
            body: _,
            compression: _,
            timeout,
            slow_upload_retry_claimed: _,
        } = req;

        let mut builder = self.client.request(
            Method::from_bytes(method.as_str().as_bytes()).unwrap_or(Method::GET),
            &url,
        );

        if let Some(timeout) = timeout {
            builder = builder.timeout(timeout);
        }

        builder = builder.headers(prepared.headers);
        let mut upload_monitor = None;
        if let Some(body) = prepared.body {
            if monitor_upload {
                let body_len = body.len();
                let (body, monitor) = monitored_body(body, self.upload_policy);
                if monitor.is_some() {
                    builder = builder.header(http::header::CONTENT_LENGTH, body_len);
                }
                builder = builder.body(body);
                upload_monitor = monitor;
            } else {
                builder = builder.body(body);
            }
        }
        Ok(BuiltRequest {
            builder,
            upload_monitor,
        })
    }

    fn map_error(err: reqwest::Error) -> TransportError {
        if err.is_connect() {
            TransportError::Connection(err.without_url())
        } else if err.is_timeout() {
            TransportError::Timeout
        } else {
            TransportError::Network(err.to_string())
        }
    }

    fn trace_request(&self, req: &Request) {
        if self.client.request_logging_enabled() && enabled!(Level::TRACE) {
            trace!(
                "{} to {}: {}",
                req.method,
                req.url,
                request_body_for_trace(req)
            );
        }
    }

    async fn send(&self, req: Request) -> Result<reqwest::Response, TransportError> {
        loop {
            let BuiltRequest {
                builder,
                upload_monitor,
            } = self.build(req.clone())?;
            let result = send_once(builder, upload_monitor).await;
            match result {
                Err(TransportError::SlowUpload {
                    total_bytes,
                    submitted_bytes,
                    elapsed,
                    bytes_per_second,
                    estimated_remaining,
                }) if req.claim_slow_upload_retry() => {
                    warn!(
                        total_bytes,
                        submitted_bytes,
                        elapsed_ms = elapsed.as_millis(),
                        bytes_per_second,
                        estimated_remaining_ms = estimated_remaining.as_millis(),
                        fresh_connection_retry = true,
                        "large request upload stalled; retrying after dropping the incomplete connection"
                    );
                }
                result => return result,
            }
        }
    }
}

struct BuiltRequest {
    builder: RequestBuilder,
    upload_monitor: Option<UploadMonitor>,
}

async fn send_once(
    builder: RequestBuilder,
    upload_monitor: Option<UploadMonitor>,
) -> Result<reqwest::Response, TransportError> {
    let send = builder.send();
    tokio::pin!(send);
    match upload_monitor {
        Some(upload_monitor) => {
            tokio::select! {
                result = &mut send => result.map_err(ReqwestTransport::map_error),
                error = upload_monitor.wait_for_stall() => Err(error),
            }
        }
        None => send.await.map_err(ReqwestTransport::map_error),
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
        let resp = self.send(req).await?;
        let status = resp.status();
        let headers = resp.headers().clone();
        let bytes = resp.bytes().await.map_err(Self::map_error)?;
        if !status.is_success() {
            let body = String::from_utf8(bytes.to_vec()).ok();
            return Err(TransportError::Http {
                status,
                url: Some(url),
                headers: Some(headers),
                body,
            });
        }
        Ok(Response {
            status,
            headers,
            body: bytes,
        })
    }

    async fn stream(&self, req: Request) -> Result<StreamResponse, TransportError> {
        self.trace_request(&req);

        let url = req.url.clone();
        let resp = self.send(req).await?;
        let status = resp.status();
        let headers = resp.headers().clone();
        if !status.is_success() {
            let body = resp.text().await.ok();
            return Err(TransportError::Http {
                status,
                url: Some(url),
                headers: Some(headers),
                body,
            });
        }
        let stream = resp
            .bytes_stream()
            .map(|result| result.map_err(Self::map_error));
        Ok(StreamResponse {
            status,
            headers,
            bytes: Box::pin(stream),
        })
    }
}

#[cfg(test)]
#[path = "transport_tests.rs"]
mod tests;
