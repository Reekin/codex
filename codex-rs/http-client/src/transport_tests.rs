use super::*;
use serde_json::json;
use std::io::Read;
use std::io::Write;
use std::net::SocketAddr;
use std::net::TcpListener;
use std::net::TcpStream;
use std::sync::Arc;
use std::sync::Mutex;
use std::time::Duration;
use std::time::Instant;
use tracing_subscriber::Layer;
use tracing_subscriber::layer::SubscriberExt;

#[tokio::test]
async fn stalled_large_upload_retries_on_a_fresh_connection() {
    let listener = bind_test_listener();
    let address = listener
        .local_addr()
        .expect("listener should have an address");
    let server = std::thread::spawn(move || {
        let first_content_length = stall_connection(&listener);

        let mut second = accept_with_deadline(&listener);
        let second_content_length = read_content_length(&mut second);
        let second_body_bytes = read_body(&mut second, second_content_length);
        write_ok(&mut second);
        assert_no_additional_connection(&listener);
        (first_content_length, second_body_bytes)
    });

    let response = upload_transport(test_upload_policy())
        .execute(prepared_json_request(address, 16 * 1024 * 1024))
        .await
        .expect("fresh connection retry should succeed");

    assert_eq!(response.body, Bytes::from_static(b"ok"));
    let (first_content_length, second_body_bytes) = server.join().expect("server should finish");
    assert!(first_content_length > 16 * 1024 * 1024);
    assert_eq!(second_body_bytes, first_content_length);
}

#[tokio::test]
async fn stalled_retries_are_monitored_until_the_budget_is_exhausted() {
    let listener = bind_test_listener();
    let address = listener
        .local_addr()
        .expect("listener should have an address");
    let server = std::thread::spawn(move || {
        // Each dropped connection fails the request unless the client already
        // canceled that attempt, so success proves every retry was monitored.
        let stalled_content_lengths: Vec<usize> =
            (0..3).map(|_| stall_connection(&listener)).collect();

        let mut last = accept_with_deadline(&listener);
        let last_content_length = read_content_length(&mut last);
        std::thread::sleep(Duration::from_millis(250));
        let last_body_bytes = read_body(&mut last, last_content_length);
        write_ok(&mut last);
        assert_no_additional_connection(&listener);
        (stalled_content_lengths, last_body_bytes)
    });

    let request = prepared_json_request(address, 16 * 1024 * 1024);
    let response = upload_transport(test_upload_policy())
        .execute(request.clone())
        .await
        .expect("the attempt after the retry budget should run to completion");

    assert_eq!(response.body, Bytes::from_static(b"ok"));
    let (stalled_content_lengths, last_body_bytes) = server.join().expect("server should finish");
    assert_eq!(stalled_content_lengths, vec![last_body_bytes; 3]);
    assert_eq!(request.slow_upload_retries(), 3);
}

#[tokio::test]
async fn progressing_large_upload_uses_one_connection() {
    let listener = bind_test_listener();
    let address = listener
        .local_addr()
        .expect("listener should have an address");
    let server = std::thread::spawn(move || {
        let mut stream = accept_with_deadline(&listener);
        let content_length = read_content_length(&mut stream);
        let body_bytes = read_body(&mut stream, content_length);
        write_ok(&mut stream);
        assert_no_additional_connection(&listener);
        body_bytes
    });

    let response = upload_transport(test_upload_policy())
        .execute(prepared_json_request(address, 2 * 1024 * 1024))
        .await
        .expect("progressing upload should succeed");

    assert_eq!(response.body, Bytes::from_static(b"ok"));
    assert!(server.join().expect("server should finish") > 2 * 1024 * 1024);
}

#[tokio::test]
async fn completed_upload_does_not_retry_while_waiting_for_headers() {
    let listener = bind_test_listener();
    let address = listener
        .local_addr()
        .expect("listener should have an address");
    let server = std::thread::spawn(move || {
        let mut stream = accept_with_deadline(&listener);
        let content_length = read_content_length(&mut stream);
        let body_bytes = read_body(&mut stream, content_length);
        std::thread::sleep(Duration::from_millis(250));
        write_ok(&mut stream);
        assert_no_additional_connection(&listener);
        body_bytes
    });

    let response = upload_transport(test_upload_policy())
        .execute(prepared_json_request(address, 2 * 1024 * 1024))
        .await
        .expect("response-header delay should not retry a completed upload");

    assert_eq!(response.body, Bytes::from_static(b"ok"));
    assert!(server.join().expect("server should finish") > 2 * 1024 * 1024);
}

#[tokio::test]
async fn small_upload_is_not_monitored() {
    let listener = bind_test_listener();
    let address = listener
        .local_addr()
        .expect("listener should have an address");
    let server = std::thread::spawn(move || {
        let mut stream = accept_with_deadline(&listener);
        let content_length = read_content_length(&mut stream);
        std::thread::sleep(Duration::from_millis(150));
        let body_bytes = read_body(&mut stream, content_length);
        write_ok(&mut stream);
        assert_no_additional_connection(&listener);
        body_bytes
    });

    let transport = upload_transport(test_upload_policy());
    let request = prepared_json_request(address, 512);
    assert!(
        transport
            .build(request.clone())
            .expect("small request should build")
            .upload_monitor
            .is_none()
    );
    let response = transport
        .execute(request)
        .await
        .expect("small upload should retain normal transport behavior");

    assert_eq!(response.body, Bytes::from_static(b"ok"));
    assert!(server.join().expect("server should finish") > 512);
}

#[tokio::test]
async fn enabled_request_logging_emits_transport_url_and_body() {
    let logs = capture_transport_logs(HttpClient::new(test_reqwest_client())).await;

    assert!(logs.contains("log capture sentinel"));
    assert!(logs.contains("url-secret"));
    assert!(logs.contains("body-secret"));
}

#[tokio::test]
async fn disabled_request_logging_suppresses_transport_url_and_body() {
    let logs = capture_transport_logs(HttpClient::new_without_request_logging(
        test_reqwest_client(),
    ))
    .await;

    assert!(logs.contains("log capture sentinel"));
    assert!(!logs.contains("url-secret"));
    assert!(!logs.contains("body-secret"));
}

#[tokio::test]
async fn connection_failures_are_classified_without_exposing_request_urls() {
    let unavailable_server =
        std::net::TcpListener::bind(("127.0.0.1", 0)).expect("server port should bind");
    let server_addr = unavailable_server
        .local_addr()
        .expect("server listener should have an address");
    drop(unavailable_server);
    let transport = ReqwestTransport::from_http_client(HttpClient::new(test_reqwest_client()));
    let request = Request::new(
        Method::POST,
        format!("http://{server_addr}/responses?token=url-secret"),
    );

    let error = match transport.stream(request).await {
        Err(TransportError::Connection(error)) => error,
        Err(error) => panic!("expected a connection failure, got {error}"),
        Ok(_) => panic!("an unavailable server should not return a response"),
    };
    assert!(!error.to_string().contains("url-secret"));
}

fn test_reqwest_client() -> reqwest::Client {
    reqwest::Client::builder()
        .no_proxy()
        .build()
        .expect("HTTP client should build")
}

fn upload_transport(policy: UploadPolicy) -> ReqwestTransport {
    ReqwestTransport::new(test_reqwest_client()).with_upload_policy(policy)
}

fn test_upload_policy() -> UploadPolicy {
    UploadPolicy {
        body_threshold: 1024,
        chunk_size: 16 * 1024,
        grace_period: Duration::from_millis(50),
        sample_period: Duration::from_millis(25),
        min_bytes_per_second: 1024,
        max_projected_remaining: Duration::from_millis(10),
        max_fresh_connection_retries: 3,
    }
}

fn prepared_json_request(address: SocketAddr, payload_bytes: usize) -> Request {
    let mut request = Request::new(Method::POST, format!("http://{address}/responses"))
        .with_json(&json!({"input": "x".repeat(payload_bytes)}))
        .into_prepared()
        .expect("request body should prepare");
    request.timeout = Some(Duration::from_secs(5));
    request
}

fn bind_test_listener() -> TcpListener {
    let listener = TcpListener::bind(("127.0.0.1", 0)).expect("test listener should bind");
    listener
        .set_nonblocking(true)
        .expect("test listener should become nonblocking");
    listener
}

fn accept_with_deadline(listener: &TcpListener) -> TcpStream {
    let deadline = Instant::now() + Duration::from_secs(5);
    loop {
        match listener.accept() {
            Ok((stream, _)) => {
                stream
                    .set_read_timeout(Some(Duration::from_secs(5)))
                    .expect("request stream should get a read timeout");
                return stream;
            }
            Err(error) if error.kind() == std::io::ErrorKind::WouldBlock => {
                assert!(
                    Instant::now() < deadline,
                    "server should receive the expected connection"
                );
                std::thread::sleep(Duration::from_millis(5));
            }
            Err(error) => panic!("server should accept connection: {error}"),
        }
    }
}

fn read_content_length(stream: &mut TcpStream) -> usize {
    let mut headers = Vec::new();
    let mut byte = [0_u8; 1];
    while !headers.ends_with(b"\r\n\r\n") {
        stream
            .read_exact(&mut byte)
            .expect("server should read request headers");
        headers.push(byte[0]);
    }
    let headers = String::from_utf8(headers).expect("request headers should be UTF-8");
    headers
        .lines()
        .find_map(|line| {
            let (name, value) = line.split_once(':')?;
            name.eq_ignore_ascii_case("content-length")
                .then(|| value.trim().parse::<usize>().ok())
                .flatten()
        })
        .expect("request should include content-length")
}

fn read_body(stream: &mut TcpStream, content_length: usize) -> usize {
    let mut remaining = content_length;
    let mut buffer = [0_u8; 64 * 1024];
    while remaining > 0 {
        let read_len = remaining.min(buffer.len());
        let bytes = stream
            .read(&mut buffer[..read_len])
            .expect("server should read request body");
        assert!(bytes > 0, "request body ended before content-length");
        remaining -= bytes;
    }
    content_length
}

/// Accepts a connection, stops reading its body long enough for the client
/// to detect a stall, then drops it. Returns the announced content length.
fn stall_connection(listener: &TcpListener) -> usize {
    let mut stream = accept_with_deadline(listener);
    let content_length = read_content_length(&mut stream);
    std::thread::sleep(Duration::from_millis(250));
    drop(stream);
    content_length
}

fn write_ok(stream: &mut TcpStream) {
    stream
        .write_all(b"HTTP/1.1 200 OK\r\nContent-Length: 2\r\nConnection: close\r\n\r\nok")
        .expect("server should write response");
}

fn assert_no_additional_connection(listener: &TcpListener) {
    let deadline = Instant::now() + Duration::from_millis(100);
    loop {
        match listener.accept() {
            Ok(_) => panic!("request unexpectedly opened another connection"),
            Err(error) if error.kind() == std::io::ErrorKind::WouldBlock => {
                if Instant::now() >= deadline {
                    return;
                }
                std::thread::sleep(Duration::from_millis(5));
            }
            Err(error) => panic!("server should inspect listener: {error}"),
        }
    }
}

async fn capture_transport_logs(client: HttpClient) -> String {
    let unavailable_server =
        std::net::TcpListener::bind(("127.0.0.1", 0)).expect("server port should bind");
    let server_addr = unavailable_server
        .local_addr()
        .expect("server listener should have an address");
    drop(unavailable_server);
    let transport = ReqwestTransport::from_http_client(client);
    let log_buffer = Arc::new(Mutex::new(Vec::new()));
    let writer_buffer = Arc::clone(&log_buffer);
    let subscriber = tracing_subscriber::registry().with(
        tracing_subscriber::fmt::layer()
            .with_ansi(false)
            .with_writer(move || TestLogWriter(Arc::clone(&writer_buffer)))
            .with_filter(
                tracing_subscriber::filter::Targets::new()
                    .with_target("codex_http_client::transport", tracing::Level::TRACE),
            ),
    );
    let _guard = tracing::subscriber::set_default(subscriber);
    tracing::trace!(target: "codex_http_client::transport", "log capture sentinel");
    let mut request = Request::new(
        Method::POST,
        format!("http://{server_addr}/request?token=url-secret"),
    )
    .with_json(&json!({"token": "body-secret"}));
    request.timeout = Some(Duration::from_secs(1));

    let _ = transport.execute(request).await;

    String::from_utf8(
        log_buffer
            .lock()
            .expect("log buffer should not be poisoned")
            .clone(),
    )
    .expect("captured logs should be UTF-8")
}

#[derive(Clone)]
struct TestLogWriter(Arc<Mutex<Vec<u8>>>);

impl Write for TestLogWriter {
    fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
        self.0
            .lock()
            .map_err(|_| std::io::Error::other("log buffer should not be poisoned"))?
            .extend_from_slice(buf);
        Ok(buf.len())
    }

    fn flush(&mut self) -> std::io::Result<()> {
        Ok(())
    }
}
