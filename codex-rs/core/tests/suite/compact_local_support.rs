//! Model-client fixtures for the structured local compaction protocol.

use anyhow::Result;
use codex_core::CodexThread;
use codex_core::config::Config;
use codex_history::RolloutItem;
use codex_protocol::protocol::EventMsg;
use core_test_support::responses::ev_assistant_message;
use core_test_support::responses::ev_completed;
use core_test_support::responses::ev_completed_with_tokens;
use core_test_support::responses::ev_function_call;
pub(crate) use core_test_support::responses::local_compaction_payload as analysis_payload;
use core_test_support::responses::sse;
use core_test_support::responses::sse_response;
use core_test_support::test_codex::TestCodex;
use serde_json::Value;
use serde_json::json;
use std::collections::VecDeque;
use std::path::Path;
use std::sync::Arc;
use std::sync::Mutex;
use wiremock::Mock;
use wiremock::MockServer;
use wiremock::Request;
use wiremock::matchers::method;
use wiremock::matchers::path;

pub(crate) const CLASSIFY: &str = "LOCAL_COMPACTION_CLASSIFY";
pub(crate) const SUMMARIZE: &str = "LOCAL_COMPACTION_SUMMARIZE";
pub(crate) const SHORTENED: &str = "verified concise evidence";
pub(crate) const LEDGER: &str = "Keep the user's offline-only constraint; verification is pending.";

#[derive(Clone, Default)]
pub(crate) enum Analysis {
    #[default]
    Clean,
    Keep,
    Malformed,
    ForeignId,
    DuplicateId,
    Oversized,
    InvalidSummary,
    HttpFailure,
}

#[derive(Default)]
struct State {
    requests: Vec<Request>,
    replies: VecDeque<String>,
    ordinary_reply_count: usize,
    marking_count: usize,
    analysis: Analysis,
    decisions: Vec<Value>,
}

#[derive(Clone, Default)]
pub(crate) struct LocalModel(Arc<Mutex<State>>);

/// HTTP proxy that holds classifier responses while forwarding ordinary requests immediately.
pub(crate) struct MarkGate {
    pub(crate) base_url: String,
    started: Arc<tokio::sync::Notify>,
    release: tokio::sync::watch::Sender<bool>,
    task: tokio::task::JoinHandle<()>,
}

impl MarkGate {
    pub(crate) async fn start(server: &MockServer) -> Result<Self> {
        use tokio::io::AsyncWriteExt;
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await?;
        let base_url = format!("http://{}/v1", listener.local_addr()?);
        let upstream = server.address().to_owned();
        let started = Arc::new(tokio::sync::Notify::new());
        let received = Arc::clone(&started);
        let (release, released) = tokio::sync::watch::channel(false);
        let task = tokio::spawn(async move {
            let mut connections = tokio::task::JoinSet::new();
            loop {
                let (mut client, _) = listener.accept().await.expect("gate connection");
                let started = Arc::clone(&received);
                let mut released = released.clone();
                connections.spawn(async move {
                    let (request, offset) = http_frame(&mut client).await;
                    let body: Value =
                        serde_json::from_slice(&request[offset..]).expect("request JSON");
                    let marking = analysis_payload(&body, CLASSIFY).is_some();
                    let mut target = tokio::net::TcpStream::connect(upstream)
                        .await
                        .expect("upstream");
                    target.write_all(&request).await.expect("forward request");
                    let (response, _) = http_frame(&mut target).await;
                    if marking {
                        started.notify_one();
                        while !*released.borrow_and_update() {
                            if released.changed().await.is_err() {
                                return;
                            }
                        }
                    }
                    // Hard compaction may have cancelled this client while its response was held.
                    let _ = client.write_all(&response).await;
                });
            }
        });
        Ok(Self {
            base_url,
            started,
            release,
            task,
        })
    }

    pub(crate) async fn wait(&self) -> Result<()> {
        tokio::time::timeout(std::time::Duration::from_secs(10), self.started.notified()).await?;
        Ok(())
    }

    pub(crate) fn release(&self) {
        self.release.send_replace(true);
    }

    pub(crate) fn hold(&self) {
        self.release.send_replace(false);
    }
}

impl Drop for MarkGate {
    fn drop(&mut self) {
        self.task.abort();
    }
}

async fn http_frame(stream: &mut tokio::net::TcpStream) -> (Vec<u8>, usize) {
    use tokio::io::AsyncReadExt;
    let mut bytes = Vec::new();
    loop {
        let mut chunk = [0; 8192];
        let size = stream.read(&mut chunk).await.expect("HTTP read");
        assert!(size > 0, "HTTP frame ended early");
        bytes.extend_from_slice(&chunk[..size]);
        if let Some(offset) = bytes.windows(4).position(|value| value == b"\r\n\r\n") {
            let offset = offset + 4;
            let headers = std::str::from_utf8(&bytes[..offset]).expect("HTTP headers");
            let length: usize = headers
                .lines()
                .find_map(|line| {
                    let (name, value) = line.split_once(':')?;
                    name.eq_ignore_ascii_case("content-length")
                        .then(|| value.trim().parse().unwrap())
                })
                .expect("fixture HTTP uses content-length");
            if bytes.len() >= offset + length {
                return (bytes, offset);
            }
        }
    }
}

impl LocalModel {
    pub(crate) async fn mount(server: &MockServer) -> Self {
        let model = Self::default();
        let responder = model.clone();
        Mock::given(method("POST"))
            .and(path("/v1/responses"))
            .respond_with(move |request: &Request| {
                let body: Value = serde_json::from_slice(&request.body).expect("model request JSON");
                let mut state = responder.0.lock().expect("model state");
                state.requests.push(request.clone());
                let response = if let Some(payload) = analysis_payload(&body, CLASSIFY) {
                    state.marking_count += 1;
                    if matches!(state.analysis, Analysis::HttpFailure) {
                        return wiremock::ResponseTemplate::new(503)
                            .set_body_json(json!({"error":{"message":"analysis unavailable","type":"server_error"}}));
                    }
                    let candidates = payload["candidates"].as_array().expect("candidates");
                    assert!(!candidates.is_empty(), "classifier needs eligible tool results");
                    let ids = candidates.iter().map(|candidate| candidate["id"].clone()).collect::<Vec<_>>();
                    // Fixture calls name their intended action; others rotate by position.
                    let mut decisions = candidates.iter().enumerate().map(|(index, candidate)| {
                        let id = &candidate["id"];
                        let call_id = candidate["call_id"].as_str().unwrap_or_default();
                        let slot = ["keep", "shorten", "drop"]
                            .iter()
                            .position(|action| call_id.contains(action))
                            .unwrap_or(index % 3);
                        match (&state.analysis, slot) {
                            (Analysis::Keep, _) | (_, 0) => json!({"id":id,"action":"keep"}),
                            (_, 1) => json!({"id":id,"action":"shorten","text":SHORTENED}),
                            _ => json!({"id":id,"action":"drop"}),
                        }
                    }).collect::<Vec<_>>();
                    match state.analysis {
                        Analysis::ForeignId => decisions[0]["id"] = json!("foreign-source-id"),
                        Analysis::DuplicateId => decisions.push(decisions[0].clone()),
                        Analysis::Oversized => decisions[0] = json!({"id":ids[0],"action":"shorten","text":"x".repeat(3001)}),
                        _ => {},
                    }
                    state.decisions.extend(decisions.clone());
                    let text = if matches!(state.analysis, Analysis::Malformed) {
                        "not valid JSON".to_string()
                    } else {
                        json!({"decisions":decisions}).to_string()
                    };
                    let response_id = if state.marking_count == 1 { "classifier-response".to_string() } else { format!("classifier-response-{}", state.marking_count) };
                    sse(vec![ev_assistant_message("classifier-private", &text), ev_completed_with_tokens(&response_id, 17)])
                } else if let Some(payload) = analysis_payload(&body, SUMMARIZE) {
                    let plan = &payload["plan"];
                    let text = if matches!(state.analysis, Analysis::InvalidSummary) {
                        "invalid tier analysis".to_string()
                    } else {
                        json!({
                            "l2": if plan["l2"].is_null() { "" } else { "Earlier dialogue and concise evidence; the earlier assumption was corrected, verification remains pending." },
                            "l3": if plan["l3"].is_null() { "" } else { "Oldest conversation overview; preserve the offline-only constraint and unresolved work." },
                            "ledger": LEDGER,
                        }).to_string()
                    };
                    sse(vec![ev_assistant_message("tiers-private", &text), ev_completed_with_tokens("tiers-response", 23)])
                } else {
                    state.replies.pop_front().expect("ordinary response queued before submitting turn")
                };
                sse_response(response)
            })
            .mount(server)
            .await;
        model
    }

    pub(crate) fn reply(&self, response: String) {
        self.0
            .lock()
            .expect("model state")
            .replies
            .push_back(response);
    }

    pub(crate) fn text(&self, text: &str) {
        let mut state = self.0.lock().expect("model state");
        state.ordinary_reply_count += 1;
        let reply_number = state.ordinary_reply_count;
        state.replies.push_back(sse(vec![
            ev_assistant_message(&format!("reply-{reply_number}"), text),
            ev_completed(&format!("ordinary-response-{reply_number}")),
        ]));
    }

    pub(crate) fn analysis(&self, analysis: Analysis) {
        self.0.lock().expect("model state").analysis = analysis;
    }

    pub(crate) fn requests(&self) -> Vec<Request> {
        self.0.lock().expect("model state").requests.clone()
    }

    pub(crate) fn bodies(&self) -> Vec<Value> {
        self.requests()
            .iter()
            .map(|request| serde_json::from_slice(&request.body).expect("request JSON"))
            .collect()
    }

    pub(crate) fn ordinary_bodies(&self) -> Vec<Value> {
        self.bodies()
            .into_iter()
            .filter(|body| {
                analysis_payload(body, CLASSIFY).is_none()
                    && analysis_payload(body, SUMMARIZE).is_none()
            })
            .collect()
    }

    pub(crate) fn decisions(&self) -> Vec<Value> {
        self.0.lock().expect("model state").decisions.clone()
    }
}

pub(crate) fn configure(config: &mut Config) {
    let _ = config
        .features
        .disable(codex_features::Feature::EnableRequestCompression);
    config.local_compaction.force_local = true;
    config.local_compaction.reclaim_percent = 1;
    // Manual/full-compaction fixtures isolate tier planning from background marking.
    config.local_compaction.mark_after_tokens_percent = 99;
    config.local_compaction.mark_after_records = u16::MAX;
    config.local_compaction.compact_target_percent = 15;
    config.model_context_window = Some(100_000);
    config.model_auto_compact_token_limit = Some(95_000);
}

pub(crate) fn configure_marking(config: &mut Config) {
    configure(config);
    config.local_compaction.mark_after_tokens_percent = 1;
    config.local_compaction.mark_after_records = 1;
}

/// Drive real sampling boundaries until background usage has been durably recorded.
pub(crate) async fn finish_marking(
    test: &TestCodex,
    model: &LocalModel,
    expected_batches: usize,
) -> Result<()> {
    for _ in 0..32 {
        model.text("Continue while tool evidence is assessed.");
        test.codex
            .start_or_steer_turn(codex_core::TurnInputRequest::user_input(vec![
                codex_protocol::user_input::UserInput::Text {
                    text: "Continue ordinary work.".to_string(),
                    text_elements: Vec::new(),
                },
            ]))
            .await?;
        complete(&test.codex).await;
        test.codex.flush_rollout().await?;
        if rollout(&test.codex.rollout_path().unwrap())?.iter().filter(|item| {
            matches!(item, RolloutItem::TokenUsageRecord(record) if record.response_id.starts_with("classifier-response"))
        }).count() >= expected_batches {
            // Installation happens at a sampling boundary after completion is consumed.
            model.text("Use the completed assessment.");
            test.submit_text_turn("Inspect the assessed evidence.").await?;
            return Ok(());
        }
    }
    anyhow::bail!("marking did not finish across 32 real sampling boundaries")
}

pub(crate) async fn complete(codex: &CodexThread) -> Vec<EventMsg> {
    let mut events = Vec::new();
    loop {
        let event = codex.next_event().await.expect("turn event").msg;
        assert!(
            !matches!(&event, EventMsg::Error(_)),
            "unexpected error: {event:?}"
        );
        let done = matches!(&event, EventMsg::TurnComplete(_));
        events.push(event);
        if done {
            return events;
        }
    }
}

pub(crate) async fn marking_ready(codex: &CodexThread) -> codex_protocol::protocol::TokenUsageInfo {
    core_test_support::wait_for_event_match(codex, |event| {
        assert!(
            !matches!(
                event,
                EventMsg::Error(_) | EventMsg::RawResponseCompleted(_)
            ),
            "background marking is private and nonfatal: {event:?}"
        );
        match event {
            EventMsg::TokenCount(count) => count.info.clone(),
            _ => None,
        }
    })
    .await
}

pub(crate) fn tool_turn() -> String {
    let mut events = ["keep", "shorten", "drop", "protected"]
        .into_iter()
        .map(|label| {
            ev_function_call(
                &format!("original-{label}"),
                &format!("{label}_{}", "evidence_".repeat(1000)),
                "{}",
            )
        })
        .collect::<Vec<_>>();
    events.push(ev_completed("tool-response"));
    sse(events)
}

pub(crate) fn rollout(path: &Path) -> Result<Vec<RolloutItem>> {
    std::fs::read_to_string(path)?
        .lines()
        .map(|line| Ok(codex_rollout::parse_rollout_line(line)?.item))
        .collect()
}

pub(crate) fn checkpoints(path: &Path) -> Result<Vec<Value>> {
    rollout(path)?
        .iter()
        .filter_map(|item| match item {
            RolloutItem::Compacted(compacted) => {
                Some(serde_json::to_value(compacted).map_err(Into::into))
            }
            _ => None,
        })
        .collect()
}
