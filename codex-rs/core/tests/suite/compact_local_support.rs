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
    analysis: Analysis,
    decisions: Vec<Value>,
    delay: std::time::Duration,
}

#[derive(Clone, Default)]
pub(crate) struct LocalModel(Arc<Mutex<State>>);

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
                    if matches!(state.analysis, Analysis::HttpFailure) {
                        return wiremock::ResponseTemplate::new(503)
                            .set_body_json(json!({"error":{"message":"analysis unavailable","type":"server_error"}}));
                    }
                    let ids = payload["eligible_ids"].as_array().expect("eligible IDs");
                    assert!(!ids.is_empty(), "classifier needs eligible tool results");
                    let mut decisions = ids.iter().enumerate().map(|(index, id)| {
                        match (&state.analysis, index % 3) {
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
                    sse(vec![ev_assistant_message("classifier-private", &text), ev_completed_with_tokens("classifier-response", 17)])
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
                let delay = if analysis_payload(&body, CLASSIFY).is_some() || analysis_payload(&body, SUMMARIZE).is_some() { state.delay } else { std::time::Duration::ZERO };
                sse_response(response).set_delay(delay)
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

    pub(crate) fn delay(&self, delay: std::time::Duration) {
        self.0.lock().expect("model state").delay = delay;
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

    pub(crate) fn decisions(&self) -> Vec<Value> {
        self.0.lock().expect("model state").decisions.clone()
    }
}

pub(crate) fn configure(config: &mut Config) {
    let _ = config
        .features
        .disable(codex_features::Feature::EnableRequestCompression);
    config.local_compaction.force_local = true;
    config.local_compaction.minimum_savings_percent = 1;
    config.model_context_window = Some(100_000);
    config.model_auto_compact_token_limit = Some(95_000);
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
