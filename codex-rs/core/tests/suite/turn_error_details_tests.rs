//! Verify error hooks retain backend usage-limit details.

use codex_core::TurnInputRequest;
use codex_extension_api::ExtensionFuture;
use codex_extension_api::ExtensionRegistryBuilder;
use codex_extension_api::TurnErrorInput;
use codex_extension_api::TurnLifecycleContributor;
use codex_protocol::error::CodexErrorDetails;
use codex_protocol::protocol::CodexErrorInfo;
use codex_protocol::protocol::EventMsg;
use codex_protocol::protocol::RateLimitReachedType;
use codex_protocol::user_input::UserInput;
use core_test_support::responses;
use core_test_support::skip_if_no_network;
use core_test_support::test_codex::test_codex;
use core_test_support::wait_for_event;
use pretty_assertions::assert_eq;
use serde_json::json;
use std::sync::Arc;
use std::sync::Mutex;
use wiremock::ResponseTemplate;

#[derive(Debug, PartialEq)]
struct ObservedUsageLimit {
    turn_id: String,
    error: CodexErrorInfo,
    resets_at: Option<i64>,
    limit_window_minutes: Option<u16>,
    limit_id: Option<String>,
    snapshot_resets_at: Option<i64>,
    rate_limit_reached_type: Option<RateLimitReachedType>,
}

struct ErrorRecorder(Arc<Mutex<Vec<ObservedUsageLimit>>>);

impl TurnLifecycleContributor for ErrorRecorder {
    fn on_turn_error<'a>(&'a self, input: TurnErrorInput<'a>) -> ExtensionFuture<'a, ()> {
        Box::pin(async move {
            let CodexErrorDetails::UsageLimitReached(details) = input.error_details else {
                panic!("expected usage-limit details");
            };
            let snapshot = details.rate_limits.as_ref();
            self.0
                .lock()
                .expect("usage limit records lock")
                .push(ObservedUsageLimit {
                    turn_id: input.turn_id.to_string(),
                    error: input.error,
                    resets_at: details.resets_at.map(|reset| reset.timestamp()),
                    limit_window_minutes: details.limit_window_minutes,
                    limit_id: snapshot.and_then(|snapshot| snapshot.limit_id.clone()),
                    snapshot_resets_at: snapshot
                        .and_then(|snapshot| snapshot.primary.as_ref())
                        .and_then(|window| window.resets_at),
                    rate_limit_reached_type: details.rate_limit_reached_type,
                });
        })
    }
}

/// Backend reset details reach the turn-error hook.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn turn_error_details_preserve_usage_reset() -> anyhow::Result<()> {
    skip_if_no_network!(Ok(()));

    let server = responses::start_mock_server().await;
    let denial = ResponseTemplate::new(429)
        .insert_header("x-codex-active-limit", "test_limit")
        .insert_header("x-test-limit-primary-used-percent", "100")
        .insert_header("x-test-limit-primary-reset-at", "1700000000")
        .insert_header("x-codex-rate-limit-reached-type", "rate_limit_reached")
        .set_body_json(json!({
            "error": {
                "type": "usage_limit_reached",
                "resets_at": 1800000000,
                "limit_window_minutes": 10080
            }
        }));
    let mock = responses::mount_response_sequence(&server, vec![denial]).await;
    let observed = Arc::new(Mutex::new(Vec::new()));
    let mut extensions = ExtensionRegistryBuilder::new();
    extensions.turn_lifecycle_contributor(Arc::new(ErrorRecorder(observed.clone())));
    let mut builder = test_codex()
        .with_extensions(Arc::new(extensions.build()))
        .with_config(|config| {
            config.model_provider.request_max_retries = Some(0);
            config.model_provider.stream_max_retries = Some(0);
        });
    let test = builder.build_with_auto_env(&server).await?;
    test.codex
        .start_or_steer_turn(TurnInputRequest::user_input(vec![UserInput::Text {
            text: "Answer the request".to_string(),
            text_elements: Vec::new(),
        }]))
        .await?;

    let mut errors = Vec::new();
    let completed = wait_for_event(&test.codex, |event| {
        if let EventMsg::Error(error) = event {
            errors.push(error.codex_error_info.clone());
        }
        matches!(event, EventMsg::TurnComplete(_))
    })
    .await;
    let EventMsg::TurnComplete(completed) = completed else {
        unreachable!()
    };
    assert_eq!(
        *observed.lock().expect("usage limit records lock"),
        vec![ObservedUsageLimit {
            turn_id: completed.turn_id.clone(),
            error: CodexErrorInfo::UsageLimitExceeded,
            resets_at: Some(1_800_000_000),
            limit_window_minutes: Some(10_080),
            limit_id: Some("test_limit".to_string()),
            snapshot_resets_at: Some(1_700_000_000),
            rate_limit_reached_type: Some(RateLimitReachedType::RateLimitReached),
        }],
    );
    assert_eq!(
        completed.error.and_then(|error| error.codex_error_info),
        Some(CodexErrorInfo::UsageLimitExceeded)
    );
    assert_eq!(errors, vec![Some(CodexErrorInfo::UsageLimitExceeded)]);
    assert_eq!(mock.requests().len(), 1);
    test.codex.shutdown_and_wait().await?;
    Ok(())
}
