use codex_core::TurnInputRequest;
use codex_protocol::protocol::EventMsg;
use codex_protocol::user_input::UserInput;
use core_test_support::responses;
use core_test_support::skip_if_no_network;
use core_test_support::streaming_sse::StreamingSseChunk;
use core_test_support::streaming_sse::start_streaming_sse_server;
use core_test_support::test_codex::test_codex;
use core_test_support::wait_for_event;
use pretty_assertions::assert_eq;
use std::time::Duration;
use tokio::sync::oneshot;

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn retry_waits_longer_and_tool_continuation_resets_idle_timeout() -> anyhow::Result<()> {
    skip_if_no_network!(Ok(()));

    let tool_response = responses::sse(vec![
        responses::ev_function_call(
            "plan-call",
            "update_plan",
            r#"{"plan":[{"step":"Check stream retry","status":"completed"}]}"#,
        ),
        responses::ev_completed("tool-response"),
    ]);
    let final_response = responses::sse(vec![
        responses::ev_assistant_message("final", "done"),
        responses::ev_completed("final-response"),
    ]);
    let mut gates = Vec::new();
    let mut streams = Vec::new();
    for body in [String::new(), tool_response, String::new(), final_response] {
        let (tx, rx) = oneshot::channel();
        gates.push(tx);
        streams.push(vec![
            StreamingSseChunk {
                gate: None,
                body: responses::sse(vec![responses::ev_response_created("response")]),
            },
            StreamingSseChunk {
                gate: Some(rx),
                body,
            },
        ]);
    }
    let (server, _) = start_streaming_sse_server(streams).await;
    let test = test_codex()
        .with_config(|config| {
            config.model_provider.stream_idle_timeout_ms = Some(100);
            config.model_provider.stream_max_retries = Some(1);
            config.model_provider.request_max_retries = Some(0);
            config.model_provider.supports_websockets = false;
        })
        .build_with_streaming_server(&server)
        .await?;
    test.codex
        .start_or_steer_turn(TurnInputRequest::user_input(vec![UserInput::Text {
            text: "Update the plan and finish.".into(),
            text_elements: Vec::new(),
        }]))
        .await?;

    let mut gates = gates.into_iter();
    for expected_requests in [2, 4] {
        let EventMsg::StreamError(error) = wait_for_event(&test.codex, |event| {
            matches!(event, EventMsg::StreamError(_))
        })
        .await
        else {
            unreachable!("predicate guarantees a stream error");
        };
        assert_eq!(error.message, "Reconnecting... 1/1");
        // Close the timed-out stream, then hold the retry longer than the initial timeout.
        drop(gates.next().unwrap());
        server.wait_for_request_count(expected_requests).await;
        tokio::time::sleep(Duration::from_millis(500)).await;
        gates.next().unwrap().send(()).unwrap();
    }

    let EventMsg::TurnComplete(completed) = wait_for_event(&test.codex, |event| {
        matches!(event, EventMsg::TurnComplete(_))
    })
    .await
    else {
        unreachable!("predicate guarantees turn completion");
    };
    assert_eq!(completed.error, None);
    assert_eq!(server.requests().await.len(), 4);
    server.shutdown().await;
    Ok(())
}
