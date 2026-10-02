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
use tokio::sync::oneshot;

#[test_case::test_case("primary"; "original finishes first")]
#[test_case::test_case("backup"; "backup finishes first")]
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn only_the_winning_request_executes_its_tool(winner: &str) -> anyhow::Result<()> {
    skip_if_no_network!(Ok(()));

    let mut gates = Vec::new();
    let mut streams = Vec::new();
    for id in ["primary", "backup"] {
        let (tx, rx) = oneshot::channel();
        gates.push(tx);
        streams.push(vec![
            StreamingSseChunk {
                gate: None,
                body: responses::sse(vec![responses::ev_response_created(id)]),
            },
            StreamingSseChunk {
                gate: Some(rx),
                body: responses::sse(vec![
                    responses::ev_function_call(
                        id,
                        "update_plan",
                        r#"{"plan":[{"step":"Check stream retry","status":"completed"}]}"#,
                    ),
                    responses::ev_completed(id),
                ]),
            },
        ]);
    }
    streams.push(vec![StreamingSseChunk {
        gate: None,
        body: responses::sse(vec![
            responses::ev_assistant_message("final", "done"),
            responses::ev_completed("final-response"),
        ]),
    }]);
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

    server.wait_for_request_count(2).await;
    gates
        .remove(usize::from(winner == "backup"))
        .send(())
        .unwrap();
    let EventMsg::TurnComplete(completed) = wait_for_event(&test.codex, |event| {
        matches!(event, EventMsg::TurnComplete(_))
    })
    .await
    else {
        unreachable!("predicate guarantees turn completion");
    };
    assert_eq!(completed.error, None);
    let requests = server.requests().await;
    assert_eq!(requests.len(), 3);
    let continuation: serde_json::Value = serde_json::from_slice(&requests[2])?;
    let calls: Vec<_> = continuation["input"]
        .as_array()
        .unwrap()
        .iter()
        .filter(|item| item["type"] == "function_call")
        .map(|item| item["call_id"].as_str().unwrap())
        .collect();
    assert_eq!(calls, vec![winner]);
    drop(gates);
    server.shutdown().await;
    Ok(())
}
