use anyhow::Result;
use codex_protocol::protocol::ChatTreeNodeStatus;
use codex_protocol::protocol::EventMsg;
use core_test_support::responses;
use core_test_support::responses::ev_assistant_message;
use core_test_support::responses::ev_completed;
use core_test_support::responses::ev_response_created;
use core_test_support::responses::start_mock_server;
use core_test_support::test_codex::test_codex;
use core_test_support::wait_for_event;
use pretty_assertions::assert_eq;
use std::time::Duration;
use tokio::time::sleep;
use tokio::time::timeout;
use wiremock::ResponseTemplate;

async fn wait_for_response_request_count(mock_response: &responses::ResponseMock, expected: usize) {
    if timeout(Duration::from_secs(5), async {
        loop {
            if mock_response.requests().len() == expected {
                return;
            }
            sleep(Duration::from_millis(10)).await;
        }
    })
    .await
    .is_err()
    {
        panic!("expected {expected} response requests");
    }
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn set_current_chat_tree_node_prunes_model_visible_history_for_next_turn() -> Result<()> {
    let server = start_mock_server().await;
    responses::mount_sse_once(
        &server,
        responses::sse(vec![ev_response_created("resp-a"), ev_completed("resp-a")]),
    )
    .await;
    responses::mount_sse_once(
        &server,
        responses::sse(vec![ev_response_created("resp-b"), ev_completed("resp-b")]),
    )
    .await;
    let branch_request = responses::mount_sse_once(
        &server,
        responses::sse(vec![ev_response_created("resp-d"), ev_completed("resp-d")]),
    )
    .await;

    let test = test_codex().with_model("gpt-5.4").build(&server).await?;
    test.submit_turn("chat-tree root A").await?;
    test.submit_turn("chat-tree child B").await?;

    let projection = test.codex.chat_tree_projection().await;
    assert_eq!(
        projection
            .nodes
            .iter()
            .map(|node| node.status)
            .collect::<Vec<_>>(),
        vec![ChatTreeNodeStatus::Completed, ChatTreeNodeStatus::Completed]
    );
    let root_node = projection
        .nodes
        .iter()
        .find(|node| node.parent_node_id.is_none())
        .expect("root chat tree node should exist");

    test.codex
        .set_current_chat_tree_node(&root_node.node_id, Some(projection.revision))
        .await
        .unwrap_or_else(|err| panic!("set current chat tree node: {err:?}"));
    test.submit_turn("chat-tree branch D").await?;

    let user_texts = branch_request.single_request().message_input_texts("user");
    assert!(
        user_texts.iter().any(|text| text == "chat-tree root A"),
        "next branch request should include the selected ancestor turn: {user_texts:?}"
    );
    assert!(
        user_texts.iter().any(|text| text == "chat-tree branch D"),
        "next branch request should include the new turn: {user_texts:?}"
    );
    assert!(
        user_texts.iter().all(|text| text != "chat-tree child B"),
        "next branch request should exclude the abandoned sibling branch: {user_texts:?}"
    );

    Ok(())
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn completed_chat_tree_node_gets_async_model_summary() -> Result<()> {
    let server = start_mock_server().await;
    responses::mount_sse_once(
        &server,
        responses::sse(vec![
            ev_response_created("resp-turn"),
            ev_assistant_message("msg-turn", "Implemented the chat tree summary producer."),
            ev_completed("resp-turn"),
        ]),
    )
    .await;
    let summary_response = responses::mount_sse_once(
        &server,
        responses::sse(vec![
            ev_response_created("resp-summary"),
            ev_assistant_message("msg-summary", "Add async chat tree summaries"),
            ev_completed("resp-summary"),
        ]),
    )
    .await;

    let test = test_codex().with_model("gpt-5.4").build(&server).await?;
    test.submit_turn("add chat tree summaries").await?;

    let summary_event = wait_for_event(&test.codex, |event| {
        matches!(event, EventMsg::ChatTreeNodeSummaryUpdated(_))
    })
    .await;
    let EventMsg::ChatTreeNodeSummaryUpdated(summary_event) = summary_event else {
        unreachable!("wait predicate ensures summary update event");
    };
    assert_eq!(
        summary_event.summary,
        Some("Add async chat tree summaries".to_string())
    );
    let summary_prompt = summary_response
        .single_request()
        .message_input_texts("user");
    assert_eq!(summary_prompt.len(), 1);
    assert!(summary_prompt[0].contains("User message:\nadd chat tree summaries"));
    assert!(
        summary_prompt[0]
            .contains("Assistant message:\nImplemented the chat tree summary producer.")
    );

    let projection = test.codex.chat_tree_projection().await;
    assert_eq!(
        projection
            .nodes
            .iter()
            .map(|node| node.summary.clone())
            .collect::<Vec<_>>(),
        vec![Some("Add async chat tree summaries".to_string())]
    );

    Ok(())
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn completed_chat_tree_node_without_assistant_output_does_not_request_summary() -> Result<()>
{
    let server = start_mock_server().await;
    responses::mount_sse_once(
        &server,
        responses::sse(vec![ev_response_created("resp-a"), ev_completed("resp-a")]),
    )
    .await;
    let second_turn = responses::mount_sse_once(
        &server,
        responses::sse(vec![ev_response_created("resp-b"), ev_completed("resp-b")]),
    )
    .await;

    let test = test_codex().with_model("gpt-5.4").build(&server).await?;
    test.submit_turn("first no assistant output").await?;
    test.submit_turn("second no assistant output").await?;

    assert!(
        second_turn
            .single_request()
            .message_input_texts("user")
            .iter()
            .any(|text| text == "second no assistant output"),
        "second turn fixture should not be consumed by a background summary request"
    );
    assert_eq!(
        test.codex
            .chat_tree_projection()
            .await
            .nodes
            .iter()
            .map(|node| node.summary.clone())
            .collect::<Vec<_>>(),
        vec![
            Some("Turn 1 · completed".to_string()),
            Some("Turn 2 · completed".to_string()),
        ]
    );

    Ok(())
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn chat_tree_summary_model_failure_keeps_default_summary() -> Result<()> {
    let server = start_mock_server().await;
    responses::mount_sse_once(
        &server,
        responses::sse(vec![
            ev_response_created("resp-turn"),
            ev_assistant_message("msg-turn", "Assistant output that should be summarized."),
            ev_completed("resp-turn"),
        ]),
    )
    .await;
    let summary_response =
        responses::mount_response_once(&server, ResponseTemplate::new(500)).await;

    let test = test_codex().with_model("gpt-5.4").build(&server).await?;
    test.submit_turn("summary model failure").await?;
    wait_for_response_request_count(&summary_response, 1).await;

    assert_eq!(
        test.codex
            .chat_tree_projection()
            .await
            .nodes
            .iter()
            .map(|node| node.summary.clone())
            .collect::<Vec<_>>(),
        vec![Some("Turn 1 · completed".to_string())]
    );

    Ok(())
}
