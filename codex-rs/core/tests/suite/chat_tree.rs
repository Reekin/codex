use anyhow::Result;
use codex_protocol::protocol::ChatTreeNodeStatus;
use core_test_support::responses;
use core_test_support::responses::ev_completed;
use core_test_support::responses::ev_response_created;
use core_test_support::responses::start_mock_server;
use core_test_support::test_codex::test_codex;
use pretty_assertions::assert_eq;

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
