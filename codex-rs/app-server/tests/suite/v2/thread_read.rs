use anyhow::Context;
use anyhow::Result;
use app_test_support::McpProcess;
use app_test_support::create_fake_rollout;
use app_test_support::create_fake_rollout_with_text_elements;
use app_test_support::create_mock_responses_server_repeating_assistant;
use app_test_support::rollout_path;
use app_test_support::test_absolute_path;
use app_test_support::to_response;
use codex_app_server_protocol::ChatTreeReadResponse;
use codex_app_server_protocol::ChatTreeSetCurrentResponse;
use codex_app_server_protocol::ChatTreeUpdatedNotification;
use codex_app_server_protocol::JSONRPCError;
use codex_app_server_protocol::JSONRPCNotification;
use codex_app_server_protocol::JSONRPCResponse;
use codex_app_server_protocol::RequestId;
use codex_app_server_protocol::SessionSource;
use codex_app_server_protocol::SortDirection;
use codex_app_server_protocol::ThreadForkParams;
use codex_app_server_protocol::ThreadForkResponse;
use codex_app_server_protocol::ThreadItem;
use codex_app_server_protocol::ThreadListParams;
use codex_app_server_protocol::ThreadListResponse;
use codex_app_server_protocol::ThreadNameUpdatedNotification;
use codex_app_server_protocol::ThreadReadParams;
use codex_app_server_protocol::ThreadReadResponse;
use codex_app_server_protocol::ThreadResumeParams;
use codex_app_server_protocol::ThreadResumeResponse;
use codex_app_server_protocol::ThreadSetNameParams;
use codex_app_server_protocol::ThreadSetNameResponse;
use codex_app_server_protocol::ThreadStartParams;
use codex_app_server_protocol::ThreadStartResponse;
use codex_app_server_protocol::ThreadStatus;
use codex_app_server_protocol::ThreadTurnsListParams;
use codex_app_server_protocol::ThreadTurnsListResponse;
use codex_app_server_protocol::TurnCompletedNotification;
use codex_app_server_protocol::TurnStartParams;
use codex_app_server_protocol::TurnStartResponse;
use codex_app_server_protocol::TurnStatus;
use codex_app_server_protocol::UserInput;
use codex_core::ARCHIVED_SESSIONS_SUBDIR;
use codex_protocol::config_types::ModeKind;
use codex_protocol::protocol::ChatTreeCurrentNodeChangedEvent;
use codex_protocol::protocol::ChatTreeNodeFinalizedEvent;
use codex_protocol::protocol::ChatTreeNodeStartedEvent;
use codex_protocol::protocol::ChatTreeNodeStatus;
use codex_protocol::protocol::EventMsg;
use codex_protocol::protocol::TurnCompleteEvent;
use codex_protocol::protocol::TurnStartedEvent;
use codex_protocol::protocol::UserMessageEvent;
use codex_protocol::user_input::ByteRange;
use codex_protocol::user_input::TextElement;
use core_test_support::responses;
use pretty_assertions::assert_eq;
use serde_json::Value;
use serde_json::json;
use std::io::Write;
use std::path::Path;
use tempfile::TempDir;
use tokio::time::timeout;

#[cfg(windows)]
const DEFAULT_READ_TIMEOUT: std::time::Duration = std::time::Duration::from_secs(25);
#[cfg(not(windows))]
const DEFAULT_READ_TIMEOUT: std::time::Duration = std::time::Duration::from_secs(10);

#[tokio::test]
async fn thread_read_returns_summary_without_turns() -> Result<()> {
    let server = create_mock_responses_server_repeating_assistant("Done").await;
    let codex_home = TempDir::new()?;
    create_config_toml(codex_home.path(), &server.uri())?;

    let preview = "Saved user message";
    let text_elements = [TextElement::new(
        ByteRange { start: 0, end: 5 },
        Some("<note>".into()),
    )];
    let conversation_id = create_fake_rollout_with_text_elements(
        codex_home.path(),
        "2025-01-05T12-00-00",
        "2025-01-05T12:00:00Z",
        preview,
        text_elements
            .iter()
            .map(|elem| serde_json::to_value(elem).expect("serialize text element"))
            .collect(),
        Some("mock_provider"),
        /*git_info*/ None,
    )?;

    let mut mcp = McpProcess::new(codex_home.path()).await?;
    timeout(DEFAULT_READ_TIMEOUT, mcp.initialize()).await??;

    let read_id = mcp
        .send_thread_read_request(ThreadReadParams {
            thread_id: conversation_id.clone(),
            include_turns: false,
        })
        .await?;
    let read_resp: JSONRPCResponse = timeout(
        DEFAULT_READ_TIMEOUT,
        mcp.read_stream_until_response_message(RequestId::Integer(read_id)),
    )
    .await??;
    let ThreadReadResponse { thread, .. } = to_response::<ThreadReadResponse>(read_resp)?;

    assert_eq!(thread.id, conversation_id);
    assert_eq!(thread.preview, preview);
    assert_eq!(thread.model_provider, "mock_provider");
    assert!(!thread.ephemeral, "stored rollouts should not be ephemeral");
    assert!(thread.path.as_ref().expect("thread path").is_absolute());
    assert_eq!(thread.cwd, test_absolute_path("/"));
    assert_eq!(thread.cli_version, "0.0.0");
    assert_eq!(thread.source, SessionSource::Cli);
    assert_eq!(thread.git_info, None);
    assert_eq!(thread.turns.len(), 0);
    assert_eq!(thread.status, ThreadStatus::NotLoaded);

    Ok(())
}

#[tokio::test]
async fn thread_read_can_include_turns() -> Result<()> {
    let server = create_mock_responses_server_repeating_assistant("Done").await;
    let codex_home = TempDir::new()?;
    create_config_toml(codex_home.path(), &server.uri())?;

    let preview = "Saved user message";
    let text_elements = vec![TextElement::new(
        ByteRange { start: 0, end: 5 },
        Some("<note>".into()),
    )];
    let conversation_id = create_fake_rollout_with_text_elements(
        codex_home.path(),
        "2025-01-05T12-00-00",
        "2025-01-05T12:00:00Z",
        preview,
        text_elements
            .iter()
            .map(|elem| serde_json::to_value(elem).expect("serialize text element"))
            .collect(),
        Some("mock_provider"),
        /*git_info*/ None,
    )?;

    let mut mcp = McpProcess::new(codex_home.path()).await?;
    timeout(DEFAULT_READ_TIMEOUT, mcp.initialize()).await??;

    let read_id = mcp
        .send_thread_read_request(ThreadReadParams {
            thread_id: conversation_id.clone(),
            include_turns: true,
        })
        .await?;
    let read_resp: JSONRPCResponse = timeout(
        DEFAULT_READ_TIMEOUT,
        mcp.read_stream_until_response_message(RequestId::Integer(read_id)),
    )
    .await??;
    let ThreadReadResponse { thread, .. } = to_response::<ThreadReadResponse>(read_resp)?;

    assert_eq!(thread.turns.len(), 1);
    let turn = &thread.turns[0];
    assert_eq!(turn.status, TurnStatus::Completed);
    assert_eq!(turn.items.len(), 1, "expected user message item");
    match &turn.items[0] {
        ThreadItem::UserMessage { content, .. } => {
            assert_eq!(
                content,
                &vec![UserInput::Text {
                    text: preview.to_string(),
                    text_elements: text_elements.clone().into_iter().map(Into::into).collect(),
                }]
            );
        }
        other => panic!("expected user message item, got {other:?}"),
    }
    assert_eq!(thread.status, ThreadStatus::NotLoaded);

    Ok(())
}

#[tokio::test]
async fn thread_read_and_turns_list_project_chat_tree_current_branch() -> Result<()> {
    let server = create_mock_responses_server_repeating_assistant("Done").await;
    let codex_home = TempDir::new()?;
    create_config_toml(codex_home.path(), &server.uri())?;

    let filename_ts = "2025-01-05T12-00-00";
    let conversation_id = create_fake_rollout(
        codex_home.path(),
        filename_ts,
        "2025-01-05T12:00:00Z",
        "legacy preview",
        Some("mock_provider"),
        /*git_info*/ None,
    )?;
    let rollout_path = rollout_path(codex_home.path(), filename_ts, &conversation_id);
    append_chat_tree_turn(
        &rollout_path,
        "2025-01-05T12:01:00Z",
        1,
        "turn-a",
        None,
        "A",
        0,
    )?;
    append_chat_tree_turn(
        &rollout_path,
        "2025-01-05T12:02:00Z",
        3,
        "turn-b",
        Some("turn-a"),
        "B",
        1,
    )?;
    append_chat_tree_turn(
        &rollout_path,
        "2025-01-05T12:03:00Z",
        5,
        "turn-c",
        Some("turn-b"),
        "C",
        2,
    )?;
    append_chat_tree_current_node_changed(&rollout_path, "2025-01-05T12:04:00Z", 7, "turn-a")?;
    append_chat_tree_turn(
        &rollout_path,
        "2025-01-05T12:05:00Z",
        8,
        "turn-d",
        Some("turn-a"),
        "D",
        3,
    )?;
    append_chat_tree_current_node_changed(&rollout_path, "2025-01-05T12:06:00Z", 10, "turn-b")?;

    let mut mcp = McpProcess::new(codex_home.path()).await?;
    timeout(DEFAULT_READ_TIMEOUT, mcp.initialize()).await??;

    let chat_tree_id = mcp
        .send_raw_request(
            "chatTree/read",
            Some(json!({ "threadId": conversation_id.clone() })),
        )
        .await?;
    let chat_tree_resp: JSONRPCResponse = timeout(
        DEFAULT_READ_TIMEOUT,
        mcp.read_stream_until_response_message(RequestId::Integer(chat_tree_id)),
    )
    .await??;
    let ChatTreeReadResponse { chat_tree, .. } =
        to_response::<ChatTreeReadResponse>(chat_tree_resp)?;
    assert_eq!(chat_tree.current_node_id, Some("turn-b".to_string()));
    assert_eq!(
        chat_tree.visible_node_ids,
        vec!["turn-a".to_string(), "turn-b".to_string()]
    );
    assert_eq!(
        chat_tree.visible_turn_ids,
        vec!["turn-a".to_string(), "turn-b".to_string()]
    );
    assert_eq!(chat_tree.nodes.len(), 4);

    let read_id = mcp
        .send_thread_read_request(ThreadReadParams {
            thread_id: conversation_id.clone(),
            include_turns: true,
        })
        .await?;
    let read_resp: JSONRPCResponse = timeout(
        DEFAULT_READ_TIMEOUT,
        mcp.read_stream_until_response_message(RequestId::Integer(read_id)),
    )
    .await??;
    let ThreadReadResponse { thread, .. } = to_response::<ThreadReadResponse>(read_resp)?;
    assert_eq!(turn_user_texts(&thread.turns), vec!["A", "B"]);

    let turns_id = mcp
        .send_thread_turns_list_request(ThreadTurnsListParams {
            thread_id: conversation_id,
            cursor: None,
            limit: Some(10),
            sort_direction: Some(SortDirection::Asc),
        })
        .await?;
    let turns_resp: JSONRPCResponse = timeout(
        DEFAULT_READ_TIMEOUT,
        mcp.read_stream_until_response_message(RequestId::Integer(turns_id)),
    )
    .await??;
    let ThreadTurnsListResponse { data, .. } = to_response::<ThreadTurnsListResponse>(turns_resp)?;
    assert_eq!(turn_user_texts(&data), vec!["A", "B"]);

    Ok(())
}

#[tokio::test]
async fn chat_tree_read_rejects_invalid_thread_id_with_stable_kind() -> Result<()> {
    let server = create_mock_responses_server_repeating_assistant("Done").await;
    let codex_home = TempDir::new()?;
    create_config_toml(codex_home.path(), &server.uri())?;

    let mut mcp = McpProcess::new(codex_home.path()).await?;
    timeout(DEFAULT_READ_TIMEOUT, mcp.initialize()).await??;

    let read_id = mcp
        .send_raw_request("chatTree/read", Some(json!({ "threadId": "not-a-uuid" })))
        .await?;
    let error = timeout(
        DEFAULT_READ_TIMEOUT,
        mcp.read_stream_until_error_message(RequestId::Integer(read_id)),
    )
    .await??;

    assert_eq!(error.error.code, -32600);
    assert!(error.error.message.contains("invalid thread id"));
    assert_eq!(
        error.error.data,
        Some(json!({
            "kind": "invalidThreadId",
            "threadId": "not-a-uuid",
        }))
    );

    Ok(())
}

#[tokio::test]
async fn chat_tree_set_current_updates_loaded_thread_and_notifies() -> Result<()> {
    let server = create_mock_responses_server_repeating_assistant("Done").await;
    let codex_home = TempDir::new()?;
    create_config_toml(codex_home.path(), &server.uri())?;

    let mut mcp = McpProcess::new(codex_home.path()).await?;
    timeout(DEFAULT_READ_TIMEOUT, mcp.initialize()).await??;

    let thread_id = start_loaded_thread(&mut mcp).await?;
    let turn_a = send_loaded_turn_and_wait(&mut mcp, &thread_id, "A").await?;
    let turn_b = send_loaded_turn_and_wait(&mut mcp, &thread_id, "B").await?;
    let before = read_chat_tree(&mut mcp, &thread_id).await?;
    assert_eq!(before.chat_tree.current_node_id, Some(turn_b));

    let set_id = mcp
        .send_raw_request(
            "chatTree/setCurrent",
            Some(json!({
                "threadId": thread_id,
                "nodeId": turn_a,
                "expectedRevision": before.chat_tree.revision,
            })),
        )
        .await?;
    let set_resp: JSONRPCResponse = timeout(
        DEFAULT_READ_TIMEOUT,
        mcp.read_stream_until_response_message(RequestId::Integer(set_id)),
    )
    .await??;
    let response = to_response::<ChatTreeSetCurrentResponse>(set_resp)?;
    assert_eq!(response.chat_tree.current_node_id, Some(turn_a.clone()));
    assert_eq!(response.chat_tree.visible_node_ids, vec![turn_a.clone()]);
    assert_eq!(response.chat_tree.visible_turn_ids, vec![turn_a.clone()]);

    let notification = wait_for_chat_tree_updated_current(&mut mcp, &thread_id, &turn_a).await?;
    assert_eq!(notification.chat_tree.current_node_id, Some(turn_a));

    Ok(())
}

#[tokio::test]
async fn chat_tree_set_current_reports_stable_error_kinds() -> Result<()> {
    let server = create_mock_responses_server_repeating_assistant("Done").await;
    let codex_home = TempDir::new()?;
    create_config_toml(codex_home.path(), &server.uri())?;

    let mut mcp = McpProcess::new(codex_home.path()).await?;
    timeout(DEFAULT_READ_TIMEOUT, mcp.initialize()).await??;

    let thread_id = start_loaded_thread(&mut mcp).await?;
    send_loaded_turn_and_wait(&mut mcp, &thread_id, "A").await?;

    let unknown_id = mcp
        .send_raw_request(
            "chatTree/setCurrent",
            Some(json!({
                "threadId": thread_id,
                "nodeId": "missing-node",
                "expectedRevision": null,
            })),
        )
        .await?;
    let error = timeout(
        DEFAULT_READ_TIMEOUT,
        mcp.read_stream_until_error_message(RequestId::Integer(unknown_id)),
    )
    .await??;
    assert_eq!(error.error.code, -32600);
    assert_eq!(
        error.error.data,
        Some(json!({
            "kind": "unknownNode",
            "threadId": thread_id,
            "nodeId": "missing-node",
        }))
    );

    let conflict_id = mcp
        .send_raw_request(
            "chatTree/setCurrent",
            Some(json!({
                "threadId": thread_id,
                "nodeId": "missing-node",
                "expectedRevision": 0,
            })),
        )
        .await?;
    let error = timeout(
        DEFAULT_READ_TIMEOUT,
        mcp.read_stream_until_error_message(RequestId::Integer(conflict_id)),
    )
    .await??;
    assert_eq!(error.error.code, -32600);
    assert_eq!(
        error.error.data,
        Some(json!({
            "kind": "revisionConflict",
            "threadId": thread_id,
            "actualRevision": 2,
            "expectedRevision": 0,
        }))
    );

    Ok(())
}

#[tokio::test]
async fn thread_turns_list_can_page_backward_and_forward() -> Result<()> {
    let server = create_mock_responses_server_repeating_assistant("Done").await;
    let codex_home = TempDir::new()?;
    create_config_toml(codex_home.path(), &server.uri())?;

    let filename_ts = "2025-01-05T12-00-00";
    let conversation_id = create_fake_rollout_with_text_elements(
        codex_home.path(),
        filename_ts,
        "2025-01-05T12:00:00Z",
        "first",
        vec![],
        Some("mock_provider"),
        /*git_info*/ None,
    )?;
    let rollout_path = rollout_path(codex_home.path(), filename_ts, &conversation_id);
    append_user_message(rollout_path.as_path(), "2025-01-05T12:01:00Z", "second")?;
    append_user_message(rollout_path.as_path(), "2025-01-05T12:02:00Z", "third")?;

    let mut mcp = McpProcess::new(codex_home.path()).await?;
    timeout(DEFAULT_READ_TIMEOUT, mcp.initialize()).await??;

    let read_id = mcp
        .send_thread_turns_list_request(ThreadTurnsListParams {
            thread_id: conversation_id.clone(),
            cursor: None,
            limit: Some(2),
            sort_direction: Some(SortDirection::Desc),
        })
        .await?;
    let read_resp: JSONRPCResponse = timeout(
        DEFAULT_READ_TIMEOUT,
        mcp.read_stream_until_response_message(RequestId::Integer(read_id)),
    )
    .await??;
    let ThreadTurnsListResponse {
        data,
        next_cursor,
        backwards_cursor,
    } = to_response::<ThreadTurnsListResponse>(read_resp)?;
    assert_eq!(turn_user_texts(&data), vec!["third", "second"]);
    let next_cursor = next_cursor.expect("expected nextCursor for older turns");
    let backwards_cursor = backwards_cursor.expect("expected backwardsCursor for newest turn");

    let read_id = mcp
        .send_thread_turns_list_request(ThreadTurnsListParams {
            thread_id: conversation_id.clone(),
            cursor: Some(next_cursor),
            limit: Some(10),
            sort_direction: Some(SortDirection::Desc),
        })
        .await?;
    let read_resp: JSONRPCResponse = timeout(
        DEFAULT_READ_TIMEOUT,
        mcp.read_stream_until_response_message(RequestId::Integer(read_id)),
    )
    .await??;
    let ThreadTurnsListResponse { data, .. } = to_response::<ThreadTurnsListResponse>(read_resp)?;
    assert_eq!(turn_user_texts(&data), vec!["first"]);

    append_user_message(rollout_path.as_path(), "2025-01-05T12:03:00Z", "fourth")?;

    let read_id = mcp
        .send_thread_turns_list_request(ThreadTurnsListParams {
            thread_id: conversation_id,
            cursor: Some(backwards_cursor),
            limit: Some(10),
            sort_direction: Some(SortDirection::Asc),
        })
        .await?;
    let read_resp: JSONRPCResponse = timeout(
        DEFAULT_READ_TIMEOUT,
        mcp.read_stream_until_response_message(RequestId::Integer(read_id)),
    )
    .await??;
    let ThreadTurnsListResponse { data, .. } = to_response::<ThreadTurnsListResponse>(read_resp)?;
    assert_eq!(turn_user_texts(&data), vec!["third", "fourth"]);

    Ok(())
}

#[tokio::test]
async fn thread_read_can_return_archived_threads_by_id() -> Result<()> {
    let server = create_mock_responses_server_repeating_assistant("Done").await;
    let codex_home = TempDir::new()?;
    create_config_toml(codex_home.path(), &server.uri())?;

    let filename_ts = "2025-01-05T12-00-00";
    let preview = "Archived saved user message";
    let conversation_id = create_fake_rollout_with_text_elements(
        codex_home.path(),
        filename_ts,
        "2025-01-05T12:00:00Z",
        preview,
        vec![],
        Some("mock_provider"),
        /*git_info*/ None,
    )?;
    let active_rollout_path = rollout_path(codex_home.path(), filename_ts, &conversation_id);
    let archived_dir = codex_home.path().join(ARCHIVED_SESSIONS_SUBDIR);
    std::fs::create_dir_all(&archived_dir)?;
    let archived_rollout_path =
        archived_dir.join(active_rollout_path.file_name().expect("rollout file name"));
    std::fs::rename(&active_rollout_path, &archived_rollout_path)?;

    let mut mcp = McpProcess::new(codex_home.path()).await?;
    timeout(DEFAULT_READ_TIMEOUT, mcp.initialize()).await??;

    let read_id = mcp
        .send_thread_read_request(ThreadReadParams {
            thread_id: conversation_id.clone(),
            include_turns: false,
        })
        .await?;
    let read_resp: JSONRPCResponse = timeout(
        DEFAULT_READ_TIMEOUT,
        mcp.read_stream_until_response_message(RequestId::Integer(read_id)),
    )
    .await??;
    let ThreadReadResponse { thread } = to_response::<ThreadReadResponse>(read_resp)?;

    assert_eq!(thread.id, conversation_id);
    assert_eq!(thread.preview, preview);
    let path = thread.path.expect("thread path");
    assert_eq!(path.canonicalize()?, archived_rollout_path.canonicalize()?);

    Ok(())
}

#[tokio::test]
async fn thread_turns_list_rejects_cursor_when_anchor_turn_is_rolled_back() -> Result<()> {
    let server = create_mock_responses_server_repeating_assistant("Done").await;
    let codex_home = TempDir::new()?;
    create_config_toml(codex_home.path(), &server.uri())?;

    let filename_ts = "2025-01-05T12-00-00";
    let conversation_id = create_fake_rollout_with_text_elements(
        codex_home.path(),
        filename_ts,
        "2025-01-05T12:00:00Z",
        "first",
        vec![],
        Some("mock_provider"),
        /*git_info*/ None,
    )?;
    let rollout_path = rollout_path(codex_home.path(), filename_ts, &conversation_id);
    append_user_message(rollout_path.as_path(), "2025-01-05T12:01:00Z", "second")?;
    append_user_message(rollout_path.as_path(), "2025-01-05T12:02:00Z", "third")?;

    let mut mcp = McpProcess::new(codex_home.path()).await?;
    timeout(DEFAULT_READ_TIMEOUT, mcp.initialize()).await??;

    let read_id = mcp
        .send_thread_turns_list_request(ThreadTurnsListParams {
            thread_id: conversation_id.clone(),
            cursor: None,
            limit: Some(2),
            sort_direction: Some(SortDirection::Desc),
        })
        .await?;
    let read_resp: JSONRPCResponse = timeout(
        DEFAULT_READ_TIMEOUT,
        mcp.read_stream_until_response_message(RequestId::Integer(read_id)),
    )
    .await??;
    let ThreadTurnsListResponse {
        backwards_cursor, ..
    } = to_response::<ThreadTurnsListResponse>(read_resp)?;
    let backwards_cursor = backwards_cursor.expect("expected backwardsCursor for newest turn");

    append_thread_rollback(
        rollout_path.as_path(),
        "2025-01-05T12:03:00Z",
        /*num_turns*/ 1,
    )?;

    let read_id = mcp
        .send_thread_turns_list_request(ThreadTurnsListParams {
            thread_id: conversation_id,
            cursor: Some(backwards_cursor),
            limit: Some(10),
            sort_direction: Some(SortDirection::Asc),
        })
        .await?;
    let read_err: JSONRPCError = timeout(
        DEFAULT_READ_TIMEOUT,
        mcp.read_stream_until_error_message(RequestId::Integer(read_id)),
    )
    .await??;

    assert_eq!(
        read_err.error.message,
        "invalid cursor: anchor turn is no longer present"
    );

    Ok(())
}

#[tokio::test]
async fn thread_read_returns_forked_from_id_for_forked_threads() -> Result<()> {
    let server = create_mock_responses_server_repeating_assistant("Done").await;
    let codex_home = TempDir::new()?;
    create_config_toml(codex_home.path(), &server.uri())?;

    let conversation_id = create_fake_rollout_with_text_elements(
        codex_home.path(),
        "2025-01-05T12-00-00",
        "2025-01-05T12:00:00Z",
        "Saved user message",
        vec![],
        Some("mock_provider"),
        /*git_info*/ None,
    )?;

    let mut mcp = McpProcess::new(codex_home.path()).await?;
    timeout(DEFAULT_READ_TIMEOUT, mcp.initialize()).await??;

    let fork_id = mcp
        .send_thread_fork_request(ThreadForkParams {
            thread_id: conversation_id.clone(),
            ..Default::default()
        })
        .await?;
    let fork_resp: JSONRPCResponse = timeout(
        DEFAULT_READ_TIMEOUT,
        mcp.read_stream_until_response_message(RequestId::Integer(fork_id)),
    )
    .await??;
    let ThreadForkResponse { thread: forked, .. } = to_response::<ThreadForkResponse>(fork_resp)?;

    let read_id = mcp
        .send_thread_read_request(ThreadReadParams {
            thread_id: forked.id,
            include_turns: false,
        })
        .await?;
    let read_resp: JSONRPCResponse = timeout(
        DEFAULT_READ_TIMEOUT,
        mcp.read_stream_until_response_message(RequestId::Integer(read_id)),
    )
    .await??;
    let ThreadReadResponse { thread, .. } = to_response::<ThreadReadResponse>(read_resp)?;

    assert_eq!(thread.forked_from_id, Some(conversation_id));

    Ok(())
}

#[tokio::test]
async fn thread_read_loaded_thread_returns_precomputed_path_before_materialization() -> Result<()> {
    let server = create_mock_responses_server_repeating_assistant("Done").await;
    let codex_home = TempDir::new()?;
    create_config_toml(codex_home.path(), &server.uri())?;

    let mut mcp = McpProcess::new(codex_home.path()).await?;
    timeout(DEFAULT_READ_TIMEOUT, mcp.initialize()).await??;

    let start_id = mcp
        .send_thread_start_request(ThreadStartParams {
            model: Some("mock-model".to_string()),
            ..Default::default()
        })
        .await?;
    let start_resp: JSONRPCResponse = timeout(
        DEFAULT_READ_TIMEOUT,
        mcp.read_stream_until_response_message(RequestId::Integer(start_id)),
    )
    .await??;
    let ThreadStartResponse { thread, .. } = to_response::<ThreadStartResponse>(start_resp)?;
    let thread_path = thread.path.clone().expect("thread path");
    assert!(
        !thread_path.exists(),
        "fresh thread rollout should not be materialized yet"
    );

    let read_id = mcp
        .send_thread_read_request(ThreadReadParams {
            thread_id: thread.id.clone(),
            include_turns: false,
        })
        .await?;
    let read_resp: JSONRPCResponse = timeout(
        DEFAULT_READ_TIMEOUT,
        mcp.read_stream_until_response_message(RequestId::Integer(read_id)),
    )
    .await??;
    let ThreadReadResponse { thread: read, .. } = to_response::<ThreadReadResponse>(read_resp)?;

    assert_eq!(read.id, thread.id);
    assert_eq!(read.path, Some(thread_path));
    assert!(read.preview.is_empty());
    assert_eq!(read.turns.len(), 0);
    assert_eq!(read.status, ThreadStatus::Idle);

    Ok(())
}

#[tokio::test]
async fn thread_name_set_is_reflected_in_read_list_and_resume() -> Result<()> {
    let server = create_mock_responses_server_repeating_assistant("Done").await;
    let codex_home = TempDir::new()?;
    create_config_toml(codex_home.path(), &server.uri())?;

    let preview = "Saved user message";
    let conversation_id = create_fake_rollout_with_text_elements(
        codex_home.path(),
        "2025-01-05T12-00-00",
        "2025-01-05T12:00:00Z",
        preview,
        vec![],
        Some("mock_provider"),
        /*git_info*/ None,
    )?;

    let mut mcp = McpProcess::new(codex_home.path()).await?;
    timeout(DEFAULT_READ_TIMEOUT, mcp.initialize()).await??;

    // Set a user-facing thread title.
    let new_name = "My renamed thread";
    let set_id = mcp
        .send_thread_set_name_request(ThreadSetNameParams {
            thread_id: conversation_id.clone(),
            name: new_name.to_string(),
        })
        .await?;
    let set_resp: JSONRPCResponse = timeout(
        DEFAULT_READ_TIMEOUT,
        mcp.read_stream_until_response_message(RequestId::Integer(set_id)),
    )
    .await??;
    let _: ThreadSetNameResponse = to_response::<ThreadSetNameResponse>(set_resp)?;
    let notification = timeout(
        DEFAULT_READ_TIMEOUT,
        mcp.read_stream_until_notification_message("thread/name/updated"),
    )
    .await??;
    let notification: ThreadNameUpdatedNotification =
        serde_json::from_value(notification.params.expect("thread/name/updated params"))?;
    assert_eq!(notification.thread_id, conversation_id);
    assert_eq!(notification.thread_name.as_deref(), Some(new_name));

    // Read should now surface `thread.name`, and the wire payload must include `name`.
    let read_id = mcp
        .send_thread_read_request(ThreadReadParams {
            thread_id: conversation_id.clone(),
            include_turns: false,
        })
        .await?;
    let read_resp: JSONRPCResponse = timeout(
        DEFAULT_READ_TIMEOUT,
        mcp.read_stream_until_response_message(RequestId::Integer(read_id)),
    )
    .await??;
    let read_result = read_resp.result.clone();
    let ThreadReadResponse { thread, .. } = to_response::<ThreadReadResponse>(read_resp)?;
    assert_eq!(thread.id, conversation_id);
    assert_eq!(thread.name.as_deref(), Some(new_name));
    let thread_json = read_result
        .get("thread")
        .and_then(Value::as_object)
        .expect("thread/read result.thread must be an object");
    assert_eq!(
        thread_json.get("name").and_then(Value::as_str),
        Some(new_name),
        "thread/read must serialize `thread.name` on the wire"
    );
    assert_eq!(
        thread_json.get("ephemeral").and_then(Value::as_bool),
        Some(false),
        "thread/read must serialize `thread.ephemeral` on the wire"
    );

    // List should also surface the name.
    let list_id = mcp
        .send_thread_list_request(ThreadListParams {
            cursor: None,
            limit: Some(50),
            sort_key: None,
            sort_direction: None,
            model_providers: Some(vec!["mock_provider".to_string()]),
            source_kinds: None,
            archived: None,
            cwd: None,
            use_state_db_only: false,
            search_term: None,
        })
        .await?;
    let list_resp: JSONRPCResponse = timeout(
        DEFAULT_READ_TIMEOUT,
        mcp.read_stream_until_response_message(RequestId::Integer(list_id)),
    )
    .await??;
    let list_result = list_resp.result.clone();
    let ThreadListResponse { data, .. } = to_response::<ThreadListResponse>(list_resp)?;
    let listed = data
        .iter()
        .find(|t| t.id == conversation_id)
        .expect("thread/list should include the created thread");
    assert_eq!(listed.name.as_deref(), Some(new_name));
    let listed_json = list_result
        .get("data")
        .and_then(Value::as_array)
        .expect("thread/list result.data must be an array")
        .iter()
        .find(|t| t.get("id").and_then(Value::as_str) == Some(&conversation_id))
        .and_then(Value::as_object)
        .expect("thread/list should include the created thread as an object");
    assert_eq!(
        listed_json.get("name").and_then(Value::as_str),
        Some(new_name),
        "thread/list must serialize `thread.name` on the wire"
    );
    assert_eq!(
        listed_json.get("ephemeral").and_then(Value::as_bool),
        Some(false),
        "thread/list must serialize `thread.ephemeral` on the wire"
    );

    // Resume should also surface the name.
    let resume_id = mcp
        .send_thread_resume_request(ThreadResumeParams {
            thread_id: conversation_id.clone(),
            ..Default::default()
        })
        .await?;
    let resume_resp: JSONRPCResponse = timeout(
        DEFAULT_READ_TIMEOUT,
        mcp.read_stream_until_response_message(RequestId::Integer(resume_id)),
    )
    .await??;
    let resume_result = resume_resp.result.clone();
    let ThreadResumeResponse {
        thread: resumed, ..
    } = to_response::<ThreadResumeResponse>(resume_resp)?;
    assert_eq!(resumed.id, conversation_id);
    assert_eq!(resumed.name.as_deref(), Some(new_name));
    let resumed_json = resume_result
        .get("thread")
        .and_then(Value::as_object)
        .expect("thread/resume result.thread must be an object");
    assert_eq!(
        resumed_json.get("name").and_then(Value::as_str),
        Some(new_name),
        "thread/resume must serialize `thread.name` on the wire"
    );
    assert_eq!(
        resumed_json.get("ephemeral").and_then(Value::as_bool),
        Some(false),
        "thread/resume must serialize `thread.ephemeral` on the wire"
    );

    Ok(())
}

#[tokio::test]
async fn thread_read_include_turns_rejects_unmaterialized_loaded_thread() -> Result<()> {
    let server = create_mock_responses_server_repeating_assistant("Done").await;
    let codex_home = TempDir::new()?;
    create_config_toml(codex_home.path(), &server.uri())?;

    let mut mcp = McpProcess::new(codex_home.path()).await?;
    timeout(DEFAULT_READ_TIMEOUT, mcp.initialize()).await??;

    let start_id = mcp
        .send_thread_start_request(ThreadStartParams {
            model: Some("mock-model".to_string()),
            ..Default::default()
        })
        .await?;
    let start_resp: JSONRPCResponse = timeout(
        DEFAULT_READ_TIMEOUT,
        mcp.read_stream_until_response_message(RequestId::Integer(start_id)),
    )
    .await??;
    let ThreadStartResponse { thread, .. } = to_response::<ThreadStartResponse>(start_resp)?;
    let thread_path = thread.path.clone().expect("thread path");
    assert!(
        !thread_path.exists(),
        "fresh thread rollout should not be materialized yet"
    );

    let read_id = mcp
        .send_thread_read_request(ThreadReadParams {
            thread_id: thread.id.clone(),
            include_turns: true,
        })
        .await?;
    let read_err: JSONRPCError = timeout(
        DEFAULT_READ_TIMEOUT,
        mcp.read_stream_until_error_message(RequestId::Integer(read_id)),
    )
    .await??;

    assert!(
        read_err
            .error
            .message
            .contains("includeTurns is unavailable before first user message"),
        "unexpected error: {}",
        read_err.error.message
    );

    Ok(())
}

#[tokio::test]
async fn thread_read_reports_system_error_idle_flag_after_failed_turn() -> Result<()> {
    let server = responses::start_mock_server().await;
    let _response_mock = responses::mount_sse_once(
        &server,
        responses::sse_failed("resp-1", "server_error", "simulated failure"),
    )
    .await;
    let codex_home = TempDir::new()?;
    create_config_toml(codex_home.path(), &server.uri())?;

    let mut mcp = McpProcess::new(codex_home.path()).await?;
    timeout(DEFAULT_READ_TIMEOUT, mcp.initialize()).await??;

    let start_id = mcp
        .send_thread_start_request(ThreadStartParams {
            model: Some("mock-model".to_string()),
            ..Default::default()
        })
        .await?;
    let start_resp: JSONRPCResponse = timeout(
        DEFAULT_READ_TIMEOUT,
        mcp.read_stream_until_response_message(RequestId::Integer(start_id)),
    )
    .await??;
    let ThreadStartResponse { thread, .. } = to_response::<ThreadStartResponse>(start_resp)?;

    let turn_start_id = mcp
        .send_turn_start_request(TurnStartParams {
            thread_id: thread.id.clone(),
            input: vec![UserInput::Text {
                text: "fail this turn".to_string(),
                text_elements: Vec::new(),
            }],
            ..Default::default()
        })
        .await?;
    let turn_start_response: JSONRPCResponse = timeout(
        DEFAULT_READ_TIMEOUT,
        mcp.read_stream_until_response_message(RequestId::Integer(turn_start_id)),
    )
    .await??;
    let _: TurnStartResponse = to_response::<TurnStartResponse>(turn_start_response)?;
    timeout(
        DEFAULT_READ_TIMEOUT,
        mcp.read_stream_until_notification_message("error"),
    )
    .await??;

    let read_id = mcp
        .send_thread_read_request(ThreadReadParams {
            thread_id: thread.id,
            include_turns: false,
        })
        .await?;
    let read_resp: JSONRPCResponse = timeout(
        DEFAULT_READ_TIMEOUT,
        mcp.read_stream_until_response_message(RequestId::Integer(read_id)),
    )
    .await??;
    let ThreadReadResponse { thread, .. } = to_response::<ThreadReadResponse>(read_resp)?;

    assert_eq!(thread.status, ThreadStatus::SystemError,);

    Ok(())
}

async fn start_loaded_thread(mcp: &mut McpProcess) -> Result<String> {
    let start_id = mcp
        .send_thread_start_request(ThreadStartParams {
            model: Some("mock-model".to_string()),
            ..Default::default()
        })
        .await?;
    let start_resp: JSONRPCResponse = timeout(
        DEFAULT_READ_TIMEOUT,
        mcp.read_stream_until_response_message(RequestId::Integer(start_id)),
    )
    .await??;
    let ThreadStartResponse { thread, .. } = to_response::<ThreadStartResponse>(start_resp)?;
    Ok(thread.id)
}

async fn send_loaded_turn_and_wait(
    mcp: &mut McpProcess,
    thread_id: &str,
    text: &str,
) -> Result<String> {
    let turn_start_id = mcp
        .send_turn_start_request(TurnStartParams {
            thread_id: thread_id.to_string(),
            input: vec![UserInput::Text {
                text: text.to_string(),
                text_elements: Vec::new(),
            }],
            ..Default::default()
        })
        .await?;
    let turn_start_response: JSONRPCResponse = timeout(
        DEFAULT_READ_TIMEOUT,
        mcp.read_stream_until_response_message(RequestId::Integer(turn_start_id)),
    )
    .await??;
    let TurnStartResponse { turn } = to_response::<TurnStartResponse>(turn_start_response)?;
    loop {
        let notification: JSONRPCNotification = timeout(
            DEFAULT_READ_TIMEOUT,
            mcp.read_stream_until_notification_message("turn/completed"),
        )
        .await??;
        let completed: TurnCompletedNotification =
            serde_json::from_value(notification.params.context("turn/completed params")?)?;
        if completed.turn.id == turn.id {
            return Ok(turn.id);
        }
    }
}

async fn read_chat_tree(mcp: &mut McpProcess, thread_id: &str) -> Result<ChatTreeReadResponse> {
    let read_id = mcp
        .send_raw_request("chatTree/read", Some(json!({ "threadId": thread_id })))
        .await?;
    let read_resp: JSONRPCResponse = timeout(
        DEFAULT_READ_TIMEOUT,
        mcp.read_stream_until_response_message(RequestId::Integer(read_id)),
    )
    .await??;
    to_response::<ChatTreeReadResponse>(read_resp)
}

async fn wait_for_chat_tree_updated_current(
    mcp: &mut McpProcess,
    thread_id: &str,
    node_id: &str,
) -> Result<ChatTreeUpdatedNotification> {
    loop {
        let notification: JSONRPCNotification = timeout(
            DEFAULT_READ_TIMEOUT,
            mcp.read_stream_until_notification_message("chatTree/updated"),
        )
        .await??;
        let updated: ChatTreeUpdatedNotification =
            serde_json::from_value(notification.params.context("chatTree/updated params")?)?;
        if updated.thread_id == thread_id
            && updated.chat_tree.current_node_id.as_deref() == Some(node_id)
        {
            return Ok(updated);
        }
    }
}

fn append_user_message(path: &Path, timestamp: &str, text: &str) -> std::io::Result<()> {
    let mut file = std::fs::OpenOptions::new().append(true).open(path)?;
    writeln!(
        file,
        "{}",
        json!({
            "timestamp": timestamp,
            "type":"event_msg",
            "payload": {
                "type":"user_message",
                "message": text,
                "text_elements": [],
                "local_images": []
            }
        })
    )
}

fn append_chat_tree_turn(
    path: &Path,
    timestamp: &str,
    start_revision: u64,
    turn_id: &str,
    parent_node_id: Option<&str>,
    text: &str,
    order: u64,
) -> std::io::Result<()> {
    append_event(
        path,
        timestamp,
        EventMsg::TurnStarted(TurnStartedEvent {
            turn_id: turn_id.to_string(),
            started_at: None,
            model_context_window: Some(128_000),
            collaboration_mode_kind: ModeKind::Default,
        }),
    )?;
    append_event(
        path,
        timestamp,
        EventMsg::ChatTreeNodeStarted(Box::new(ChatTreeNodeStartedEvent {
            revision: start_revision,
            node_id: turn_id.to_string(),
            parent_node_id: parent_node_id.map(str::to_string),
            turn_id: Some(turn_id.to_string()),
            order,
        })),
    )?;
    append_event(
        path,
        timestamp,
        EventMsg::UserMessage(UserMessageEvent {
            message: text.to_string(),
            images: None,
            local_images: Vec::new(),
            text_elements: Vec::new(),
        }),
    )?;
    append_event(
        path,
        timestamp,
        EventMsg::ChatTreeNodeFinalized(Box::new(ChatTreeNodeFinalizedEvent {
            revision: start_revision + 1,
            node_id: turn_id.to_string(),
            status: ChatTreeNodeStatus::Completed,
        })),
    )?;
    append_event(
        path,
        timestamp,
        EventMsg::TurnComplete(TurnCompleteEvent {
            turn_id: turn_id.to_string(),
            last_agent_message: None,
            completed_at: None,
            duration_ms: None,
            time_to_first_token_ms: None,
        }),
    )
}

fn append_chat_tree_current_node_changed(
    path: &Path,
    timestamp: &str,
    revision: u64,
    node_id: &str,
) -> std::io::Result<()> {
    append_event(
        path,
        timestamp,
        EventMsg::ChatTreeCurrentNodeChanged(Box::new(ChatTreeCurrentNodeChangedEvent {
            revision,
            node_id: node_id.to_string(),
        })),
    )
}

fn append_event(path: &Path, timestamp: &str, event: EventMsg) -> std::io::Result<()> {
    let mut file = std::fs::OpenOptions::new().append(true).open(path)?;
    let payload = serde_json::to_value(event).map_err(std::io::Error::other)?;
    writeln!(
        file,
        "{}",
        json!({
            "timestamp": timestamp,
            "type": "event_msg",
            "payload": payload
        })
    )
}

fn append_thread_rollback(path: &Path, timestamp: &str, num_turns: u32) -> std::io::Result<()> {
    let mut file = std::fs::OpenOptions::new().append(true).open(path)?;
    writeln!(
        file,
        "{}",
        json!({
            "timestamp": timestamp,
            "type":"event_msg",
            "payload": {
                "type":"thread_rolled_back",
                "num_turns": num_turns
            }
        })
    )
}

fn turn_user_texts(turns: &[codex_app_server_protocol::Turn]) -> Vec<&str> {
    turns
        .iter()
        .filter_map(|turn| match turn.items.first()? {
            ThreadItem::UserMessage { content, .. } => match content.first()? {
                UserInput::Text { text, .. } => Some(text.as_str()),
                UserInput::Image { .. }
                | UserInput::LocalImage { .. }
                | UserInput::Skill { .. }
                | UserInput::Mention { .. } => None,
            },
            _ => None,
        })
        .collect()
}

// Helper to create a config.toml pointing at the mock model server.
fn create_config_toml(codex_home: &Path, server_uri: &str) -> std::io::Result<()> {
    let config_toml = codex_home.join("config.toml");
    std::fs::write(
        config_toml,
        format!(
            r#"
model = "mock-model"
approval_policy = "never"
sandbox_mode = "read-only"

model_provider = "mock_provider"

[model_providers.mock_provider]
name = "Mock provider for test"
base_url = "{server_uri}/v1"
wire_api = "responses"
request_max_retries = 0
stream_max_retries = 0
"#
        ),
    )
}
