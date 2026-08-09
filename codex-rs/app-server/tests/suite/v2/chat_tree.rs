use anyhow::Context;
use anyhow::Result;
use app_test_support::TestAppServer;
use app_test_support::create_final_assistant_message_sse_response;
use app_test_support::create_mock_responses_server_sequence_unchecked;
use app_test_support::to_response;
use codex_app_server_protocol::ChatTreeReadResponse;
use codex_app_server_protocol::ChatTreeSetCurrentResponse;
use codex_app_server_protocol::JSONRPCError;
use codex_app_server_protocol::JSONRPCResponse;
use codex_app_server_protocol::RequestId;
use codex_app_server_protocol::SortDirection;
use codex_app_server_protocol::ThreadHistoryMode;
use codex_app_server_protocol::ThreadReadParams;
use codex_app_server_protocol::ThreadReadResponse;
use codex_app_server_protocol::ThreadResumeInitialTurnsPageParams;
use codex_app_server_protocol::ThreadResumeParams;
use codex_app_server_protocol::ThreadResumeResponse;
use codex_app_server_protocol::ThreadStartParams;
use codex_app_server_protocol::ThreadStartResponse;
use codex_app_server_protocol::ThreadTurnsListParams;
use codex_app_server_protocol::ThreadTurnsListResponse;
use codex_app_server_protocol::TurnItemsView;
use codex_app_server_protocol::TurnStartParams;
use codex_app_server_protocol::UserInput;
use pretty_assertions::assert_eq;
use serde_json::json;
use std::path::Path;
use tempfile::TempDir;
use tokio::time::timeout;

#[cfg(windows)]
const DEFAULT_READ_TIMEOUT: std::time::Duration = std::time::Duration::from_secs(25);
#[cfg(not(windows))]
const DEFAULT_READ_TIMEOUT: std::time::Duration = std::time::Duration::from_secs(10);

#[tokio::test]
async fn chat_tree_rpc_switches_current_node_and_refreshes_visible_turns() -> Result<()> {
    let server = create_mock_responses_server_sequence_unchecked(vec![
        create_final_assistant_message_sse_response("first answer")?,
        create_final_assistant_message_sse_response("first summary")?,
        create_final_assistant_message_sse_response("second answer")?,
        create_final_assistant_message_sse_response("second summary")?,
    ])
    .await;
    let codex_home = TempDir::new()?;
    create_config_toml(codex_home.path(), &server.uri())?;

    let mut app = TestAppServer::builder()
        .with_codex_home(codex_home.path())
        .build_initialized()
        .await?;

    let thread_id = start_thread(&mut app).await?;
    start_completed_turn(&mut app, &thread_id, "first prompt").await?;
    wait_for_responses_request_count(&server, 2).await?;

    let first_tree = read_chat_tree(&mut app, &thread_id).await?;
    let first_node_id = first_tree
        .chat_tree
        .current_node_id
        .clone()
        .expect("first turn should create a current chat tree node");
    assert_eq!(first_tree.chat_tree.visible_turn_ids.len(), 1);

    start_completed_turn(&mut app, &thread_id, "second prompt").await?;
    wait_for_responses_request_count(&server, 4).await?;

    let second_tree = read_chat_tree(&mut app, &thread_id).await?;
    assert_eq!(second_tree.chat_tree.nodes.len(), 2);
    assert_eq!(second_tree.chat_tree.visible_turn_ids.len(), 2);

    let switched = set_current_chat_tree_node(
        &mut app,
        &thread_id,
        &first_node_id,
        Some(second_tree.chat_tree.revision),
    )
    .await?;
    assert_eq!(
        switched.chat_tree.current_node_id,
        Some(first_node_id.clone())
    );
    assert_eq!(switched.chat_tree.visible_turn_ids.len(), 1);

    let thread = read_thread_with_turns(&mut app, &thread_id).await?;
    assert_eq!(thread.thread.turns.len(), 1);

    Ok(())
}

#[tokio::test]
async fn selected_chat_tree_branch_is_consistent_across_transcript_apis() -> Result<()> {
    for history_mode in [ThreadHistoryMode::Legacy, ThreadHistoryMode::Paginated] {
        assert_selected_branch_transcript_contract(history_mode).await?;
    }
    Ok(())
}

#[tokio::test]
async fn chat_tree_read_returns_stable_error_kinds() -> Result<()> {
    let codex_home = TempDir::new()?;
    let mut app = TestAppServer::builder()
        .with_codex_home(codex_home.path())
        .build_initialized()
        .await?;

    assert_chat_tree_read_error_kind(&mut app, "not-a-thread-id", "invalidThreadId").await?;
    assert_chat_tree_read_error_kind(
        &mut app,
        "00000000-0000-4000-8000-000000000000",
        "threadNotMaterialized",
    )
    .await?;

    Ok(())
}

async fn assert_selected_branch_transcript_contract(history_mode: ThreadHistoryMode) -> Result<()> {
    let server = create_mock_responses_server_sequence_unchecked(vec![
        create_final_assistant_message_sse_response("first answer")?,
        create_final_assistant_message_sse_response("first summary")?,
        create_final_assistant_message_sse_response("second answer")?,
        create_final_assistant_message_sse_response("second summary")?,
    ])
    .await;
    let codex_home = TempDir::new()?;
    create_config_toml(codex_home.path(), &server.uri())?;

    let mut app = TestAppServer::builder()
        .with_codex_home(codex_home.path())
        .build_initialized()
        .await?;
    let thread_id = start_thread_with_history_mode(&mut app, history_mode).await?;
    start_completed_turn(&mut app, &thread_id, "first prompt").await?;
    wait_for_responses_request_count(&server, 2).await?;
    let first_tree = read_chat_tree(&mut app, &thread_id).await?;
    let first_node_id = first_tree
        .chat_tree
        .current_node_id
        .clone()
        .expect("first turn should create a current chat tree node");

    start_completed_turn(&mut app, &thread_id, "second prompt").await?;
    wait_for_responses_request_count(&server, 4).await?;
    let second_tree = read_chat_tree(&mut app, &thread_id).await?;
    let switched = set_current_chat_tree_node(
        &mut app,
        &thread_id,
        &first_node_id,
        Some(second_tree.chat_tree.revision),
    )
    .await?;
    let expected_turn_ids = switched.chat_tree.visible_turn_ids.clone();

    let thread_read = read_thread_with_turns(&mut app, &thread_id).await?;
    assert_eq!(turn_ids(&thread_read.thread.turns), expected_turn_ids);

    let turns_list = list_thread_turns(&mut app, &thread_id).await?;
    assert_eq!(turn_ids(&turns_list.data), expected_turn_ids);

    let hot_resume = resume_thread(&mut app, &thread_id).await?;
    assert_eq!(turn_ids(&hot_resume.thread.turns), expected_turn_ids);
    assert_eq!(
        turn_ids(
            &hot_resume
                .initial_turns_page
                .expect("resume should include the requested initial page")
                .data
        ),
        expected_turn_ids
    );

    drop(app);

    let mut resumed_app = TestAppServer::builder()
        .with_codex_home(codex_home.path())
        .build_initialized()
        .await?;
    let cold_resume = resume_thread(&mut resumed_app, &thread_id).await?;
    assert_eq!(turn_ids(&cold_resume.thread.turns), expected_turn_ids);
    assert_eq!(
        turn_ids(
            &cold_resume
                .initial_turns_page
                .expect("cold resume should include the requested initial page")
                .data
        ),
        expected_turn_ids
    );

    Ok(())
}

async fn start_thread(app: &mut TestAppServer) -> Result<String> {
    start_thread_with_history_mode(app, ThreadHistoryMode::Legacy).await
}

async fn start_thread_with_history_mode(
    app: &mut TestAppServer,
    history_mode: ThreadHistoryMode,
) -> Result<String> {
    let request_id = app
        .send_thread_start_request(ThreadStartParams {
            model: Some("gpt-5.4".to_string()),
            history_mode: Some(history_mode),
            ..Default::default()
        })
        .await?;
    let response: JSONRPCResponse = timeout(
        DEFAULT_READ_TIMEOUT,
        app.read_stream_until_response_message(RequestId::Integer(request_id)),
    )
    .await??;
    let ThreadStartResponse { thread, .. } = to_response(response)?;
    Ok(thread.id)
}

async fn list_thread_turns(
    app: &mut TestAppServer,
    thread_id: &str,
) -> Result<ThreadTurnsListResponse> {
    let request_id = app
        .send_thread_turns_list_request(ThreadTurnsListParams {
            thread_id: thread_id.to_string(),
            cursor: None,
            limit: None,
            sort_direction: Some(SortDirection::Asc),
            items_view: Some(TurnItemsView::Summary),
        })
        .await?;
    timeout(DEFAULT_READ_TIMEOUT, app.read_response(request_id)).await?
}

async fn resume_thread(app: &mut TestAppServer, thread_id: &str) -> Result<ThreadResumeResponse> {
    let request_id = app
        .send_thread_resume_request(ThreadResumeParams {
            thread_id: thread_id.to_string(),
            initial_turns_page: Some(ThreadResumeInitialTurnsPageParams {
                limit: None,
                sort_direction: Some(SortDirection::Asc),
                items_view: Some(TurnItemsView::Summary),
            }),
            ..Default::default()
        })
        .await?;
    timeout(DEFAULT_READ_TIMEOUT, app.read_response(request_id)).await?
}

async fn assert_chat_tree_read_error_kind(
    app: &mut TestAppServer,
    thread_id: &str,
    expected_kind: &str,
) -> Result<()> {
    let request_id = app
        .send_raw_request("chatTree/read", Some(json!({ "threadId": thread_id })))
        .await?;
    let error: JSONRPCError = timeout(
        DEFAULT_READ_TIMEOUT,
        app.read_stream_until_error_message(RequestId::Integer(request_id)),
    )
    .await??;
    assert_eq!(
        error.error.data,
        Some(json!({
            "kind": expected_kind,
            "threadId": thread_id,
        }))
    );
    Ok(())
}

fn turn_ids(turns: &[codex_app_server_protocol::Turn]) -> Vec<String> {
    turns.iter().map(|turn| turn.id.clone()).collect()
}

async fn start_completed_turn(app: &mut TestAppServer, thread_id: &str, text: &str) -> Result<()> {
    let request_id = app
        .send_turn_start_request(TurnStartParams {
            thread_id: thread_id.to_string(),
            input: vec![UserInput::Text {
                text: text.to_string(),
                text_elements: Vec::new(),
            }],
            ..Default::default()
        })
        .await?;
    timeout(
        DEFAULT_READ_TIMEOUT,
        app.read_stream_until_response_message(RequestId::Integer(request_id)),
    )
    .await??;
    timeout(
        DEFAULT_READ_TIMEOUT,
        app.read_stream_until_notification_message("turn/completed"),
    )
    .await??;
    Ok(())
}

async fn read_chat_tree(app: &mut TestAppServer, thread_id: &str) -> Result<ChatTreeReadResponse> {
    let request_id = app
        .send_raw_request("chatTree/read", Some(json!({ "threadId": thread_id })))
        .await?;
    let response = timeout(
        DEFAULT_READ_TIMEOUT,
        app.read_stream_until_response_message(RequestId::Integer(request_id)),
    )
    .await??;
    to_response(response)
}

async fn set_current_chat_tree_node(
    app: &mut TestAppServer,
    thread_id: &str,
    node_id: &str,
    expected_revision: Option<u64>,
) -> Result<ChatTreeSetCurrentResponse> {
    let request_id = app
        .send_raw_request(
            "chatTree/setCurrent",
            Some(json!({
                "threadId": thread_id,
                "nodeId": node_id,
                "expectedRevision": expected_revision,
            })),
        )
        .await?;
    let response = timeout(
        DEFAULT_READ_TIMEOUT,
        app.read_stream_until_response_message(RequestId::Integer(request_id)),
    )
    .await??;
    to_response(response)
}

async fn read_thread_with_turns(
    app: &mut TestAppServer,
    thread_id: &str,
) -> Result<ThreadReadResponse> {
    let request_id = app
        .send_thread_read_request(ThreadReadParams {
            thread_id: thread_id.to_string(),
            include_turns: true,
        })
        .await?;
    let response = timeout(
        DEFAULT_READ_TIMEOUT,
        app.read_stream_until_response_message(RequestId::Integer(request_id)),
    )
    .await??;
    to_response(response)
}

async fn wait_for_responses_request_count(
    server: &wiremock::MockServer,
    expected_count: usize,
) -> Result<()> {
    timeout(DEFAULT_READ_TIMEOUT, async {
        loop {
            let requests = server
                .received_requests()
                .await
                .context("failed to fetch wiremock requests")?;
            let count = requests
                .iter()
                .filter(|request| {
                    request.method == "POST" && request.url.path().ends_with("/responses")
                })
                .count();
            if count == expected_count {
                return Ok::<(), anyhow::Error>(());
            }
            if count > expected_count {
                anyhow::bail!("expected {expected_count} /responses requests, got {count}");
            }
            tokio::time::sleep(std::time::Duration::from_millis(10)).await;
        }
    })
    .await??;
    Ok(())
}

fn create_config_toml(codex_home: &Path, server_uri: &str) -> std::io::Result<()> {
    std::fs::write(
        codex_home.join("config.toml"),
        format!(
            r#"
model = "gpt-5.4"
approval_policy = "never"
sandbox_mode = "danger-full-access"
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
