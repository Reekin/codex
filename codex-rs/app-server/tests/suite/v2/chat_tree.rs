use anyhow::Context;
use anyhow::Result;
use app_test_support::TestAppServer;
use app_test_support::create_final_assistant_message_sse_response;
use app_test_support::create_mock_responses_server_sequence_unchecked;
use app_test_support::create_shell_command_sse_response;
use app_test_support::to_response;
use codex_app_server_protocol::ChatTreeReadResponse;
use codex_app_server_protocol::ChatTreeSetCurrentResponse;
use codex_app_server_protocol::ItemStartedNotification;
use codex_app_server_protocol::JSONRPCError;
use codex_app_server_protocol::JSONRPCResponse;
use codex_app_server_protocol::RequestId;
use codex_app_server_protocol::ThreadItem;
use codex_app_server_protocol::ThreadReadParams;
use codex_app_server_protocol::ThreadReadResponse;
use codex_app_server_protocol::ThreadStartParams;
use codex_app_server_protocol::ThreadStartResponse;
use codex_app_server_protocol::TurnStartParams;
use codex_app_server_protocol::TurnStartResponse;
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
    let responses = vec![
        create_final_assistant_message_sse_response("first answer")?,
        create_final_assistant_message_sse_response("first summary")?,
        create_final_assistant_message_sse_response("second answer")?,
        create_final_assistant_message_sse_response("second summary")?,
        create_shell_command_sse_response(
            vec![
                "python".to_string(),
                "-c".to_string(),
                "import time; time.sleep(5)".to_string(),
            ],
            /*workdir*/ None,
            Some(5000),
            "call-running",
        )?,
    ];
    let server = create_mock_responses_server_sequence_unchecked(responses).await;
    let codex_home = TempDir::new()?;
    create_config_toml(codex_home.path(), &server.uri())?;

    let mut app = TestAppServer::new(codex_home.path()).await?;
    timeout(DEFAULT_READ_TIMEOUT, app.initialize()).await??;

    let thread_id = start_thread(&mut app).await?;
    start_completed_turn(&mut app, &thread_id, "first prompt").await?;
    wait_for_responses_request_count(&server, 2).await?;

    let first_tree = read_chat_tree(&mut app, &thread_id).await?;
    let first_node_id = first_tree
        .chat_tree
        .current_node_id
        .clone()
        .expect("first turn should create current chat tree node");
    assert_eq!(first_tree.chat_tree.visible_turn_ids.len(), 1);

    start_completed_turn(&mut app, &thread_id, "second prompt").await?;
    wait_for_responses_request_count(&server, 4).await?;

    let second_tree = read_chat_tree(&mut app, &thread_id).await?;
    assert_ne!(
        second_tree.chat_tree.current_node_id,
        Some(first_node_id.clone())
    );
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

    let running_turn = start_running_turn(&mut app, &thread_id).await?;
    wait_for_command_execution_started(&mut app).await?;
    let running_switch_id = app
        .send_raw_request(
            "chatTree/setCurrent",
            Some(json!({
                "threadId": thread_id,
                "nodeId": first_node_id,
                "expectedRevision": switched.chat_tree.revision,
            })),
        )
        .await?;
    let running_switch_error: JSONRPCError = timeout(
        DEFAULT_READ_TIMEOUT,
        app.read_stream_until_error_message(RequestId::Integer(running_switch_id)),
    )
    .await??;
    let error_data = running_switch_error
        .error
        .data
        .as_ref()
        .expect("taskRunning error data");
    assert_eq!(error_data["kind"], "taskRunning");
    assert!(
        running_switch_error
            .error
            .message
            .contains("Cannot switch chat tree nodes while a task is running.")
    );

    app.interrupt_turn_and_wait_for_aborted(thread_id, running_turn.turn.id, DEFAULT_READ_TIMEOUT)
        .await?;

    Ok(())
}

async fn wait_for_command_execution_started(
    app: &mut TestAppServer,
) -> Result<ItemStartedNotification> {
    loop {
        let notification = app
            .read_stream_until_notification_message("item/started")
            .await?;
        let started: ItemStartedNotification = serde_json::from_value(
            notification
                .params
                .ok_or_else(|| anyhow::anyhow!("missing item/started params"))?,
        )?;
        if matches!(&started.item, ThreadItem::CommandExecution { .. }) {
            return Ok(started);
        }
    }
}

async fn start_thread(app: &mut TestAppServer) -> Result<String> {
    let start_id = app
        .send_thread_start_request(ThreadStartParams {
            model: Some("gpt-5.4".to_string()),
            ..Default::default()
        })
        .await?;
    let start_response: JSONRPCResponse = timeout(
        DEFAULT_READ_TIMEOUT,
        app.read_stream_until_response_message(RequestId::Integer(start_id)),
    )
    .await??;
    let ThreadStartResponse { thread, .. } = to_response::<ThreadStartResponse>(start_response)?;
    Ok(thread.id)
}

async fn start_completed_turn(app: &mut TestAppServer, thread_id: &str, text: &str) -> Result<()> {
    let turn_id = app
        .send_turn_start_request(TurnStartParams {
            thread_id: thread_id.to_string(),
            client_user_message_id: None,
            input: vec![UserInput::Text {
                text: text.to_string(),
                text_elements: Vec::new(),
            }],
            ..Default::default()
        })
        .await?;
    timeout(
        DEFAULT_READ_TIMEOUT,
        app.read_stream_until_response_message(RequestId::Integer(turn_id)),
    )
    .await??;
    timeout(
        DEFAULT_READ_TIMEOUT,
        app.read_stream_until_notification_message("turn/completed"),
    )
    .await??;
    Ok(())
}

async fn start_running_turn(app: &mut TestAppServer, thread_id: &str) -> Result<TurnStartResponse> {
    let turn_id = app
        .send_turn_start_request(TurnStartParams {
            thread_id: thread_id.to_string(),
            client_user_message_id: None,
            input: vec![UserInput::Text {
                text: "keep running".to_string(),
                text_elements: Vec::new(),
            }],
            ..Default::default()
        })
        .await?;
    let turn_response: JSONRPCResponse = timeout(
        DEFAULT_READ_TIMEOUT,
        app.read_stream_until_response_message(RequestId::Integer(turn_id)),
    )
    .await??;
    timeout(
        DEFAULT_READ_TIMEOUT,
        app.read_stream_until_notification_message("turn/started"),
    )
    .await??;
    to_response::<TurnStartResponse>(turn_response)
}

async fn read_chat_tree(app: &mut TestAppServer, thread_id: &str) -> Result<ChatTreeReadResponse> {
    let request_id = app
        .send_raw_request(
            "chatTree/read",
            Some(json!({
                "threadId": thread_id,
            })),
        )
        .await?;
    let response: JSONRPCResponse = timeout(
        DEFAULT_READ_TIMEOUT,
        app.read_stream_until_response_message(RequestId::Integer(request_id)),
    )
    .await??;
    to_response::<ChatTreeReadResponse>(response)
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
    let response: JSONRPCResponse = timeout(
        DEFAULT_READ_TIMEOUT,
        app.read_stream_until_response_message(RequestId::Integer(request_id)),
    )
    .await??;
    to_response::<ChatTreeSetCurrentResponse>(response)
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
    let response: JSONRPCResponse = timeout(
        DEFAULT_READ_TIMEOUT,
        app.read_stream_until_response_message(RequestId::Integer(request_id)),
    )
    .await??;
    to_response::<ThreadReadResponse>(response)
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
            let responses_request_count = requests
                .iter()
                .filter(|request| {
                    request.method == "POST" && request.url.path().ends_with("/responses")
                })
                .count();
            if responses_request_count == expected_count {
                return Ok::<(), anyhow::Error>(());
            }
            if responses_request_count > expected_count {
                anyhow::bail!(
                    "expected exactly {expected_count} /responses requests, got {responses_request_count}"
                );
            }
            tokio::time::sleep(std::time::Duration::from_millis(10)).await;
        }
    })
    .await??;
    Ok(())
}

fn create_config_toml(codex_home: &Path, server_uri: &str) -> std::io::Result<()> {
    let config_toml = codex_home.join("config.toml");
    std::fs::write(
        config_toml,
        format!(
            r#"
model = "gpt-5.3-codex"
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
