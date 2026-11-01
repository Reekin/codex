//! Context usage on live token usage notifications, attach replay, and `thread/tokenUsage/read`.

use anyhow::Result;
use app_test_support::MockResponsesConfig;
use app_test_support::TestAppServer;
use app_test_support::create_fake_rollout_with_token_usage;
use app_test_support::create_mock_responses_server_sequence;
use codex_app_server_protocol::ServerNotification;
use codex_app_server_protocol::ThreadContextUsage;
use codex_app_server_protocol::ThreadResumeParams;
use codex_app_server_protocol::ThreadResumeResponse;
use codex_app_server_protocol::ThreadStartParams;
use codex_app_server_protocol::ThreadStartResponse;
use codex_app_server_protocol::ThreadTokenUsageReadParams;
use codex_app_server_protocol::ThreadTokenUsageReadResponse;
use codex_app_server_protocol::ThreadTokenUsageUpdatedNotification;
use codex_app_server_protocol::TurnStartParams;
use codex_app_server_protocol::TurnStartResponse;
use codex_app_server_protocol::UserInput;
use core_test_support::responses;
use pretty_assertions::assert_eq;
use std::time::Duration;
use tempfile::TempDir;
use tokio::time::timeout;

const DEFAULT_READ_TIMEOUT: Duration = Duration::from_secs(10);
const AUTO_COMPACT_TOKEN_LIMIT: i64 = 50_000;
const DEVELOPER_INSTRUCTIONS: &str =
    "Report context usage fixture results in short sentences and keep every answer factual.";
const AGENTS_MD_INSTRUCTIONS: &str =
    "Workspace instructions for the context usage fixture: prefer small focused changes.";

async fn read_token_usage_read(
    mcp: &mut TestAppServer,
    thread_id: &str,
) -> Result<ThreadTokenUsageReadResponse> {
    let params = ThreadTokenUsageReadParams {
        thread_id: thread_id.to_string(),
    };
    let request_id = mcp
        .send_request(
            "thread/tokenUsage/read",
            Some(serde_json::to_value(params)?),
        )
        .await?;
    timeout(DEFAULT_READ_TIMEOUT, mcp.read_response(request_id)).await?
}

async fn read_token_usage_notification(
    mcp: &mut TestAppServer,
) -> Result<ThreadTokenUsageUpdatedNotification> {
    let note = timeout(
        DEFAULT_READ_TIMEOUT,
        mcp.read_stream_until_notification_message("thread/tokenUsage/updated"),
    )
    .await??;
    let ServerNotification::ThreadTokenUsageUpdated(notification) = note.try_into()? else {
        panic!("expected thread/tokenUsage/updated notification");
    };
    Ok(notification)
}

async fn run_turn(
    mcp: &mut TestAppServer,
    thread_id: &str,
    text: &str,
    response_total_tokens: i64,
) -> Result<(String, ThreadTokenUsageUpdatedNotification)> {
    let request_id = mcp
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
    let TurnStartResponse { turn } =
        timeout(DEFAULT_READ_TIMEOUT, mcp.read_response(request_id)).await??;
    // Skip usage updates emitted before this turn's response completed.
    let notification = loop {
        let notification = read_token_usage_notification(mcp).await?;
        if notification.turn_id == turn.id
            && notification.token_usage.last.total_tokens == response_total_tokens
        {
            break notification;
        }
    };
    timeout(
        DEFAULT_READ_TIMEOUT,
        mcp.read_stream_until_notification_message("turn/completed"),
    )
    .await??;
    Ok((turn.id, notification))
}

fn breakdown_sum(usage: &ThreadContextUsage) -> i64 {
    let breakdown = &usage.breakdown;
    [
        breakdown.base_instructions,
        breakdown.developer_instructions,
        breakdown.agents_md,
        breakdown.skills,
        breakdown.tools,
        breakdown.environment,
        breakdown.user_messages,
        breakdown.agent_messages,
        breakdown.tool_calls,
        breakdown.reasoning,
        breakdown.compaction,
        breakdown.other,
    ]
    .into_iter()
    .sum()
}

#[tokio::test]
async fn token_usage_carries_context_usage_and_is_readable_per_turn() -> Result<()> {
    let server = create_mock_responses_server_sequence(vec![
        responses::sse(vec![
            responses::ev_response_created("resp-1"),
            responses::ev_assistant_message("msg-1", "first answer"),
            responses::ev_completed_with_tokens("resp-1", /*total_tokens*/ 5_000),
        ]),
        responses::sse(vec![
            responses::ev_response_created("resp-tool"),
            responses::ev_function_call("call-1", "unknown_test_tool", "{}"),
            responses::ev_completed_with_tokens("resp-tool", /*total_tokens*/ 5_500),
        ]),
        responses::sse(vec![
            responses::ev_response_created("resp-2"),
            responses::ev_assistant_message("msg-2", "second answer"),
            responses::ev_completed_with_tokens("resp-2", /*total_tokens*/ 6_000),
        ]),
    ])
    .await;
    let codex_home = TempDir::new()?;
    MockResponsesConfig::new(&server.uri())
        .with_model("gpt-5.4")
        .with_root_config(&format!(
            "model_auto_compact_token_limit = {AUTO_COMPACT_TOKEN_LIMIT}\ndeveloper_instructions = \"{DEVELOPER_INSTRUCTIONS}\""
        ))
        .write(codex_home.path())?;

    let mut mcp = TestAppServer::builder()
        .with_codex_home(codex_home.path())
        .build_initialized()
        .await?;
    {
        let auto_env = mcp.auto_env()?;
        let environment_cwd = auto_env.selection().cwd.clone();
        let agents_md = environment_cwd.join("AGENTS.md")?;
        auto_env
            .environment()
            .get_filesystem()
            .write_file(
                &agents_md,
                AGENTS_MD_INSTRUCTIONS.as_bytes().to_vec(),
                Default::default(),
                /*sandbox*/ None,
            )
            .await?;
    }
    let start_id = mcp
        .send_thread_start_request_with_auto_env(ThreadStartParams {
            model: Some("gpt-5.4".to_string()),
            ..Default::default()
        })
        .await?;
    let ThreadStartResponse { thread, .. } =
        timeout(DEFAULT_READ_TIMEOUT, mcp.read_response(start_id)).await??;

    let (first_turn_id, first) = run_turn(&mut mcp, &thread.id, "first question", 5_000).await?;
    // The second turn calls a tool and receives its output before the final message.
    let (second_turn_id, second) = run_turn(&mut mcp, &thread.id, "second question", 6_000).await?;

    // The live notification reports the active context that drives auto-compaction. The turn
    // ends on the model's message, so nothing is pending on top of the provider total.
    let context_usage = second
        .token_usage
        .context_usage
        .clone()
        .expect("live token usage should carry context usage");
    assert_eq!(context_usage.tokens, 6_000);
    assert_eq!(breakdown_sum(&context_usage), context_usage.tokens);
    assert_eq!(
        context_usage.auto_compact_token_limit,
        Some(AUTO_COMPACT_TOKEN_LIMIT)
    );
    assert!(context_usage.breakdown.developer_instructions > 0);
    assert!(context_usage.breakdown.agents_md > 0);
    assert!(context_usage.breakdown.user_messages > 0);
    assert!(context_usage.breakdown.agent_messages > 0);
    assert!(context_usage.breakdown.tool_calls > 0);
    assert!(context_usage.breakdown.tools > 0);
    assert!(first.token_usage.context_usage.is_some());

    let live_read = read_token_usage_read(&mut mcp, &thread.id).await?;
    assert_eq!(
        live_read
            .data
            .iter()
            .map(|entry| (entry.turn_id.clone(), entry.token_usage.last.total_tokens))
            .collect::<Vec<_>>(),
        vec![(first_turn_id, 5_000), (second_turn_id, 6_000)]
    );
    let persisted_context_usage = live_read.data[1].token_usage.context_usage.clone();
    assert!(persisted_context_usage.is_some());
    mcp.shutdown_gracefully().await?;

    // A new app-server reads the same per-turn usage without loading the thread, and replays
    // the persisted context usage when a client attaches.
    let mut restarted = TestAppServer::builder()
        .with_codex_home(codex_home.path())
        .build_initialized()
        .await?;
    let unloaded_read = read_token_usage_read(&mut restarted, &thread.id).await?;
    assert_eq!(unloaded_read, live_read);

    let resume_id = restarted
        .send_thread_resume_request(ThreadResumeParams {
            thread_id: thread.id.clone(),
            ..Default::default()
        })
        .await?;
    let _: ThreadResumeResponse =
        timeout(DEFAULT_READ_TIMEOUT, restarted.read_response(resume_id)).await??;
    let replayed = read_token_usage_notification(&mut restarted).await?;
    assert_eq!(replayed.token_usage.context_usage, persisted_context_usage);
    Ok(())
}

#[tokio::test]
async fn token_usage_read_returns_legacy_snapshots_without_context_usage() -> Result<()> {
    let server = create_mock_responses_server_sequence(Vec::new()).await;
    let codex_home = TempDir::new()?;
    MockResponsesConfig::new(&server.uri())
        .with_model("gpt-5.4")
        .write(codex_home.path())?;
    let thread_id = create_fake_rollout_with_token_usage(
        codex_home.path(),
        "2025-01-05T12-00-00",
        "2025-01-05T12:00:00Z",
        "Saved user message",
        Some("mock_provider"),
    )?;
    let mut mcp = TestAppServer::builder()
        .with_codex_home(codex_home.path())
        .without_auto_env()
        .build_initialized()
        .await?;

    let read = read_token_usage_read(&mut mcp, &thread_id).await?;

    assert_eq!(read.data.len(), 1);
    assert_eq!(read.data[0].token_usage.total.total_tokens, 150);
    assert_eq!(read.data[0].token_usage.last.total_tokens, 90);
    assert_eq!(read.data[0].token_usage.context_usage, None);
    Ok(())
}
