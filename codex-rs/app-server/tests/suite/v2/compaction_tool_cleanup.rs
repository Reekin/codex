//! `thread/toolCleanup/{read,apply}` with local context compaction enabled.

use anyhow::Result;
use app_test_support::TestAppServer;
use codex_app_server_protocol::ThreadToolCleanupApplyParams;
use codex_app_server_protocol::ThreadToolCleanupApplyResponse;
use codex_app_server_protocol::ThreadToolCleanupReadParams;
use codex_app_server_protocol::ThreadToolCleanupReadResponse;
use codex_app_server_protocol::ThreadToolCleanupStatus;
use codex_features::Feature;
use core_test_support::responses;
use core_test_support::skip_if_no_network;
use pretty_assertions::assert_eq;
use std::collections::BTreeMap;
use tempfile::TempDir;
use tokio::time::timeout;

use super::DEFAULT_READ_TIMEOUT;
use super::compaction_config;
use super::send_turn_and_wait;
use super::start_thread;

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn thread_tool_cleanup_reports_marks_and_applies_them_on_request() -> Result<()> {
    skip_if_no_network!(Ok(()));

    let server = responses::start_mock_server().await;
    let tools = responses::sse(vec![
        responses::ev_function_call("call-a", &format!("a_{}", "evidence_".repeat(1000)), "{}"),
        responses::ev_function_call("call-b", &format!("b_{}", "evidence_".repeat(1000)), "{}"),
        responses::ev_completed_with_tokens("tools", /*total_tokens*/ 120),
    ]);
    let mut ordinary = vec![tools];
    ordinary.extend((0..10).map(|index| {
        responses::sse(vec![
            responses::ev_assistant_message(&format!("reply-{index}"), "Continuing."),
            responses::ev_completed_with_tokens(
                &format!("reply-{index}"),
                /*total_tokens*/ 120,
            ),
        ])
    }));
    let model = responses::mount_local_compaction_sequence(
        &server,
        ordinary,
        responses::ev_completed_with_tokens("local-analysis", /*total_tokens*/ 50),
    )
    .await;

    let codex_home = TempDir::new()?;
    compaction_config(&server.uri(), /*auto_compact_limit*/ 90_000)
        .with_root_config(
            "model_context_window = 100000\n\
             local_compaction.reclaim_percent = 50\n\
             local_compaction.mark_after_tokens_percent = 1",
        )
        .with_features(&BTreeMap::from([(Feature::LocalContextCompaction, true)]))
        .write(codex_home.path())?;
    let mut mcp = TestAppServer::builder()
        .with_codex_home(codex_home.path())
        .build_initialized_with_timeout(DEFAULT_READ_TIMEOUT)
        .await?;
    let thread_id = start_thread(&mut mcp).await?;
    let initial = read_tool_cleanup(&mut mcp, &thread_id).await?;
    assert_eq!(
        initial,
        ThreadToolCleanupStatus {
            marking: false,
            pending_savings_tokens: 0,
            ..initial
        }
    );
    assert!(initial.required_savings_tokens > 0);

    send_turn_and_wait(&mut mcp, &thread_id, "Collect evidence.").await?;
    let mut status = read_tool_cleanup(&mut mcp, &thread_id).await?;
    for _ in 0..5 {
        if status.pending_savings_tokens > 0 {
            break;
        }
        send_turn_and_wait(&mut mcp, &thread_id, "Continue.").await?;
        // A status read consumes a finished batch without waiting for the next request.
        for _ in 0..50 {
            status = read_tool_cleanup(&mut mcp, &thread_id).await?;
            if !status.marking {
                break;
            }
            tokio::time::sleep(std::time::Duration::from_millis(100)).await;
        }
    }
    assert!(status.pending_savings_tokens > 0);
    assert!(status.required_savings_tokens > status.pending_savings_tokens);

    let apply_id = mcp
        .send_request(
            "thread/toolCleanup/apply",
            Some(serde_json::to_value(ThreadToolCleanupApplyParams {
                thread_id: thread_id.clone(),
            })?),
        )
        .await?;
    let applied: ThreadToolCleanupApplyResponse =
        timeout(DEFAULT_READ_TIMEOUT, mcp.read_response(apply_id)).await??;
    assert_eq!(
        applied,
        ThreadToolCleanupApplyResponse {
            released_tokens: status.pending_savings_tokens,
            status: ThreadToolCleanupStatus {
                pending_savings_tokens: 0,
                ..status
            },
        }
    );

    send_turn_and_wait(&mut mcp, &thread_id, "Inspect cleaned evidence.").await?;
    let last = model.requests().pop().expect("final request").body_json();
    assert!(last.to_string().contains("Output omitted; original item:"));
    Ok(())
}

async fn read_tool_cleanup(
    mcp: &mut TestAppServer,
    thread_id: &str,
) -> Result<ThreadToolCleanupStatus> {
    let request_id = mcp
        .send_request(
            "thread/toolCleanup/read",
            Some(serde_json::to_value(ThreadToolCleanupReadParams {
                thread_id: thread_id.to_string(),
            })?),
        )
        .await?;
    let ThreadToolCleanupReadResponse { status } =
        timeout(DEFAULT_READ_TIMEOUT, mcp.read_response(request_id)).await??;
    Ok(status)
}
