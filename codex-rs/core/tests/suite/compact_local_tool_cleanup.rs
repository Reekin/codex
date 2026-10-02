//! Durable tool marks, cleanup status, and manual cleanup through the production session.

use super::local_support::*;
use anyhow::Result;
use codex_core::ToolCleanupOutcome;
use codex_core::ToolCleanupStatus;
use codex_core::config::Config;
use core_test_support::responses::start_mock_server;
use core_test_support::skip_if_no_network;
use core_test_support::test_codex::test_codex;
use pretty_assertions::assert_eq;

/// Marks are validated, but the automatic savings requirement is far out of reach.
fn configure_unreached_savings(config: &mut Config) {
    configure_marking(config);
    config.local_compaction.reclaim_percent = 50;
}

fn classifier_requests(model: &LocalModel) -> usize {
    model
        .bodies()
        .iter()
        .filter(|body| analysis_payload(body, CLASSIFY).is_some())
        .count()
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn validated_marks_survive_resume_and_apply_manually() -> Result<()> {
    skip_if_no_network!(Ok(()));
    let server = start_mock_server().await;
    let model = LocalModel::mount(&server).await;
    let mut builder = test_codex().with_config(configure_unreached_savings);
    let test = builder.build_with_auto_env(&server).await?;
    model.reply(tool_turn());
    model.text("Evidence received.");
    test.submit_text_turn("Collect evidence.").await?;
    model.text("Recent dialogue.");
    test.submit_text_turn("Continue.").await?;
    finish_marking(&test, &model, /*expected_batches*/ 1).await?;
    test.codex.flush_rollout().await?;
    let path = test.codex.rollout_path().unwrap();
    assert!(checkpoints(&path)?.is_empty());

    // Usage is recorded before validation finishes; a status read consumes the finished batch.
    let mut status = test.codex.tool_cleanup_status().await?;
    for _ in 0..100 {
        if !status.marking {
            break;
        }
        tokio::time::sleep(std::time::Duration::from_millis(50)).await;
        status = test.codex.tool_cleanup_status().await?;
    }
    assert!(status.pending_savings_tokens > 0);
    assert!(status.required_savings_tokens > status.pending_savings_tokens);
    assert_eq!(
        status,
        ToolCleanupStatus {
            enabled: true,
            marking: false,
            ..status
        }
    );
    let marks = test
        .config
        .codex_home
        .as_path()
        .join("local_compaction")
        .join(format!("{}.jsonl", test.session_configured.thread_id));
    assert!(std::fs::read_to_string(&marks)?.contains(SHORTENED));
    let classified = classifier_requests(&model);

    let resumed = builder.restart(&server, &test).await?;
    assert_eq!(resumed.codex.tool_cleanup_status().await?, status);
    model.text("Resumed without reclassifying.");
    resumed.submit_text_turn("After resume.").await?;
    assert_eq!(classifier_requests(&model), classified);

    let outcome = resumed.codex.apply_tool_cleanup().await?;
    assert_eq!(
        outcome,
        ToolCleanupOutcome {
            released_tokens: status.pending_savings_tokens,
            status: ToolCleanupStatus {
                pending_savings_tokens: 0,
                ..status
            },
        }
    );
    assert_eq!(
        resumed.codex.apply_tool_cleanup().await?,
        ToolCleanupOutcome {
            released_tokens: 0,
            status: outcome.status,
        }
    );
    model.text("Using the cleaned view.");
    resumed
        .submit_text_turn("Inspect cleaned evidence.")
        .await?;
    let request = model.ordinary_bodies().pop().unwrap().to_string();
    assert!(request.contains(SHORTENED));
    assert!(request.contains("Output omitted; original item:"));
    resumed.codex.flush_rollout().await?;
    assert_eq!(checkpoints(&path)?.len(), 1);
    assert_eq!(classifier_requests(&model), classified);
    Ok(())
}
