//! Durable tool marks, cleanup status, and manual cleanup through the production session.

use super::local_support::*;
use anyhow::Result;
use codex_core::ToolCleanupOutcome;
use codex_core::ToolCleanupStatus;
use codex_core::config::Config;
use codex_protocol::models::ResponseItem;
use core_test_support::responses::ev_assistant_message;
use core_test_support::responses::ev_completed;
use core_test_support::responses::ev_completed_with_tokens;
use core_test_support::responses::ev_function_call;
use core_test_support::responses::ev_reasoning_item;
use core_test_support::responses::sse;
use core_test_support::responses::start_mock_server;
use core_test_support::skip_if_no_network;
use core_test_support::test_codex::test_codex;
use pretty_assertions::assert_eq;
use serde_json::Value;
use serde_json::json;

const SCREENSHOT: &str = "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNkYPhfDwAChwGA60e6kgAAAABJRU5ErkJggg==";

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
async fn many_small_outputs_are_marked_in_one_batch() -> Result<()> {
    skip_if_no_network!(Ok(()));
    let server = start_mock_server().await;
    let model = LocalModel::mount(&server).await;
    let test = test_codex()
        .with_config(configure_unreached_savings)
        .build_with_auto_env(&server)
        .await?;
    // Small outputs leave little room for replacements, so the batch is not cut at a fixed count.
    let mut events = (0..80)
        .map(|index| {
            ev_function_call(
                &format!("small-{index}"),
                &format!("small_{index}_{}", "evidence_".repeat(12)),
                "{}",
            )
        })
        .collect::<Vec<_>>();
    events.push(ev_completed("small-tools"));
    model.reply(sse(events));
    model.text("Small results received.");
    test.submit_text_turn("Collect small evidence.").await?;
    model.text("Recent dialogue.");
    test.submit_text_turn("Continue.").await?;
    finish_marking(&test, &model, /*expected_batches*/ 1).await?;
    let batches = model
        .bodies()
        .iter()
        .filter_map(|body| analysis_payload(body, CLASSIFY))
        .map(|payload| payload["candidates"].as_array().unwrap().len())
        .collect::<Vec<_>>();
    // Later batches may reconsider kept results after new user input.
    assert_eq!(batches[0], 80);
    Ok(())
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn tool_cleanup_runs_on_the_native_compaction_route() -> Result<()> {
    skip_if_no_network!(Ok(()));
    let server = start_mock_server().await;
    let model = LocalModel::mount(&server).await;
    // The default test provider and model both support native compaction.
    let test = test_codex()
        .with_config(|config| {
            configure_budgets(config);
            config.local_compaction.mark_after_tokens_percent = 1;
        })
        .build_with_auto_env(&server)
        .await?;
    model.reply(tool_turn());
    model.text("Evidence received.");
    test.submit_text_turn("Collect evidence.").await?;
    model.text("Recent dialogue.");
    test.submit_text_turn("Continue.").await?;
    finish_marking(&test, &model, /*expected_batches*/ 1).await?;
    model.text("Following installed view.");
    test.submit_text_turn("Inspect the cleaned view.").await?;
    let installed = model.bodies().pop().expect("follow-up");
    let shortened = installed["input"]
        .as_array()
        .unwrap()
        .iter()
        .find(|item| {
            item["type"] == "function_call_output" && item["call_id"] == "original-shorten"
        })
        .expect("shortened result")
        .to_string();
    assert!(shortened.contains(SHORTENED), "{shortened}");
    Ok(())
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn cleanup_status_follows_provider_token_counts() -> Result<()> {
    skip_if_no_network!(Ok(()));
    let server = start_mock_server().await;
    let model = LocalModel::mount(&server).await;
    let test = test_codex()
        .with_config(configure_unreached_savings)
        .build_with_auto_env(&server)
        .await?;
    model.reply(tool_turn());
    model.text("Evidence received.");
    test.submit_text_turn("Collect evidence.").await?;
    model.text("Recent dialogue.");
    test.submit_text_turn("Continue.").await?;
    finish_marking(&test, &model, /*expected_batches*/ 1).await?;
    let mut estimated = test.codex.tool_cleanup_status().await?;
    for _ in 0..100 {
        if !estimated.marking {
            break;
        }
        tokio::time::sleep(std::time::Duration::from_millis(50)).await;
        estimated = test.codex.tool_cleanup_status().await?;
    }
    assert!(estimated.pending_savings_tokens > 0);

    // The provider counts this context well above the local estimate.
    model.reply(sse(vec![
        ev_assistant_message("counted", "Counted by the provider."),
        ev_completed_with_tokens("counted", /*total_tokens*/ 60_000),
    ]));
    test.submit_text_turn("Report provider usage.").await?;
    let reported = test.codex.tool_cleanup_status().await?;
    assert!(
        reported.pending_savings_tokens > estimated.pending_savings_tokens,
        "{estimated:?} -> {reported:?}"
    );
    assert!(
        reported.pending_savings_tokens <= estimated.pending_savings_tokens * 2,
        "{estimated:?} -> {reported:?}"
    );
    // The requirement is a share of the provider-counted window either way.
    assert!(
        reported
            .required_savings_tokens
            .abs_diff(estimated.required_savings_tokens)
            <= 2,
        "{estimated:?} -> {reported:?}"
    );
    assert!(
        model
            .bodies()
            .iter()
            .all(|body| analysis_payload(body, SUMMARIZE).is_none())
    );
    Ok(())
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn validated_marks_survive_resume_and_apply_manually() -> Result<()> {
    skip_if_no_network!(Ok(()));
    let server = start_mock_server().await;
    let model = LocalModel::mount(&server).await;
    let test = test_codex()
        .with_config(configure_unreached_savings)
        .build_with_auto_env(&server)
        .await?;
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

    // Builders consume their config mutators; resume with a fresh one carrying the same config.
    let resumed = test_codex()
        .with_config(configure_unreached_savings)
        .restart(&server, &test)
        .await?;
    assert_eq!(resumed.codex.tool_cleanup_status().await?, status);
    model.text("Resumed without reclassifying.");
    resumed.submit_text_turn("After resume.").await?;
    assert_eq!(classifier_requests(&model), classified);

    // A fork inherits still-pending marks and keeps its own copy of them.
    resumed.codex.flush_rollout().await?;
    let forked = resumed
        .thread_manager
        .fork_thread(
            usize::MAX,
            resumed.config.clone(),
            path.clone(),
            /*thread_source*/ None,
            /*parent_trace*/ None,
        )
        .await?;
    assert_eq!(forked.thread.tool_cleanup_status().await?, status);
    let fork_marks = test
        .config
        .codex_home
        .as_path()
        .join("local_compaction")
        .join(format!("{}.jsonl", forked.thread_id));
    assert!(std::fs::read_to_string(&fork_marks)?.contains(SHORTENED));
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

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn screenshots_and_large_calls_are_cleaned_and_images_recalled() -> Result<()> {
    skip_if_no_network!(Ok(()));
    let server = start_mock_server().await;
    let model = LocalModel::mount(&server).await;
    let test = test_codex()
        .with_config(configure_unreached_savings)
        .build_with_auto_env(&server)
        .await?;
    model.text("Ready.");
    test.submit_text_turn("Start the GUI check.").await?;
    let patch = format!(
        "*** Begin Patch\n{}*** End Patch",
        "+fixture line\n".repeat(200)
    );
    let items: Vec<ResponseItem> = serde_json::from_value(json!([
        {"type":"function_call","id":"fc_shot","call_id":"drop-shot","name":"js","arguments":"{}"},
        {"type":"function_call_output","id":"fco_shot","call_id":"drop-shot","output":[
            {"type":"input_text","text":"Screenshot of the settings page. ".repeat(10)},
            {"type":"input_image","image_url":SCREENSHOT}
        ]},
        {"type":"custom_tool_call","id":"ctc_patch","call_id":"shorten-patch","name":"apply_patch","input":patch},
        {"type":"custom_tool_call_output","id":"ctco_patch","call_id":"shorten-patch","output":"Success. Updated config.toml"}
    ]))?;
    test.codex.inject_response_items(items).await?;
    model.text("Noted.");
    test.submit_text_turn("Continue.").await?;
    finish_marking(&test, &model, /*expected_batches*/ 1).await?;
    let batch = model
        .bodies()
        .iter()
        .find_map(|body| analysis_payload(body, CLASSIFY))
        .expect("classification request");
    let summarize = batch["candidates"]
        .as_array()
        .unwrap()
        .iter()
        .map(|candidate| (candidate["id"].clone(), candidate["summarize_call"].clone()))
        .collect::<Vec<_>>();
    assert!(summarize.contains(&(json!("ctco_patch"), json!(true))));
    assert!(summarize.contains(&(json!("fco_shot"), Value::Null)));

    let mut status = test.codex.tool_cleanup_status().await?;
    for _ in 0..100 {
        if !status.marking {
            break;
        }
        tokio::time::sleep(std::time::Duration::from_millis(50)).await;
        status = test.codex.tool_cleanup_status().await?;
    }
    assert!(test.codex.apply_tool_cleanup().await?.released_tokens > 0);

    model.text("Using the cleaned view.");
    test.submit_text_turn("Continue with the cleaned view.")
        .await?;
    let body = model.ordinary_bodies().pop().unwrap();
    let input = body["input"].as_array().unwrap();
    let item = |kind: &str, call_id: &str| {
        input
            .iter()
            .find(|item| item["type"] == kind && item["call_id"] == call_id)
            .cloned()
            .expect("paired record stays in history")
    };
    let shot = item("function_call_output", "drop-shot");
    assert!(!shot.to_string().contains("input_image"));
    assert!(shot.to_string().contains("1 images"));
    let call = item("custom_tool_call", "shorten-patch");
    assert!(call["input"].as_str().unwrap().starts_with(CALL_SUMMARY));
    assert_eq!(call["name"], "apply_patch");
    // The short patch result stays as is; only its large call shrinks.
    assert_eq!(
        item("custom_tool_call_output", "shorten-patch")["output"],
        "Success. Updated config.toml"
    );

    // The model can still look at the original screenshot.
    model.reply(sse(vec![
        ev_function_call(
            "recall-shot",
            "recall_read_item",
            &json!({"item_id":"fco_shot"}).to_string(),
        ),
        ev_completed("recall-call"),
    ]));
    model.text("Screenshot reviewed.");
    test.submit_text_turn("Look at the earlier screenshot.")
        .await?;
    let body = model.ordinary_bodies().pop().unwrap();
    let recalled = body["input"]
        .as_array()
        .unwrap()
        .iter()
        .find(|item| item["type"] == "function_call_output" && item["call_id"] == "recall-shot")
        .expect("recall output")
        .to_string();
    assert!(recalled.contains("[image 1]"));
    assert!(recalled.contains("input_image") && recalled.contains(SCREENSHOT));
    Ok(())
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn tool_cleanup_trims_earlier_turn_reasoning_beyond_its_share() -> Result<()> {
    skip_if_no_network!(Ok(()));
    let server = start_mock_server().await;
    let model = LocalModel::mount(&server).await;
    let test = test_codex()
        .with_config(|config| {
            configure_unreached_savings(config);
            config.local_compaction.keep_reasoning_percent = 1;
        })
        .build_with_auto_env(&server)
        .await?;
    let thinking = "earlier deliberation ".repeat(800);
    for turn in ["first", "second"] {
        model.reply(sse(vec![
            ev_reasoning_item(&format!("reasoning-{turn}"), &["Thinking."], &[&thinking]),
            ev_assistant_message(&format!("answer-{turn}"), "Answered."),
            ev_completed(&format!("response-{turn}")),
        ]));
        test.submit_text_turn(&format!("Question {turn}.")).await?;
    }
    model.text("No deliberation needed.");
    test.submit_text_turn("Question third.").await?;
    let reasoning = |body: &Value| {
        body["input"]
            .as_array()
            .unwrap()
            .iter()
            .filter(|item| item["type"] == "reasoning")
            .count()
    };
    // The model keeps earlier reasoning, so later requests still carry both earlier turns'.
    assert!(
        model
            .ordinary_bodies()
            .last()
            .is_some_and(|body| reasoning(body) == 2)
    );
    let status = test.codex.tool_cleanup_status().await?;
    assert!(status.pending_savings_tokens > 0, "{status:?}");
    assert!(test.codex.apply_tool_cleanup().await?.released_tokens > 0);
    model.reply(sse(vec![
        ev_reasoning_item("reasoning-current", &["Thinking."], &[&thinking]),
        ev_function_call("current-call", "unavailable", "{}"),
        ev_completed("response-current"),
    ]));
    model.text("Continued.");
    test.submit_text_turn("Question fourth.").await?;
    // Earlier turns' reasoning is gone; the current turn's stays within its tool loop.
    assert_eq!(reasoning(model.ordinary_bodies().last().unwrap()), 1);
    Ok(())
}
