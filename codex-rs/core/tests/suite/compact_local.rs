use super::local_support::*;
use anyhow::Result;
use codex_core::config::Config;
use codex_history::RolloutItem;
use codex_protocol::items::TurnItem;
use codex_protocol::protocol::EventMsg;
use codex_protocol::protocol::Op;
use codex_rollout::recall::RecallArchive;
use core_test_support::hooks::trust_discovered_hooks;
use core_test_support::responses::ev_completed;
use core_test_support::responses::ev_function_call;
use core_test_support::responses::sse;
use core_test_support::responses::start_mock_server;
use core_test_support::skip_if_no_network;
use core_test_support::test_codex::TestCodex;
use core_test_support::test_codex::test_codex;
use pretty_assertions::assert_eq;
use serde_json::Value;
use serde_json::json;

async fn seed_tools(test: &TestCodex, model: &LocalModel) -> Result<()> {
    model.reply(tool_turn());
    model.text("Evidence received; verification is still pending.");
    test.submit_turn("Keep the offline-only constraint.")
        .await?;
    model.text("Recent dialogue remains verbatim.");
    test.submit_turn("Continue, preserving exact evidence.")
        .await?;
    Ok(())
}

fn output<'a>(body: &'a Value, call_id: &str) -> &'a Value {
    body["input"]
        .as_array()
        .expect("input")
        .iter()
        .find(|item| item["type"] == "function_call_output" && item["call_id"] == call_id)
        .expect("paired tool result")
}

fn assert_dialogue(body: &Value) {
    let texts = body["input"]
        .as_array()
        .expect("input")
        .iter()
        .filter(|item| item["type"] == "message")
        .flat_map(|item| item["content"].as_array().into_iter().flatten())
        .filter_map(|content| content["text"].as_str())
        .collect::<Vec<_>>();
    let expected = [
        "Keep the offline-only constraint.",
        "Evidence received; verification is still pending.",
        "Continue, preserving exact evidence.",
        "Recent dialogue remains verbatim.",
    ];
    let positions = expected.map(|text| {
        texts
            .iter()
            .position(|candidate| *candidate == text)
            .expect("verbatim dialogue")
    });
    assert!(positions.windows(2).all(|pair| pair[0] < pair[1]));
    assert!(!body.to_string().contains("classifier-private"));
    assert!(!body.to_string().contains("\"decisions\""));
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn keep_shorten_drop_preserves_dialogue_pairs_and_recalls_originals() -> Result<()> {
    skip_if_no_network!(Ok(()));
    let server = start_mock_server().await;
    let model = LocalModel::mount(&server).await;
    let test = test_codex()
        .with_config(configure)
        .build_with_auto_env(&server)
        .await?;
    seed_tools(&test, &model).await?;
    test.codex.submit(Op::Compact).await?;
    let events = complete(&test.codex).await;
    model.text("Following installed view.");
    test.submit_turn("Inspect the cleaned view.").await?;
    let bodies = model.bodies();
    let installed = bodies.last().expect("follow-up");
    assert_dialogue(installed);
    let original = &bodies[2];
    assert_eq!(
        output(installed, "original-keep"),
        output(original, "original-keep")
    );
    assert!(
        output(installed, "original-shorten")
            .to_string()
            .contains(SHORTENED)
    );
    for call_id in ["original-shorten", "original-drop"] {
        assert!(
            output(installed, call_id)
                .to_string()
                .contains("recall_read_item")
        );
        assert!(
            !output(installed, call_id)
                .to_string()
                .contains(&"evidence_".repeat(1000))
        );
        assert_eq!(
            installed["input"]
                .as_array()
                .unwrap()
                .iter()
                .filter(|item| item["type"] == "function_call" && item["call_id"] == call_id)
                .count(),
            1
        );
    }
    assert!(
        events
            .iter()
            .any(|event| matches!(event, EventMsg::ContextCompacted(_)))
    );
    let starts = events
        .iter()
        .filter_map(|event| match event {
            EventMsg::ItemStarted(item) => match &item.item {
                TurnItem::ContextCompaction(item) => Some(item.id.clone()),
                _ => None,
            },
            _ => None,
        })
        .collect::<Vec<_>>();
    let ends = events
        .iter()
        .filter_map(|event| match event {
            EventMsg::ItemCompleted(item) => match &item.item {
                TurnItem::ContextCompaction(item) => Some(item.id.clone()),
                _ => None,
            },
            _ => None,
        })
        .collect::<Vec<_>>();
    assert_eq!(starts.len(), 1);
    assert_eq!(starts, ends);
    test.codex.flush_rollout().await?;
    let path = test.codex.rollout_path().expect("rollout");
    let archive = RecallArchive::load(&path, Some(test.home.path())).await?;
    let decisions = model.decisions();
    assert_eq!(decisions.len(), 4);
    let classifier = bodies
        .iter()
        .find(|body| analysis_payload(body, CLASSIFY).is_some())
        .expect("classification request");
    let visible_ids = classifier["input"]
        .as_array()
        .unwrap()
        .iter()
        .filter_map(|item| {
            let text = item["output"].as_str()?;
            let labelled = text.strip_prefix("LOCAL_COMPACTION_SOURCE\n")?;
            let header: Value = serde_json::from_str(labelled.split_once('\n')?.0).unwrap();
            Some(header["item_id"].clone())
        })
        .collect::<Vec<_>>();
    assert!(!installed.to_string().contains("LOCAL_COMPACTION_SOURCE"));
    for decision in &decisions {
        assert!(
            visible_ids.contains(&decision["id"]),
            "model must see a label on the actual evidence"
        );
        let query = serde_json::from_value(
            json!({"action":"read_item","item_id":decision["id"],"start_char":0,"max_chars":500}),
        )?;
        let page = archive.query(query)?;
        assert_eq!(page["item_id"], decision["id"]);
        assert!(page["text"].as_str().unwrap().contains("unsupported call:"));
        let cursor = page["next_char"]
            .as_u64()
            .expect("original output has another page");
        let next = archive.query(serde_json::from_value(json!({"action":"read_item","item_id":decision["id"],"start_char":cursor,"max_chars":500}))?)?;
        assert_eq!(next["start_char"], cursor);
        assert!(next["text"].as_str().unwrap().contains("evidence_"));
    }
    // The registered model tool must expose the same original record after installation.
    let recalled = &decisions[1]["id"];
    model.reply(sse(vec![
        ev_function_call(
            "recall-original",
            "recall_read_item",
            &json!({"item_id":recalled,"max_chars":500}).to_string(),
        ),
        ev_completed("recall-call"),
    ]));
    model.text("Original recovered.");
    test.submit_turn("Read the original evidence.").await?;
    let bodies = model.bodies();
    let response = output(bodies.last().unwrap(), "recall-original")["output"]
        .as_str()
        .expect("recall output");
    let response: Value = serde_json::from_str(response)?;
    assert_eq!(response["item_id"], *recalled);
    assert!(
        response["text"]
            .as_str()
            .unwrap()
            .contains("unsupported call:")
    );
    Ok(())
}

#[test_case::test_case(Analysis::Malformed; "malformed JSON")]
#[test_case::test_case(Analysis::ForeignId; "foreign source ID")]
#[test_case::test_case(Analysis::DuplicateId; "duplicate source ID")]
#[test_case::test_case(Analysis::Oversized; "oversized replacement")]
#[test_case::test_case(Analysis::HttpFailure; "model service failure")]
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn rejected_analysis_does_not_change_history(analysis: Analysis) -> Result<()> {
    skip_if_no_network!(Ok(()));
    let server = start_mock_server().await;
    let model = LocalModel::mount(&server).await;
    let test = test_codex()
        .with_config(|config| {
            configure(config);
            config.model_provider.request_max_retries = Some(0);
        })
        .build_with_auto_env(&server)
        .await?;
    seed_tools(&test, &model).await?;
    let before = model.bodies().last().unwrap().clone();
    model.analysis(analysis);
    test.codex.submit(Op::Compact).await?;
    let mut error = false;
    loop {
        match test.codex.next_event().await?.msg {
            EventMsg::Error(_) => error = true,
            EventMsg::ContextCompacted(_) => panic!("invalid analysis installed history"),
            EventMsg::TurnComplete(_) => break,
            _ => {}
        }
    }
    assert!(error, "analysis failure should be reported");
    model.text("Continue unchanged.");
    test.submit_turn("Continue after rejected cleanup.").await?;
    let after = model.bodies().last().unwrap().clone();
    assert_dialogue(&after);
    for call in ["original-keep", "original-shorten", "original-drop"] {
        assert_eq!(output(&before, call), output(&after, call));
    }
    test.codex.flush_rollout().await?;
    assert!(checkpoints(&test.codex.rollout_path().unwrap())?.is_empty());
    Ok(())
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn insufficient_cleanup_then_invalid_tiers_installs_nothing() -> Result<()> {
    skip_if_no_network!(Ok(()));
    let server = start_mock_server().await;
    let model = LocalModel::mount(&server).await;
    let test = test_codex()
        .with_config(|config| {
            configure(config);
            config.local_compaction.target_percent = 15;
        })
        .build_with_auto_env(&server)
        .await?;
    seed_tools(&test, &model).await?;
    let before = model.bodies().last().unwrap().clone();
    model.analysis(Analysis::InvalidSummary);
    test.codex.submit(Op::Compact).await?;
    loop {
        match test.codex.next_event().await?.msg {
            EventMsg::ContextCompacted(_) => {
                panic!("staged cleanup must not install before tiers validate")
            }
            EventMsg::TurnComplete(_) => break,
            _ => {}
        }
    }
    assert!(
        model
            .bodies()
            .iter()
            .any(|body| analysis_payload(body, SUMMARIZE).is_some())
    );
    model.text("Still using original evidence.");
    test.submit_turn("Inspect after failed tier analysis.")
        .await?;
    let after = model.bodies().last().unwrap().clone();
    for call in ["original-keep", "original-shorten", "original-drop"] {
        assert_eq!(output(&before, call), output(&after, call));
    }
    test.codex.flush_rollout().await?;
    assert!(checkpoints(&test.codex.rollout_path().unwrap())?.is_empty());
    Ok(())
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn custom_guidance_supplements_protocol_and_usage_is_durable() -> Result<()> {
    skip_if_no_network!(Ok(()));
    let server = start_mock_server().await;
    let model = LocalModel::mount(&server).await;
    let test = test_codex()
        .with_config(|config| {
            configure(config);
            config.compact_prompt = Some("Retain unresolved verification details.".to_string());
        })
        .build_with_auto_env(&server)
        .await?;
    seed_tools(&test, &model).await?;
    test.codex.submit(Op::Compact).await?;
    let events = complete(&test.codex).await;
    let bodies = model.bodies();
    let classifier = bodies
        .iter()
        .find(|body| analysis_payload(body, CLASSIFY).is_some())
        .unwrap();
    assert!(
        classifier
            .to_string()
            .contains("Retain unresolved verification details.")
    );
    assert!(
        events
            .iter()
            .any(|event| matches!(event, EventMsg::TokenCount(_)))
    );
    test.codex.flush_rollout().await?;
    let records = rollout(&test.codex.rollout_path().unwrap())?;
    let usage = records
        .iter()
        .find_map(|item| match item {
            RolloutItem::TokenUsageRecord(record)
                if record.response_id == "classifier-response" =>
            {
                Some(record)
            }
            _ => None,
        })
        .expect("classifier token usage persisted");
    assert_eq!(usage.usage.total_tokens, 17);
    let checkpoint = records
        .iter()
        .find_map(|item| match item {
            RolloutItem::Compacted(record) => Some(record),
            _ => None,
        })
        .unwrap();
    assert_eq!(
        checkpoint.compaction_response_id.as_deref(),
        Some("classifier-response")
    );
    assert_eq!(checkpoint.latest_token_usage_record.as_ref(), Some(usage));
    Ok(())
}

#[test_case::test_case(false; "hook matchers and post compact")]
#[test_case::test_case(true; "unsupported pre compact block")]
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn hooks_preserve_matchers_and_failure_semantics(block: bool) -> Result<()> {
    skip_if_no_network!(Ok(()));
    let server = start_mock_server().await;
    let model = LocalModel::mount(&server).await;
    let test = test_codex()
        .with_pre_build_hook(move |home| {
            if block {
                super::write_unsupported_blocking_pre_compact_hook(home);
            } else {
                super::write_matching_compact_hooks(home);
            }
        })
        .with_config(|config: &mut Config| {
            configure(config);
            trust_discovered_hooks(config);
        })
        .build_with_auto_env(&server)
        .await?;
    seed_tools(&test, &model).await?;
    test.codex.submit(Op::Compact).await?;
    let events = complete(&test.codex).await;
    assert!(
        events
            .iter()
            .any(|event| matches!(event, EventMsg::ContextCompacted(_)))
    );
    let log = if block {
        "pre_compact_block_log.jsonl"
    } else {
        "post_compact_manual_log.jsonl"
    };
    let inputs = super::read_hook_inputs(&test.home.path().join(log));
    assert_eq!(inputs.len(), 1);
    assert_eq!(inputs[0]["trigger"], "manual");
    assert_eq!(
        inputs[0]["hook_event_name"],
        if block { "PreCompact" } else { "PostCompact" }
    );
    if block {
        assert!(
            events
                .iter()
                .any(|event| matches!(event, EventMsg::HookCompleted(completed)
            if completed.run.status == codex_protocol::protocol::HookRunStatus::Failed))
        );
    } else {
        assert!(!test.home.path().join("pre_compact_auto_log.jsonl").exists());
    }
    Ok(())
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn automatic_cleanup_runs_before_followup_sampling_with_incoming_direction() -> Result<()> {
    skip_if_no_network!(Ok(()));
    let server = start_mock_server().await;
    let model = LocalModel::mount(&server).await;
    let test = test_codex()
        .with_config(|config| {
            configure(config);
            config.local_compaction.trigger_percent = 40;
            config.local_compaction.target_percent = 30;
        })
        .build_with_auto_env(&server)
        .await?;
    model.reply(tool_turn());
    model.text(&"Earlier observations remain provisional. ".repeat(3000));
    test.submit_turn("Keep the offline-only constraint.")
        .await?;
    model.text("Recent dialogue remains verbatim.");
    test.codex
        .start_or_steer_turn(codex_core::TurnInputRequest::user_input(vec![
            codex_protocol::user_input::UserInput::Text {
                text: "Continue, preserving exact evidence.".to_string(),
                text_elements: Vec::new(),
            },
        ]))
        .await?;
    super::assert_compaction_uses_turn_lifecycle_id(&test.codex).await;
    let bodies = model.bodies();
    let classify_index = bodies
        .iter()
        .position(|body| analysis_payload(body, CLASSIFY).is_some())
        .expect("automatic classifier");
    assert!(
        bodies[classify_index]
            .to_string()
            .contains("Continue, preserving exact evidence.")
    );
    let followup = bodies.last().unwrap();
    assert!(
        followup
            .to_string()
            .contains("Continue, preserving exact evidence.")
    );
    assert!(analysis_payload(followup, CLASSIFY).is_none());
    assert!(analysis_payload(followup, SUMMARIZE).is_none());
    assert!(followup.to_string().contains("recall_read_item"));
    test.codex.flush_rollout().await?;
    let path = test.codex.rollout_path().unwrap();
    let installed = checkpoints(&path)?;
    assert_eq!(installed.len(), 1);
    model.text("Tiny continuation.");
    test.submit_turn("Continue briefly.").await?;
    test.codex.flush_rollout().await?;
    assert_eq!(
        checkpoints(&path)?,
        installed,
        "no rewrite for insignificant growth"
    );
    Ok(())
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn cancellation_during_analysis_keeps_original_history() -> Result<()> {
    skip_if_no_network!(Ok(()));
    let server = start_mock_server().await;
    let model = LocalModel::mount(&server).await;
    let test = test_codex()
        .with_config(configure)
        .build_with_auto_env(&server)
        .await?;
    seed_tools(&test, &model).await?;
    let before = model.bodies().last().unwrap().clone();
    model.delay(std::time::Duration::from_secs(30));
    test.codex.submit(Op::Compact).await?;
    tokio::time::timeout(std::time::Duration::from_secs(10), async {
        loop {
            if model
                .bodies()
                .iter()
                .any(|body| analysis_payload(body, CLASSIFY).is_some())
            {
                break;
            }
            tokio::time::sleep(std::time::Duration::from_millis(10)).await;
        }
    })
    .await?;
    test.codex.submit(Op::Interrupt).await?;
    core_test_support::wait_for_event(&test.codex, |event| {
        matches!(event, EventMsg::TurnAborted(_))
    })
    .await;
    model.delay(std::time::Duration::ZERO);
    model.text("After cancellation.");
    test.submit_turn("New direction after cancellation.")
        .await?;
    let after = model.bodies().last().unwrap().clone();
    for call in ["original-keep", "original-shorten", "original-drop"] {
        assert_eq!(output(&before, call), output(&after, call));
    }
    assert_dialogue(&after);
    test.codex.flush_rollout().await?;
    assert!(checkpoints(&test.codex.rollout_path().unwrap())?.is_empty());
    Ok(())
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn short_history_manual_cleanup_completes_without_analysis_or_checkpoint() -> Result<()> {
    skip_if_no_network!(Ok(()));
    let server = start_mock_server().await;
    let model = LocalModel::mount(&server).await;
    let test = test_codex()
        .with_config(configure)
        .build_with_auto_env(&server)
        .await?;
    model.text("Short answer.");
    test.submit_turn("Short instruction.").await?;
    test.codex.submit(Op::Compact).await?;
    let events = complete(&test.codex).await;
    assert!(events.iter().any(|event| matches!(event, EventMsg::ItemStarted(item) if matches!(&item.item, TurnItem::ContextCompaction(_)))));
    assert!(events.iter().any(|event| matches!(event, EventMsg::ItemCompleted(item) if matches!(&item.item, TurnItem::ContextCompaction(_)))));
    assert_eq!(model.bodies().len(), 1);
    test.codex.flush_rollout().await?;
    assert!(checkpoints(&test.codex.rollout_path().unwrap())?.is_empty());
    Ok(())
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn new_user_direction_invalidates_private_staged_keep_decisions() -> Result<()> {
    skip_if_no_network!(Ok(()));
    let server = start_mock_server().await;
    let model = LocalModel::mount(&server).await;
    let test = test_codex()
        .with_config(configure)
        .build_with_auto_env(&server)
        .await?;
    seed_tools(&test, &model).await?;
    model.analysis(Analysis::Keep);
    test.codex.submit(Op::Compact).await?;
    complete(&test.codex).await;
    test.codex.submit(Op::Compact).await?;
    complete(&test.codex).await;
    assert_eq!(
        model
            .bodies()
            .iter()
            .filter(|body| analysis_payload(body, CLASSIFY).is_some())
            .count(),
        1
    );
    test.codex.flush_rollout().await?;
    assert!(checkpoints(&test.codex.rollout_path().unwrap())?.is_empty());
    model.text("New direction acknowledged.");
    test.submit_turn("The previous evidence may now be shortened.")
        .await?;
    model.analysis(Analysis::Clean);
    test.codex.submit(Op::Compact).await?;
    complete(&test.codex).await;
    assert_eq!(
        model
            .bodies()
            .iter()
            .filter(|body| analysis_payload(body, CLASSIFY).is_some())
            .count(),
        2
    );
    model.text("Using the newly assessed view.");
    test.submit_turn("Inspect the new view.").await?;
    let bodies = model.bodies();
    assert!(
        output(bodies.last().unwrap(), "original-shorten")
            .to_string()
            .contains(SHORTENED)
    );
    assert!(
        bodies
            .last()
            .unwrap()
            .to_string()
            .contains("The previous evidence may now be shortened.")
    );
    Ok(())
}
