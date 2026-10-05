use super::local_support::*;
use anyhow::Result;
use codex_core::config::Config;
use codex_history::RolloutItem;
use codex_protocol::items::TurnItem;
use codex_protocol::protocol::EventMsg;
use codex_protocol::protocol::Op;
use codex_rollout::recall::RecallArchive;
use core_test_support::hooks::trust_discovered_hooks;
use core_test_support::responses::ev_assistant_message;
use core_test_support::responses::ev_completed;
use core_test_support::responses::ev_completed_with_tokens;
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
    test.submit_text_turn("Keep the offline-only constraint.")
        .await?;
    model.text("Recent dialogue remains verbatim.");
    test.submit_text_turn("Continue, preserving exact evidence.")
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
    let gate = MarkGate::start(&server).await?;
    let base_url = gate.base_url.clone();
    let test = test_codex()
        .with_config(move |config| {
            configure_marking(config);
            config.model_provider.base_url = Some(base_url);
        })
        .build_with_auto_env(&server)
        .await?;
    seed_tools(&test, &model).await?;
    gate.wait().await?;
    gate.release();
    finish_marking(&test, &model, /*expected_batches*/ 1).await?;
    model.text("Following installed view.");
    test.submit_text_turn("Inspect the cleaned view.").await?;
    let bodies = model.bodies();
    let installed = bodies.last().expect("follow-up");
    assert_dialogue(installed);
    let original = bodies
        .iter()
        .find(|body| {
            analysis_payload(body, CLASSIFY).is_none()
                && body["input"].as_array().unwrap().iter().any(|item| {
                    item["call_id"] == "original-keep" && item["type"] == "function_call_output"
                })
        })
        .unwrap();
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
    test.codex.flush_rollout().await?;
    let path = test.codex.rollout_path().expect("rollout");
    let archive = RecallArchive::load(&path, Some(test.home.path())).await?;
    let decisions = model.decisions();
    assert_eq!(decisions.len(), 4);
    let classifier = bodies
        .iter()
        .find(|body| analysis_payload(body, CLASSIFY).is_some())
        .expect("classification request");
    // The classifier repeats the ordinary request verbatim and only appends its question,
    // so the provider can serve the shared prefix from its prompt cache.
    let classifier_input = classifier["input"].as_array().unwrap();
    let prefix = &classifier_input[..classifier_input.len() - 1];
    assert!(
        bodies.iter().any(|body| {
            analysis_payload(body, CLASSIFY).is_none()
                && body["tools"] == classifier["tools"]
                && body["instructions"] == classifier["instructions"]
                && body["input"].as_array().unwrap().starts_with(prefix)
        }),
        "classifier must reuse an ordinary request prefix"
    );
    assert!(!classifier.to_string().contains("LOCAL_COMPACTION_SOURCE"));
    // Candidates are visible tool results, largest first.
    let candidates = analysis_payload(classifier, CLASSIFY).unwrap()["candidates"]
        .as_array()
        .unwrap()
        .clone();
    let sizes = candidates
        .iter()
        .map(|candidate| {
            prefix
                .iter()
                .find(|item| {
                    item["type"] == "function_call_output"
                        && item["call_id"] == candidate["call_id"]
                })
                .expect("candidate call ID is visible")["output"]
                .to_string()
                .len()
        })
        .collect::<Vec<_>>();
    assert!(sizes.windows(2).all(|pair| pair[0] >= pair[1]), "{sizes:?}");
    let candidate_ids = candidates
        .iter()
        .map(|candidate| candidate["id"].clone())
        .collect::<Vec<_>>();
    for decision in &decisions {
        assert!(candidate_ids.contains(&decision["id"]));
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
    test.submit_text_turn("Read the original evidence.").await?;
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

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn unreadable_analysis_does_not_change_history() -> Result<()> {
    skip_if_no_network!(Ok(()));
    let server = start_mock_server().await;
    let model = LocalModel::mount(&server).await;
    model.analysis(Analysis::Malformed);
    let gate = MarkGate::start(&server).await?;
    let base_url = gate.base_url.clone();
    let test = test_codex()
        .with_config(move |config| {
            configure_marking(config);
            config.model_provider.base_url = Some(base_url);
            config.model_provider.request_max_retries = Some(0);
        })
        .build_with_auto_env(&server)
        .await?;
    seed_tools(&test, &model).await?;
    gate.wait().await?;
    let before = model
        .bodies()
        .into_iter()
        .rev()
        .find(|body| analysis_payload(body, CLASSIFY).is_none())
        .unwrap();
    gate.release();
    marking_ready(&test.codex).await;
    model.text("Continue unchanged.");
    test.submit_text_turn("Continue after rejected cleanup.")
        .await?;
    let after = model.ordinary_bodies().last().unwrap().clone();
    assert_dialogue(&after);
    for call in ["original-keep", "original-shorten", "original-drop"] {
        assert_eq!(output(&before, call), output(&after, call));
    }
    test.codex.flush_rollout().await?;
    assert!(checkpoints(&test.codex.rollout_path().unwrap())?.is_empty());
    assert_eq!(
        model
            .bodies()
            .iter()
            .filter(|body| analysis_payload(body, CLASSIFY).is_some())
            .count(),
        1,
        "failed batch must not spin on unchanged IDs"
    );
    assert!(
        model
            .bodies()
            .iter()
            .all(|body| analysis_payload(body, SUMMARIZE).is_none()),
        "marking failure must not escalate into full compaction"
    );
    Ok(())
}

#[test_case::test_case(Analysis::ForeignId; "foreign source ID")]
#[test_case::test_case(Analysis::DuplicateId; "duplicate source ID")]
#[test_case::test_case(Analysis::Oversized; "oversized replacement")]
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn flawed_analysis_still_applies_its_usable_decisions(analysis: Analysis) -> Result<()> {
    skip_if_no_network!(Ok(()));
    let server = start_mock_server().await;
    let model = LocalModel::mount(&server).await;
    model.analysis(analysis);
    let test = test_codex()
        .with_config(configure_marking)
        .build_with_auto_env(&server)
        .await?;
    seed_tools(&test, &model).await?;
    finish_marking(&test, &model, /*expected_batches*/ 1).await?;
    model.text("Continue with the cleaned view.");
    test.submit_text_turn("Continue after partial cleanup.")
        .await?;
    let after = model.ordinary_bodies().last().unwrap().clone();
    assert_dialogue(&after);
    // The largest candidate carries the flaw; the other decisions still install.
    assert!(
        output(&after, "original-drop")
            .to_string()
            .contains("Output omitted"),
        "{}",
        output(&after, "original-drop")
    );
    assert!(
        output(&after, "original-keep")
            .to_string()
            .contains("keep_evidence_")
    );
    Ok(())
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn invalid_manual_tiers_installs_nothing() -> Result<()> {
    skip_if_no_network!(Ok(()));
    let server = start_mock_server().await;
    let model = LocalModel::mount(&server).await;
    let test = test_codex()
        .with_config(|config| {
            configure(config);
            config.local_compaction.compact_target_percent = 15;
        })
        .build_with_auto_env(&server)
        .await?;
    seed_tools(&test, &model).await?;
    let before = model.ordinary_bodies().last().unwrap().clone();
    model.analysis(Analysis::InvalidSummary);
    test.codex.submit(Op::Compact).await?;
    loop {
        match test.codex.next_event().await?.msg {
            EventMsg::ContextCompacted(_) => {
                panic!("full view must not install before tiers validate")
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
    test.submit_text_turn("Inspect after failed tier analysis.")
        .await?;
    let after = model.ordinary_bodies().last().unwrap().clone();
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
            configure_marking(config);
            config.compact_prompt = Some("Retain unresolved verification details.".to_string());
        })
        .build_with_auto_env(&server)
        .await?;
    seed_tools(&test, &model).await?;
    finish_marking(&test, &model, /*expected_batches*/ 1).await?;
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
    assert!(
        checkpoint
            .latest_token_usage_record
            .as_ref()
            .is_none_or(|record| record.response_id != "classifier-response"),
        "background billing must not replace foreground occupancy"
    );
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
    let starts = events
        .iter()
        .filter_map(|event| match event {
            EventMsg::ItemStarted(item) => match &item.item {
                TurnItem::ContextCompaction(item) => Some(&item.id),
                _ => None,
            },
            _ => None,
        })
        .collect::<Vec<_>>();
    let ends = events
        .iter()
        .filter_map(|event| match event {
            EventMsg::ItemCompleted(item) => match &item.item {
                TurnItem::ContextCompaction(item) => Some(&item.id),
                _ => None,
            },
            _ => None,
        })
        .collect::<Vec<_>>();
    assert_eq!(starts.len(), 1);
    assert_eq!(starts, ends);
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
async fn background_marking_overlaps_new_user_turn_and_keeps_one_job() -> Result<()> {
    skip_if_no_network!(Ok(()));
    let server = start_mock_server().await;
    let model = LocalModel::mount(&server).await;
    let gate = MarkGate::start(&server).await?;
    let base_url = gate.base_url.clone();
    let test = test_codex()
        .with_config(move |config| {
            configure_marking(config);
            config.model_provider.base_url = Some(base_url);
        })
        .build_with_auto_env(&server)
        .await?;
    seed_tools(&test, &model).await?;
    gate.wait().await?;
    let before = model
        .bodies()
        .into_iter()
        .rev()
        .find(|body| analysis_payload(body, CLASSIFY).is_none())
        .unwrap();
    model.reply(sse(vec![
        ev_assistant_message(
            "foreground-proof",
            "New direction completed while analysis is held.",
        ),
        ev_completed_with_tokens("foreground-proof", /*total_tokens*/ 1234),
    ]));
    test.codex
        .start_or_steer_turn(codex_core::TurnInputRequest::user_input(vec![
            codex_protocol::user_input::UserInput::Text {
                text: "New direction must not invalidate unchanged original IDs.".to_string(),
                text_elements: Vec::new(),
            },
        ]))
        .await?;
    let events =
        tokio::time::timeout(std::time::Duration::from_secs(10), complete(&test.codex)).await?;
    let foreground = events
        .iter()
        .rev()
        .find_map(|event| match event {
            EventMsg::TokenCount(count) => count.info.clone(),
            _ => None,
        })
        .expect("foreground usage");
    let bodies = model.bodies();
    let held = bodies.last().unwrap();
    assert!(
        held.to_string()
            .contains("New direction must not invalidate unchanged original IDs.")
    );
    for call in ["original-keep", "original-shorten", "original-drop"] {
        assert_eq!(output(&before, call), output(held, call));
    }
    assert_eq!(
        bodies
            .iter()
            .filter(|body| analysis_payload(body, CLASSIFY).is_some())
            .count(),
        1
    );
    let classifier = bodies
        .iter()
        .find(|body| analysis_payload(body, CLASSIFY).is_some())
        .unwrap();
    assert_eq!(classifier["model"], held["model"]);
    assert_eq!(classifier["reasoning"], held["reasoning"]);
    test.codex.flush_rollout().await?;
    assert!(checkpoints(&test.codex.rollout_path().unwrap())?.is_empty());
    gate.release();
    let background = core_test_support::wait_for_event_match(&test.codex, |event| {
        assert!(
            !matches!(event, EventMsg::RawResponseCompleted(_)),
            "background must not emit foreground completion"
        );
        match event {
            EventMsg::TokenCount(count) => count.info.clone(),
            _ => None,
        }
    })
    .await;
    assert_eq!(
        background.total_token_usage.total_tokens,
        foreground.total_token_usage.total_tokens + 17
    );
    assert_eq!(background.last_token_usage, foreground.last_token_usage);
    assert_eq!(
        background.model_context_window,
        foreground.model_context_window
    );
    assert_eq!(background.context_usage, foreground.context_usage);
    finish_marking(&test, &model, /*expected_batches*/ 1).await?;
    let bodies = model.bodies();
    assert!(
        output(bodies.last().unwrap(), "original-shorten")
            .to_string()
            .contains(SHORTENED)
    );
    assert_dialogue(bodies.last().unwrap());
    assert!(
        bodies
            .iter()
            .all(|body| analysis_payload(body, SUMMARIZE).is_none())
    );
    test.codex.flush_rollout().await?;
    let path = test.codex.rollout_path().unwrap();
    let installed = checkpoints(&path)?;
    assert_eq!(installed.len(), 1);
    model.text("Tiny continuation.");
    test.submit_text_turn("Continue briefly.").await?;
    test.codex.flush_rollout().await?;
    assert_eq!(
        checkpoints(&path)?,
        installed,
        "no rewrite for insignificant growth"
    );
    Ok(())
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn manual_full_compaction_cancels_old_background_results() -> Result<()> {
    skip_if_no_network!(Ok(()));
    let server = start_mock_server().await;
    let model = LocalModel::mount(&server).await;
    let gate = MarkGate::start(&server).await?;
    let base_url = gate.base_url.clone();
    let test = test_codex()
        .with_config(move |config| {
            configure_marking(config);
            config.model_provider.base_url = Some(base_url);
        })
        .build_with_auto_env(&server)
        .await?;
    seed_tools(&test, &model).await?;
    gate.wait().await?;
    test.codex.submit(Op::Compact).await?;
    complete(&test.codex).await;
    test.codex.flush_rollout().await?;
    let path = test.codex.rollout_path().unwrap();
    let installed = checkpoints(&path)?;
    assert_eq!(installed.len(), 1, "manual full compaction installed");
    assert!(
        model
            .bodies()
            .iter()
            .any(|body| analysis_payload(body, SUMMARIZE).is_some())
    );
    gate.release();
    for _ in 0..3 {
        model.text("Continue after the full window replacement.");
        test.submit_text_turn("Old classifier results must stay private.")
            .await?;
    }
    test.codex.flush_rollout().await?;
    assert_eq!(
        checkpoints(&path)?,
        installed,
        "late classifier must not rewrite the new full window"
    );
    assert!(
        !model
            .bodies()
            .last()
            .unwrap()
            .to_string()
            .contains(SHORTENED)
    );
    Ok(())
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn earlier_turn_background_billing_preserves_current_turn_cumulative_usage() -> Result<()> {
    use codex_protocol::request_user_input::RequestUserInputAnswer;
    use codex_protocol::request_user_input::RequestUserInputResponse;
    use std::collections::HashMap;

    skip_if_no_network!(Ok(()));
    let server = start_mock_server().await;
    let model = LocalModel::mount(&server).await;
    let gate = MarkGate::start(&server).await?;
    let base_url = gate.base_url.clone();
    let test = test_codex()
        .with_config(move |config| {
            configure_marking(config);
            config.model_provider.base_url = Some(base_url);
            let _ = config
                .features
                .enable(codex_features::Feature::DefaultModeRequestUserInput);
        })
        .build_with_auto_env(&server)
        .await?;
    seed_tools(&test, &model).await?;
    gate.wait().await?;
    model.reply(sse(vec![
        ev_function_call("billing-pause", "request_user_input", &json!({"questions":[{
            "id":"continue","header":"Continue","question":"Continue the same turn?",
            "options":[{"label":"Yes (Recommended)","description":"Continue."},{"label":"No","description":"Stop."}]
        }]}).to_string()),
        ev_completed_with_tokens("current-turn-first", /*total_tokens*/ 1234),
    ]));
    test.codex
        .start_or_steer_turn(codex_core::TurnInputRequest::user_input(vec![
            codex_protocol::user_input::UserInput::Text {
                text: "Pause between two responses of this turn.".to_string(),
                text_elements: Vec::new(),
            },
        ]))
        .await?;
    let pause = core_test_support::wait_for_event_match(&test.codex, |event| match event {
        EventMsg::RequestUserInput(request) => Some(request.clone()),
        _ => None,
    })
    .await;
    gate.release();
    marking_ready(&test.codex).await;
    model.reply(sse(vec![
        ev_assistant_message("billing-done", "Continued the same turn."),
        ev_completed_with_tokens("current-turn-second", /*total_tokens*/ 100),
    ]));
    test.codex
        .submit(Op::UserInputAnswer {
            id: pause.turn_id.clone(),
            response: RequestUserInputResponse {
                answers: HashMap::from([(
                    "continue".to_string(),
                    RequestUserInputAnswer {
                        answers: vec!["Yes (Recommended)".to_string()],
                    },
                )]),
            },
        })
        .await?;
    complete(&test.codex).await;
    test.codex.flush_rollout().await?;
    let records = rollout(&test.codex.rollout_path().unwrap())?;
    let usage = records
        .iter()
        .filter_map(|record| match record {
            RolloutItem::TokenUsageRecord(record) => Some(record),
            _ => None,
        })
        .collect::<Vec<_>>();
    let earlier = usage
        .iter()
        .find(|record| record.response_id == "classifier-response")
        .unwrap();
    let second = usage
        .iter()
        .find(|record| record.response_id == "current-turn-second")
        .unwrap();
    assert_ne!(earlier.turn_id, second.turn_id);
    assert_eq!(second.turn_id, pause.turn_id);
    assert_eq!(earlier.turn_token_usage.total_tokens, 17);
    assert_eq!(second.turn_token_usage.total_tokens, 1334);
    assert_eq!(second.thread_token_usage.total_tokens, 1351);
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
    test.submit_text_turn("Short instruction.").await?;
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
async fn keeps_are_reconsidered_after_new_user_input() -> Result<()> {
    skip_if_no_network!(Ok(()));
    let server = start_mock_server().await;
    let model = LocalModel::mount(&server).await;
    model.analysis(Analysis::Keep);
    let test = test_codex()
        .with_config(configure_marking)
        .build_with_auto_env(&server)
        .await?;
    seed_tools(&test, &model).await?;
    finish_marking(&test, &model, /*expected_batches*/ 1).await?;
    let kept = model
        .decisions()
        .iter()
        .map(|decision| decision["id"].clone())
        .collect::<Vec<_>>();
    assert_eq!(kept.len(), 4);
    // Without new user input the keeps hold and nothing is classified again.
    test.codex.flush_rollout().await?;
    assert_eq!(model.decisions().len(), kept.len());
    model.text("New direction acknowledged.");
    test.submit_text_turn("The task has moved on.").await?;
    finish_marking(&test, &model, /*expected_batches*/ 2).await?;
    let batches = model
        .bodies()
        .iter()
        .filter_map(|body| analysis_payload(body, CLASSIFY))
        .collect::<Vec<_>>();
    assert!(batches.len() >= 2);
    let second: Vec<_> = batches[1]["candidates"]
        .as_array()
        .unwrap()
        .iter()
        .map(|candidate| candidate["id"].clone())
        .collect();
    assert!(kept.iter().all(|id| second.contains(id)), "{second:?}");
    Ok(())
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn network_failure_is_nonfatal_and_does_not_retry_unchanged_batch() -> Result<()> {
    skip_if_no_network!(Ok(()));
    let server = start_mock_server().await;
    let model = LocalModel::mount(&server).await;
    model.analysis(Analysis::HttpFailure);
    let gate = MarkGate::start(&server).await?;
    let base_url = gate.base_url.clone();
    let test = test_codex()
        .with_config(move |config| {
            configure_marking(config);
            config.model_provider.base_url = Some(base_url);
            config.model_provider.request_max_retries = Some(0);
        })
        .build_with_auto_env(&server)
        .await?;
    seed_tools(&test, &model).await?;
    gate.wait().await?;
    gate.release();
    marking_ready(&test.codex).await;
    for _ in 0..8 {
        model.text("Foreground succeeds despite unavailable classification.");
        test.codex
            .start_or_steer_turn(codex_core::TurnInputRequest::user_input(vec![
                codex_protocol::user_input::UserInput::Text {
                    text: "Continue normally.".to_string(),
                    text_elements: Vec::new(),
                },
            ]))
            .await?;
        complete(&test.codex).await;
    }
    let bodies = model.bodies();
    assert_eq!(
        bodies
            .iter()
            .filter(|body| analysis_payload(body, CLASSIFY).is_some())
            .count(),
        1
    );
    assert!(
        bodies
            .iter()
            .all(|body| analysis_payload(body, SUMMARIZE).is_none())
    );
    assert_dialogue(bodies.last().unwrap());
    test.codex.flush_rollout().await?;
    assert!(checkpoints(&test.codex.rollout_path().unwrap())?.is_empty());
    Ok(())
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn compatible_budget_change_applies_disjoint_cleanup_without_cancelling_batch() -> Result<()>
{
    use codex_protocol::openai_models::ModelsResponse;
    use codex_protocol::protocol::ThreadSettingsOverrides;

    skip_if_no_network!(Ok(()));
    let server = start_mock_server().await;
    let model = LocalModel::mount(&server).await;
    let gate = MarkGate::start(&server).await?;
    let base_url = gate.base_url.clone();
    let test = test_codex()
        .with_model("wide-window")
        .with_config(move |config| {
            configure_marking(config);
            // A's reductions (about 5k) fall short of 6k at 100k but reach 4.2k at 70k.
            config.local_compaction.reclaim_percent = 6;
            config.model_context_window = None;
            let mut wide = codex_models_manager::bundled_models_response()
                .unwrap()
                .models
                .into_iter()
                .find(|model| model.slug == "gpt-5.4")
                .unwrap();
            wide.slug = "wide-window".to_string();
            wide.context_window = Some(100_000);
            wide.effective_context_window_percent = 100;
            wide.comp_hash = Some("compatible-local-window".to_string());
            let mut narrow = wide.clone();
            narrow.slug = "narrow-window".to_string();
            narrow.context_window = Some(70_000);
            config.model_catalog = Some(ModelsResponse {
                models: vec![wide, narrow],
            });
            config.model_provider.base_url = Some(base_url);
        })
        .build_with_auto_env(&server)
        .await?;
    seed_tools(&test, &model).await?;
    gate.wait().await?;
    gate.release();
    marking_ready(&test.codex).await;
    finish_marking(&test, &model, /*expected_batches*/ 1).await?;
    test.codex.flush_rollout().await?;
    let path = test.codex.rollout_path().unwrap();
    assert!(
        checkpoints(&path)?.is_empty(),
        "about 5k savings is below 6k required savings"
    );

    gate.hold();
    let mut calls = ["keep", "shorten", "drop"]
        .into_iter()
        .map(|name| {
            ev_function_call(
                &format!("next-{name}"),
                &format!("next_{name}_{}", "evidence_".repeat(1000)),
                "{}",
            )
        })
        .collect::<Vec<_>>();
    calls.push(ev_completed("next-tools"));
    model.reply(sse(calls));
    model.text("Second batch tools completed.");
    test.submit_text_turn("Collect a disjoint batch.").await?;
    model.text("Second batch is now eligible.");
    test.submit_text_turn("Continue after collecting the next batch.")
        .await?;
    gate.wait().await?;
    core_test_support::submit_thread_settings(
        &test.codex,
        ThreadSettingsOverrides {
            model: Some("narrow-window".to_string()),
            ..Default::default()
        },
    )
    .await?;
    model.text("Continue within the compatible smaller window.");
    test.codex
        .start_or_steer_turn(codex_core::TurnInputRequest::user_input(vec![
            codex_protocol::user_input::UserInput::Text {
                text: "Use the current window budget.".to_string(),
                text_elements: Vec::new(),
            },
        ]))
        .await?;
    let events = complete(&test.codex).await;
    let usage = events
        .iter()
        .rev()
        .find_map(|event| match event {
            EventMsg::TokenCount(count) => count.info.as_ref(),
            _ => None,
        })
        .expect("selected model usage");
    assert_eq!(usage.model_context_window, Some(70_000));
    let held = model.ordinary_bodies().last().unwrap().clone();
    assert_eq!(held["model"], "narrow-window");
    assert!(
        output(&held, "original-shorten")
            .to_string()
            .contains(SHORTENED)
    );
    assert!(
        !output(&held, "next-shorten")
            .to_string()
            .contains(SHORTENED)
    );
    test.codex.flush_rollout().await?;
    assert_eq!(
        checkpoints(&path)?.len(),
        1,
        "A installs while disjoint B is held"
    );
    gate.release();
    marking_ready(&test.codex).await;
    finish_marking(&test, &model, /*expected_batches*/ 2).await?;
    let bodies = model.bodies();
    assert!(
        output(bodies.last().unwrap(), "next-shorten")
            .to_string()
            .contains(SHORTENED)
    );
    assert_eq!(
        bodies
            .iter()
            .filter(|body| analysis_payload(body, CLASSIFY).is_some())
            .count(),
        2,
        "B result survives disjoint installation without being relaunched"
    );
    assert!(
        bodies
            .iter()
            .all(|body| analysis_payload(body, SUMMARIZE).is_none())
    );
    test.codex.flush_rollout().await?;
    assert_eq!(checkpoints(&path)?.len(), 2);
    Ok(())
}
