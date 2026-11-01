//! Replay, fork, and repeated cleanup through the production local pipeline.

use super::local_support::*;
use anyhow::Result;
use codex_core::TurnInputRequest;
use codex_protocol::protocol::Op;
use codex_protocol::user_input::UserInput;
use codex_rollout::recall::RecallArchive;
use core_test_support::responses::start_mock_server;
use core_test_support::skip_if_no_network;
use core_test_support::test_codex::test_codex;
use pretty_assertions::assert_eq;
use serde_json::Value;
use serde_json::json;
use std::io::Write as _;
use std::sync::Arc;

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn compact_resume_and_fork_preserve_installed_history_and_original_ids() -> Result<()> {
    skip_if_no_network!(Ok(()));
    let server = start_mock_server().await;
    let model = LocalModel::mount(&server).await;
    let mut builder = test_codex().with_config(configure_marking);
    let test = builder.build_with_auto_env(&server).await?;
    model.reply(tool_turn());
    model.text("Exact original assistant dialogue.");
    test.submit_text_turn("Original user instruction.").await?;
    model.text("Protected recent dialogue.");
    test.submit_text_turn("Keep the recent turn.").await?;
    finish_marking(&test, &model, /*expected_batches*/ 1).await?;
    model.text("Following installed history.");
    test.submit_text_turn("After cleanup.").await?;
    test.codex.flush_rollout().await?;
    let path = test.codex.rollout_path().unwrap();
    let installed = checkpoints(&path)?.pop().expect("installed view");
    let decisions = model.decisions();
    let resumed = test_codex()
        .with_config(configure_marking)
        .restart(&server, &test)
        .await?;
    model.text("Resumed.");
    resumed.submit_text_turn("After resume.").await?;
    let bodies = model.ordinary_bodies();
    let resumed_input = bodies.last().unwrap()["input"].as_array().unwrap();
    let before = &bodies[bodies.len() - 2]["input"];
    assert_eq!(
        resumed_input.get(..before.as_array().unwrap().len()),
        Some(before.as_array().unwrap().as_slice())
    );
    assert!(
        resumed_input
            .iter()
            .any(|item| item.to_string().contains(SHORTENED))
    );
    resumed.codex.flush_rollout().await?;
    assert_eq!(checkpoints(&path)?.last(), Some(&installed));
    let archive = RecallArchive::load(&path, Some(test.home.path())).await?;
    for decision in &decisions {
        let page = archive.query(serde_json::from_value(
            json!({"action":"read_item","item_id":decision["id"],"max_chars":500}),
        )?)?;
        assert_eq!(page["item_id"], decision["id"]);
        assert!(page["text"].as_str().unwrap().contains("unsupported call:"));
    }

    // A fork at the end replays the installed view and can resolve its original references.
    let forked = resumed
        .thread_manager
        .fork_legacy_thread(
            usize::MAX,
            codex_core::StartThreadOptions::new(resumed.config.clone()),
            path,
        )
        .await?;
    model.text("Fork continued.");
    forked
        .thread
        .start_or_steer_turn(TurnInputRequest::user_input(vec![UserInput::Text {
            text: "After fork.".to_string(),
            text_elements: Vec::new(),
        }]))
        .await?;
    complete(&forked.thread).await;
    assert!(
        model
            .ordinary_bodies()
            .last()
            .unwrap()
            .to_string()
            .contains(SHORTENED)
    );
    forked.thread.flush_rollout().await?;
    let fork_archive = RecallArchive::load(
        &forked.thread.rollout_path().unwrap(),
        Some(test.home.path()),
    )
    .await?;
    let original = fork_archive.query(serde_json::from_value(
        json!({"action":"read_item","item_id":decisions[1]["id"],"max_chars":500}),
    )?)?;
    assert_eq!(original["item_id"], decisions[1]["id"]);
    assert!(original["text"].as_str().unwrap().contains("evidence_"));
    Ok(())
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn repeated_tiers_preserve_constraints_recent_dialogue_and_resume() -> Result<()> {
    skip_if_no_network!(Ok(()));
    let server = start_mock_server().await;
    let model = LocalModel::mount(&server).await;
    let configure_tiers = |config: &mut codex_core::config::Config| {
        configure(config);
        config.local_compaction.compact_target_percent = 15;
    };
    let mut builder = test_codex().with_config(configure_tiers);
    let test = builder.build_with_auto_env(&server).await?;
    for cycle in 0..3 {
        for turn in 0..4 {
            model.text(&format!(
                "Evidence {cycle}-{turn}: {}",
                "Verification pending; earlier assumption corrected. ".repeat(200)
            ));
            test.submit_text_turn(&format!(
                "Offline-only constraint, cycle {cycle} turn {turn}: {}",
                "Preserve uncertainty and pending work. ".repeat(200)
            ))
            .await?;
        }
        model.text("Recent original answer must survive verbatim.");
        test.submit_text_turn("Recent original instruction must survive verbatim.")
            .await?;
        test.codex.submit(Op::Compact).await?;
        complete(&test.codex).await;
        model.text("Continue after promotion.");
        test.submit_text_turn("Read installed tiers.").await?;
        let bodies = model.ordinary_bodies();
        let installed = bodies.last().unwrap().to_string();
        assert!(installed.contains(LEDGER));
        assert!(installed.contains("Recent original instruction must survive verbatim."));
        assert!(installed.contains("Recent original answer must survive verbatim."));
        // One bounded summary replaces earlier summaries instead of accumulating per compaction.
        let summaries = bodies.last().unwrap()["input"]
            .as_array()
            .unwrap()
            .iter()
            .filter(|item| item.to_string().contains("Earlier conversation:"))
            .collect::<Vec<_>>();
        assert_eq!(summaries.len(), 1);
        for item in summaries {
            assert!(item.to_string().len() < 12_000);
        }
    }
    test.codex.flush_rollout().await?;
    let path = test.codex.rollout_path().unwrap();
    let before = checkpoints(&path)?;
    assert!(before.len() >= 3);
    let records = rollout(&path)?;
    let originals = records
        .iter()
        .filter_map(|record| match record {
            codex_history::RolloutItem::ResponseItem(item) => {
                item.item.id().map(ToString::to_string)
            }
            _ => None,
        })
        .collect::<Vec<_>>();
    let latest = records
        .iter()
        .rev()
        .find_map(|record| match record {
            codex_history::RolloutItem::Compacted(item) => item.replacement_history.as_ref(),
            _ => None,
        })
        .expect("latest replacement");
    let ranges = latest
        .iter()
        .filter_map(|item| item.metadata.as_ref()?.local_compaction.as_ref())
        .filter(|source| {
            matches!(
                source.kind,
                codex_history::LocalCompactionKind::OldestOverview
                    | codex_history::LocalCompactionKind::CondensedDialogue
            )
        })
        .map(|source| {
            let first = originals
                .iter()
                .position(|id| id == &source.first_item_id)
                .expect("range begins at an archived original");
            let last = originals
                .iter()
                .position(|id| id == &source.last_item_id)
                .expect("range ends at an archived original");
            assert!(first <= last, "source range is chronological");
            (first, last)
        })
        .collect::<Vec<_>>();
    assert!(!ranges.is_empty());
    assert!(
        ranges.windows(2).all(|ranges| ranges[0].1 < ranges[1].0),
        "tiers preserve disjoint chronological source ranges"
    );
    // Builder config mutators are consumed by the initial build; configure the new process too.
    let mut resume_builder = test_codex().with_config(configure_tiers);
    let resumed = resume_builder.restart(&server, &test).await?;
    assert_eq!(
        resumed.config.local_compaction,
        test.config.local_compaction
    );
    model.text("Resumed bounded history.");
    resumed
        .submit_text_turn("Continue after tier resume.")
        .await?;
    assert!(
        model
            .ordinary_bodies()
            .last()
            .unwrap()
            .to_string()
            .contains(LEDGER)
    );
    resumed.codex.flush_rollout().await?;
    assert_eq!(checkpoints(&path)?, before);
    // Listing uses the original turn archive, not only the small installed tier view.
    let archive = RecallArchive::load(&path, Some(test.home.path())).await?;
    let page = archive.query(serde_json::from_value(
        json!({"action":"list_turns","limit":2}),
    )?)?;
    assert_eq!(page["data"].as_array().unwrap().len(), 2);
    let offset = page["next_offset"].as_u64().expect("more original turns");
    let second = archive.query(serde_json::from_value(
        json!({"action":"list_turns","offset":offset,"limit":2}),
    )?)?;
    assert_ne!(page["data"], second["data"]);
    model.text(&"More original evidence. ".repeat(2000));
    resumed
        .submit_text_turn("New evidence after resume.")
        .await?;
    model.text("Recent protected answer.");
    resumed.submit_text_turn("New recent direction.").await?;
    resumed.codex.submit(Op::Compact).await?;
    complete(&resumed.codex).await;
    resumed.codex.flush_rollout().await?;
    assert!(checkpoints(&path)?.len() > before.len());
    Ok(())
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn rollback_discards_later_dialogue_after_local_cleanup() -> Result<()> {
    skip_if_no_network!(Ok(()));
    let server = start_mock_server().await;
    let model = LocalModel::mount(&server).await;
    let test = test_codex()
        .with_config(configure_marking)
        .build_with_auto_env(&server)
        .await?;
    model.reply(tool_turn());
    model.text("Retained answer.");
    test.submit_text_turn("Retained instruction.").await?;
    model.text("Recent answer.");
    test.submit_text_turn("Recent instruction.").await?;
    finish_marking(&test, &model, /*expected_batches*/ 1).await?;
    model.text("Discard this answer.");
    test.submit_text_turn("Discard this instruction.").await?;
    // Live rollback is unsupported; replay a persisted rollback marker over the installed
    // cleanup checkpoint, as older rollouts do.
    test.codex.flush_rollout().await?;
    let path = test.codex.rollout_path().unwrap();
    test.codex.shutdown_and_wait().await?;
    let marker = json!({"timestamp": "2026-01-01T00:00:00Z", "type": "event_msg", "payload": {
        "type": "thread_rolled_back", "num_turns": 1
    }});
    writeln!(
        std::fs::OpenOptions::new().append(true).open(&path)?,
        "{marker}"
    )?;
    let test = test_codex()
        .with_config(configure_marking)
        .resume(&server, Arc::clone(&test.home), path)
        .await?;
    model.text("Edited answer.");
    test.submit_text_turn("Edited instruction.").await?;
    let body: Value = model.ordinary_bodies().last().unwrap().clone();
    assert!(body.to_string().contains("Retained instruction."));
    assert!(body.to_string().contains(SHORTENED));
    assert!(!body.to_string().contains("Discard this instruction."));
    assert!(!body.to_string().contains("Discard this answer."));
    Ok(())
}
