//! Replay, fork, and repeated cleanup through the production local pipeline.

use super::compact::local_support::*;
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

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn snapshot_rollback_followup_turn_trims_context_updates() -> Result<()> {
    use codex_protocol::config_types::CollaborationMode;
    use codex_protocol::config_types::ModeKind;
    use codex_protocol::config_types::Settings;
    use codex_protocol::protocol::EventMsg;
    use codex_protocol::protocol::ThreadSettingsOverrides;
    use core_test_support::context_snapshot;
    use core_test_support::context_snapshot::ContextSnapshotOptions;
    use core_test_support::context_snapshot::ContextSnapshotRenderMode;
    use core_test_support::responses::ev_assistant_message;
    use core_test_support::responses::ev_completed;
    use core_test_support::responses::ev_response_created;
    use core_test_support::responses::mount_sse_sequence;
    use core_test_support::responses::sse;
    use core_test_support::test_codex::local_selections;
    use core_test_support::wait_for_event;
    use wiremock::MockServer;

    skip_if_no_network!(Ok(()));

    const MODEL: &str = "gpt-5.4";
    const TURN_ONE_USER: &str = "turn 1 user";
    const TURN_TWO_USER: &str = "turn 2 user";
    const FOLLOWUP_USER: &str = "follow-up user";
    const ROLLED_BACK_DEV_INSTRUCTIONS: &str = "ROLLED_BACK_DEV_INSTRUCTIONS";
    const PRETURN_CONTEXT_DIFF_CWD: &str = "PRETURN_CONTEXT_DIFF_CWD";

    let server = MockServer::start().await;
    let request_log = mount_sse_sequence(
        &server,
        vec![
            sse(vec![
                ev_assistant_message("m1", "turn 1 assistant"),
                ev_completed("r1"),
            ]),
            sse(vec![
                ev_assistant_message("m2", "turn 2 assistant"),
                ev_completed("r2"),
            ]),
            sse(vec![ev_response_created("r3"), ev_completed("r3")]),
        ],
    )
    .await;

    let test = test_codex()
        .with_model(MODEL)
        .with_config(|config| {
            config.update_plan_enabled = true;
            config.model_provider.name = "Non-OpenAI Model provider".to_string();
        })
        .build(&server)
        .await?;
    let config = &test.config;
    let conversation = &test.codex;

    test.submit_text_turn(TURN_ONE_USER).await?;

    let override_cwd = config.cwd.join(PRETURN_CONTEXT_DIFF_CWD);
    std::fs::create_dir_all(&override_cwd)?;
    core_test_support::submit_thread_settings(
        &conversation,
        ThreadSettingsOverrides {
            environments: Some(local_selections(override_cwd.clone())),
            collaboration_mode: Some(CollaborationMode {
                mode: ModeKind::Default,
                settings: Settings {
                    model: MODEL.to_string(),
                    reasoning_effort: None,
                    developer_instructions: Some(ROLLED_BACK_DEV_INSTRUCTIONS.to_string()),
                },
            }),
            ..Default::default()
        },
    )
    .await?;

    test.submit_text_turn(TURN_TWO_USER).await?;

    conversation
        .submit(Op::ThreadRollback { num_turns: 1 })
        .await?;
    let rollback_event = wait_for_event(&conversation, |ev| {
        matches!(ev, EventMsg::ThreadRolledBack(_))
    })
    .await;
    let EventMsg::ThreadRolledBack(rollback_event) = rollback_event else {
        panic!("expected thread rolled back event");
    };
    assert_eq!(rollback_event.num_turns, 1);

    test.submit_text_turn(FOLLOWUP_USER).await?;

    let requests = request_log.requests();
    assert_eq!(requests.len(), 3);

    let before_rollback_developer_count = requests[1]
        .message_input_texts("developer")
        .iter()
        .filter(|text| text.contains(ROLLED_BACK_DEV_INSTRUCTIONS))
        .count();
    assert_eq!(before_rollback_developer_count, 1);
    assert_eq!(
        requests[1]
            .message_input_texts("user")
            .iter()
            .filter(|text| text.contains(PRETURN_CONTEXT_DIFF_CWD))
            .count(),
        1
    );

    let after_rollback_developer_count = requests[2]
        .message_input_texts("developer")
        .iter()
        .filter(|text| text.contains(ROLLED_BACK_DEV_INSTRUCTIONS))
        .count();
    assert_eq!(after_rollback_developer_count, 1);

    let after_rollback_user_texts = requests[2].message_input_texts("user");
    assert_eq!(
        after_rollback_user_texts
            .iter()
            .filter(|text| text.contains(PRETURN_CONTEXT_DIFF_CWD))
            .count(),
        1
    );
    assert_eq!(
        after_rollback_user_texts.last().map(String::as_str),
        Some(FOLLOWUP_USER)
    );

    insta::assert_snapshot!(
        "rollback_followup_turn_trims_context_updates",
        context_snapshot::format_labeled_requests_snapshot(
            "rollback trims pre-turn override context updates before the follow-up request",
            &[
                ("rolled-back turn request", &requests[1]),
                ("follow-up request after rollback", &requests[2]),
            ],
            &ContextSnapshotOptions::default()
                .strip_capability_instructions()
                .render_mode(ContextSnapshotRenderMode::KindWithTextPrefix { max_chars: 96 }),
        )
    );

    Ok(())
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn compact_resume_and_fork_preserve_installed_history_and_original_ids() -> Result<()> {
    skip_if_no_network!(Ok(()));
    let server = start_mock_server().await;
    let model = LocalModel::mount(&server).await;
    let mut builder = test_codex().with_config(configure);
    let test = builder.build_with_auto_env(&server).await?;
    model.reply(tool_turn());
    model.text("Exact original assistant dialogue.");
    test.submit_turn("Original user instruction.").await?;
    model.text("Protected recent dialogue.");
    test.submit_turn("Keep the recent turn.").await?;
    test.codex.submit(Op::Compact).await?;
    complete(&test.codex).await;
    model.text("Following installed history.");
    test.submit_turn("After cleanup.").await?;
    test.codex.flush_rollout().await?;
    let path = test.codex.rollout_path().unwrap();
    let installed = checkpoints(&path)?.pop().expect("installed view");
    let decisions = model.decisions();
    let resumed = builder.restart(&server, &test).await?;
    model.text("Resumed.");
    resumed.submit_turn("After resume.").await?;
    let bodies = model.bodies();
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
        .fork_thread(
            usize::MAX,
            resumed.config.clone(),
            path,
            /*thread_source*/ None,
            /*parent_trace*/ None,
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
            .bodies()
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
        config.local_compaction.target_percent = 15;
    };
    let mut builder = test_codex().with_config(configure_tiers);
    let test = builder.build_with_auto_env(&server).await?;
    for cycle in 0..3 {
        for turn in 0..4 {
            model.text(&format!(
                "Evidence {cycle}-{turn}: {}",
                "Verification pending; earlier assumption corrected. ".repeat(200)
            ));
            test.submit_turn(&format!(
                "Offline-only constraint, cycle {cycle} turn {turn}: {}",
                "Preserve uncertainty and pending work. ".repeat(200)
            ))
            .await?;
        }
        model.text("Recent original answer must survive verbatim.");
        test.submit_turn("Recent original instruction must survive verbatim.")
            .await?;
        test.codex.submit(Op::Compact).await?;
        complete(&test.codex).await;
        model.text("Continue after promotion.");
        test.submit_turn("Read installed tiers.").await?;
        let bodies = model.bodies();
        let installed = bodies.last().unwrap().to_string();
        assert!(installed.contains(LEDGER));
        assert!(installed.contains("Recent original instruction must survive verbatim."));
        assert!(installed.contains("Recent original answer must survive verbatim."));
        assert!(!installed.contains("tiers-private"));
        // Each generated tier is bounded, and old promotions do not accumulate one entry per turn.
        let tier_items = bodies.last().unwrap()["input"]
            .as_array()
            .unwrap()
            .iter()
            .filter(|item| {
                item.to_string()
                    .contains("Earlier dialogue and concise evidence")
                    || item.to_string().contains("Oldest conversation overview")
            })
            .collect::<Vec<_>>();
        assert!(tier_items.len() <= 3);
        for item in tier_items {
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
    resumed.submit_turn("Continue after tier resume.").await?;
    assert!(model.bodies().last().unwrap().to_string().contains(LEDGER));
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
    resumed.submit_turn("New evidence after resume.").await?;
    model.text("Recent protected answer.");
    resumed.submit_turn("New recent direction.").await?;
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
        .with_config(configure)
        .build_with_auto_env(&server)
        .await?;
    model.reply(tool_turn());
    model.text("Retained answer.");
    test.submit_turn("Retained instruction.").await?;
    model.text("Recent answer.");
    test.submit_turn("Recent instruction.").await?;
    test.codex.submit(Op::Compact).await?;
    complete(&test.codex).await;
    model.text("Discard this answer.");
    test.submit_turn("Discard this instruction.").await?;
    test.codex
        .submit(Op::ThreadRollback { num_turns: 1 })
        .await?;
    core_test_support::wait_for_event(&test.codex, |event| {
        matches!(
            event,
            codex_protocol::protocol::EventMsg::ThreadRolledBack(_)
        )
    })
    .await;
    model.text("Edited answer.");
    test.submit_turn("Edited instruction.").await?;
    let body: Value = model.bodies().last().unwrap().clone();
    assert!(body.to_string().contains("Retained instruction."));
    assert!(body.to_string().contains(SHORTENED));
    assert!(!body.to_string().contains("Discard this instruction."));
    assert!(!body.to_string().contains("Discard this answer."));
    Ok(())
}
