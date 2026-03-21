use anyhow::Result;
use codex_protocol::protocol::EventMsg;
use codex_protocol::protocol::Op;
use codex_protocol::user_input::ByteRange;
use codex_protocol::user_input::TextElement;
use codex_protocol::user_input::UserInput;
use core_test_support::responses::ev_assistant_message;
use core_test_support::responses::ev_completed;
use core_test_support::responses::ev_reasoning_item;
use core_test_support::responses::ev_response_created;
use core_test_support::responses::mount_sse_sequence;
use core_test_support::responses::sse;
use core_test_support::responses::start_mock_server;
use core_test_support::skip_if_no_network;
use core_test_support::test_codex::test_codex;
use core_test_support::wait_for_event;
use pretty_assertions::assert_eq;
use std::sync::Arc;
use wiremock::matchers::body_string_contains;

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn resume_includes_initial_messages_from_rollout_events() -> Result<()> {
    skip_if_no_network!(Ok(()));

    let server = start_mock_server().await;
    let mut builder = test_codex();
    let initial = builder.build(&server).await?;
    let codex = Arc::clone(&initial.codex);
    let home = initial.home.clone();
    let rollout_path = initial
        .session_configured
        .rollout_path
        .clone()
        .expect("rollout path");

    let initial_mock = mount_sse_sequence(
        &server,
        vec![
            sse(vec![
                ev_response_created("resp-initial"),
                ev_assistant_message("msg-1", "Completed first turn"),
                ev_completed("resp-initial"),
            ]),
            sse(vec![
                ev_response_created("resp-summary"),
                ev_assistant_message("msg-summary", "record messages summary"),
                ev_completed("resp-summary"),
            ]),
        ],
    )
    .await;

    let text_elements = vec![TextElement::new(
        ByteRange { start: 0, end: 6 },
        Some("<note>".into()),
    )];

    codex
        .submit(Op::UserInput {
            items: vec![UserInput::Text {
                text: "Record some messages".into(),
                text_elements: text_elements.clone(),
            }],
            final_output_json_schema: None,
        })
        .await?;

    wait_for_event(&codex, |event| matches!(event, EventMsg::TurnComplete(_))).await;
    wait_for_event(&codex, |event| {
        matches!(
            event,
            EventMsg::ChatTreeNodeUpdated(update)
                if update.chat_tree.summary.as_deref() == Some("record messages summary")
        )
    })
    .await;

    let resumed = builder.resume(&server, home, rollout_path).await?;
    let initial_messages = resumed
        .session_configured
        .initial_messages
        .expect("expected initial messages to be present for resumed session");
    match initial_messages.as_slice() {
        [
            EventMsg::TurnStarted(started),
            EventMsg::UserMessage(first_user),
            EventMsg::TokenCount(_),
            EventMsg::AgentMessage(assistant_message),
            EventMsg::TokenCount(_),
            EventMsg::TurnComplete(completed),
            EventMsg::ChatTreeNodeUpdated(chat_tree_updated),
        ] => {
            assert_eq!(first_user.message, "Record some messages");
            assert_eq!(first_user.text_elements, text_elements);
            assert_eq!(assistant_message.message, "Completed first turn");
            assert_eq!(completed.turn_id, started.turn_id);
            assert_eq!(
                completed.last_agent_message.as_deref(),
                Some("Completed first turn")
            );
            assert_eq!(
                chat_tree_updated.chat_tree.node_id, completed.turn_id,
                "chat-tree update should reference the resumed turn",
            );
            assert_eq!(
                chat_tree_updated.chat_tree.summary.as_deref(),
                Some("record messages summary"),
            );
        }
        other => panic!("unexpected initial messages after resume: {other:#?}"),
    }
    assert_eq!(initial_mock.requests().len(), 2);

    Ok(())
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn resume_includes_initial_messages_from_reasoning_events() -> Result<()> {
    skip_if_no_network!(Ok(()));

    let server = start_mock_server().await;
    let mut builder = test_codex().with_config(|config| {
        config.show_raw_agent_reasoning = true;
    });
    let initial = builder.build(&server).await?;
    let codex = Arc::clone(&initial.codex);
    let home = initial.home.clone();
    let rollout_path = initial
        .session_configured
        .rollout_path
        .clone()
        .expect("rollout path");

    let initial_mock = mount_sse_sequence(
        &server,
        vec![
            sse(vec![
                ev_response_created("resp-initial"),
                ev_reasoning_item("reason-1", &["Summarized step"], &["raw detail"]),
                ev_assistant_message("msg-1", "Completed reasoning turn"),
                ev_completed("resp-initial"),
            ]),
            sse(vec![
                ev_response_created("resp-summary"),
                ev_assistant_message("msg-summary", "reasoning summary label"),
                ev_completed("resp-summary"),
            ]),
        ],
    )
    .await;

    codex
        .submit(Op::UserInput {
            items: vec![UserInput::Text {
                text: "Record reasoning messages".into(),
                text_elements: Vec::new(),
            }],
            final_output_json_schema: None,
        })
        .await?;

    wait_for_event(&codex, |event| matches!(event, EventMsg::TurnComplete(_))).await;
    wait_for_event(&codex, |event| {
        matches!(
            event,
            EventMsg::ChatTreeNodeUpdated(update)
                if update.chat_tree.summary.as_deref() == Some("reasoning summary label")
        )
    })
    .await;

    let resumed = builder.resume(&server, home, rollout_path).await?;
    let initial_messages = resumed
        .session_configured
        .initial_messages
        .expect("expected initial messages to be present for resumed session");
    match initial_messages.as_slice() {
        [
            EventMsg::TurnStarted(started),
            EventMsg::UserMessage(first_user),
            EventMsg::TokenCount(_),
            EventMsg::AgentReasoning(reasoning),
            EventMsg::AgentReasoningRawContent(raw),
            EventMsg::AgentMessage(assistant_message),
            EventMsg::TokenCount(_),
            EventMsg::TurnComplete(completed),
            EventMsg::ChatTreeNodeUpdated(chat_tree_updated),
        ] => {
            assert_eq!(first_user.message, "Record reasoning messages");
            assert_eq!(reasoning.text, "Summarized step");
            assert_eq!(raw.text, "raw detail");
            assert_eq!(assistant_message.message, "Completed reasoning turn");
            assert_eq!(completed.turn_id, started.turn_id);
            assert_eq!(
                completed.last_agent_message.as_deref(),
                Some("Completed reasoning turn")
            );
            assert_eq!(
                chat_tree_updated.chat_tree.node_id, completed.turn_id,
                "chat-tree update should reference the resumed turn",
            );
            assert_eq!(
                chat_tree_updated.chat_tree.summary.as_deref(),
                Some("reasoning summary label"),
            );
        }
        other => panic!("unexpected initial messages after resume: {other:#?}"),
    }
    assert_eq!(initial_mock.requests().len(), 2);

    Ok(())
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn turn_completion_emits_async_chat_tree_summary_update() -> Result<()> {
    skip_if_no_network!(Ok(()));

    let server = start_mock_server().await;
    let initial = test_codex().build(&server).await?;
    let codex = Arc::clone(&initial.codex);

    let requests = mount_sse_sequence(
        &server,
        vec![
            sse(vec![
                ev_response_created("resp-initial"),
                ev_assistant_message("msg-1", "Completed first turn"),
                ev_completed("resp-initial"),
            ]),
            sse(vec![
                ev_response_created("resp-summary"),
                ev_assistant_message("msg-summary", "branch label"),
                ev_completed("resp-summary"),
            ]),
        ],
    )
    .await;

    codex
        .submit(Op::UserInput {
            items: vec![UserInput::Text {
                text: "Record some messages".into(),
                text_elements: Vec::new(),
            }],
            final_output_json_schema: None,
        })
        .await?;

    wait_for_event(&codex, |event| matches!(event, EventMsg::TurnComplete(_))).await;
    let chat_tree_updated = wait_for_event(&codex, |event| {
        matches!(
            event,
            EventMsg::ChatTreeNodeUpdated(update)
                if update.chat_tree.summary.as_deref() == Some("branch label")
        )
    })
    .await;
    match chat_tree_updated {
        EventMsg::ChatTreeNodeUpdated(update) => {
            assert_eq!(update.chat_tree.summary.as_deref(), Some("branch label"));
        }
        other => panic!("unexpected event: {other:#?}"),
    }
    assert_eq!(requests.requests().len(), 2);

    Ok(())
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn resume_switches_models_preserves_base_instructions() -> Result<()> {
    skip_if_no_network!(Ok(()));

    let server = start_mock_server().await;
    let mut builder = test_codex().with_config(|config| {
        config.model = Some("gpt-5.2".to_string());
    });
    let initial = builder.build(&server).await?;
    let codex = Arc::clone(&initial.codex);
    let home = initial.home.clone();
    let rollout_path = initial
        .session_configured
        .rollout_path
        .clone()
        .expect("rollout path");

    let _initial_summary_mock = core_test_support::responses::mount_sse_once_match(
        &server,
        body_string_contains("Summarize this turn for a chat tree node."),
        sse(vec![
            ev_response_created("resp-summary"),
            ev_assistant_message("msg-summary", "initial summary"),
            ev_completed("resp-summary"),
        ]),
    )
    .await;
    let initial_mock = core_test_support::responses::mount_sse_once(
        &server,
        sse(vec![
            ev_response_created("resp-initial"),
            ev_assistant_message("msg-1", "Completed first turn"),
            ev_completed("resp-initial"),
        ]),
    )
    .await;

    codex
        .submit(Op::UserInput {
            items: vec![UserInput::Text {
                text: "Record initial instructions".into(),
                text_elements: Vec::new(),
            }],
            final_output_json_schema: None,
        })
        .await?;
    wait_for_event(&codex, |event| matches!(event, EventMsg::TurnComplete(_))).await;

    let initial_body = initial_mock.single_request().body_json();
    let initial_instructions = initial_body
        .get("instructions")
        .and_then(|v| v.as_str())
        .unwrap_or_default()
        .to_string();

    let _resumed_summary_mock_1 = core_test_support::responses::mount_sse_once_match(
        &server,
        body_string_contains("Summarize this turn for a chat tree node."),
        sse(vec![
            ev_response_created("resp-resume-summary-1"),
            ev_assistant_message("msg-resume-summary-1", "resume summary one"),
            ev_completed("resp-resume-summary-1"),
        ]),
    )
    .await;
    let _resumed_summary_mock_2 = core_test_support::responses::mount_sse_once_match(
        &server,
        body_string_contains("Summarize this turn for a chat tree node."),
        sse(vec![
            ev_response_created("resp-resume-summary-2"),
            ev_assistant_message("msg-resume-summary-2", "resume summary two"),
            ev_completed("resp-resume-summary-2"),
        ]),
    )
    .await;
    let resumed_mock = mount_sse_sequence(
        &server,
        vec![
            sse(vec![
                ev_response_created("resp-resume-1"),
                ev_assistant_message("msg-2", "Resumed turn"),
                ev_completed("resp-resume-1"),
            ]),
            sse(vec![
                ev_response_created("resp-resume-2"),
                ev_assistant_message("msg-3", "Second resumed turn"),
                ev_completed("resp-resume-2"),
            ]),
        ],
    )
    .await;

    let mut resume_builder = test_codex().with_config(|config| {
        config.model = Some("gpt-5.2-codex".to_string());
    });
    let resumed = resume_builder.resume(&server, home, rollout_path).await?;
    resumed
        .codex
        .submit(Op::UserInput {
            items: vec![UserInput::Text {
                text: "Resume with different model".into(),
                text_elements: Vec::new(),
            }],
            final_output_json_schema: None,
        })
        .await?;
    wait_for_event(&resumed.codex, |event| {
        matches!(event, EventMsg::TurnComplete(_))
    })
    .await;

    resumed
        .codex
        .submit(Op::UserInput {
            items: vec![UserInput::Text {
                text: "Second turn after resume".into(),
                text_elements: Vec::new(),
            }],
            final_output_json_schema: None,
        })
        .await?;
    wait_for_event(&resumed.codex, |event| {
        matches!(event, EventMsg::TurnComplete(_))
    })
    .await;

    let requests = resumed_mock.requests();
    assert_eq!(requests.len(), 2, "expected two resumed requests");

    let first_resumed = &requests[0];
    assert_eq!(first_resumed.instructions_text(), initial_instructions);
    let first_developer_texts = first_resumed.message_input_texts("developer");
    let first_model_switch_count = first_developer_texts
        .iter()
        .filter(|text| text.contains("<model_switch>"))
        .count();
    assert!(
        first_model_switch_count >= 1,
        "expected model switch message on first post-resume turn"
    );

    let second_resumed = &requests[1];
    assert_eq!(second_resumed.instructions_text(), initial_instructions);
    let second_developer_texts = second_resumed.message_input_texts("developer");
    let second_model_switch_count = second_developer_texts
        .iter()
        .filter(|text| text.contains("<model_switch>"))
        .count();
    assert_eq!(
        second_model_switch_count, 1,
        "did not expect duplicate model switch message after first post-resume turn"
    );

    Ok(())
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn resume_model_switch_is_not_duplicated_after_pre_turn_override() -> Result<()> {
    skip_if_no_network!(Ok(()));

    let server = start_mock_server().await;
    let mut builder = test_codex().with_config(|config| {
        config.model = Some("gpt-5.2".to_string());
    });
    let initial = builder.build(&server).await?;
    let codex = Arc::clone(&initial.codex);
    let home = initial.home.clone();
    let rollout_path = initial
        .session_configured
        .rollout_path
        .clone()
        .expect("rollout path");

    let _initial_summary_mock = core_test_support::responses::mount_sse_once_match(
        &server,
        body_string_contains("Summarize this turn for a chat tree node."),
        sse(vec![
            ev_response_created("resp-summary"),
            ev_assistant_message("msg-summary", "initial summary"),
            ev_completed("resp-summary"),
        ]),
    )
    .await;
    let initial_mock = core_test_support::responses::mount_sse_once(
        &server,
        sse(vec![
            ev_response_created("resp-initial"),
            ev_assistant_message("msg-1", "Completed first turn"),
            ev_completed("resp-initial"),
        ]),
    )
    .await;
    codex
        .submit(Op::UserInput {
            items: vec![UserInput::Text {
                text: "Record initial instructions".into(),
                text_elements: Vec::new(),
            }],
            final_output_json_schema: None,
        })
        .await?;
    wait_for_event(&codex, |event| matches!(event, EventMsg::TurnComplete(_))).await;
    let _ = initial_mock.single_request();

    let _resumed_summary_mock = core_test_support::responses::mount_sse_once_match(
        &server,
        body_string_contains("Summarize this turn for a chat tree node."),
        sse(vec![
            ev_response_created("resp-resume-summary"),
            ev_assistant_message("msg-resume-summary", "resume summary"),
            ev_completed("resp-resume-summary"),
        ]),
    )
    .await;
    let resumed_mock = core_test_support::responses::mount_sse_once(
        &server,
        sse(vec![
            ev_response_created("resp-resume"),
            ev_assistant_message("msg-2", "Resumed turn"),
            ev_completed("resp-resume"),
        ]),
    )
    .await;

    let mut resume_builder = test_codex().with_config(|config| {
        config.model = Some("gpt-5.2-codex".to_string());
    });
    let resumed = resume_builder.resume(&server, home, rollout_path).await?;
    resumed
        .codex
        .submit(Op::OverrideTurnContext {
            cwd: None,
            approval_policy: None,
            approvals_reviewer: None,
            sandbox_policy: None,
            windows_sandbox_level: None,
            model: Some("gpt-5.1-codex-max".to_string()),
            effort: None,
            summary: None,
            service_tier: None,
            collaboration_mode: None,
            personality: None,
        })
        .await?;
    resumed
        .codex
        .submit(Op::UserInput {
            items: vec![UserInput::Text {
                text: "first turn after override".into(),
                text_elements: Vec::new(),
            }],
            final_output_json_schema: None,
        })
        .await?;
    wait_for_event(&resumed.codex, |event| {
        matches!(event, EventMsg::TurnComplete(_))
    })
    .await;

    let request = resumed_mock.single_request();
    let developer_texts = request.message_input_texts("developer");
    let model_switch_count = developer_texts
        .iter()
        .filter(|text| text.contains("<model_switch>"))
        .count();
    assert_eq!(model_switch_count, 1);

    Ok(())
}
