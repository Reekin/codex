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
use core_test_support::responses::mount_sse_once;
use core_test_support::responses::mount_sse_sequence;
use core_test_support::responses::sse;
use core_test_support::responses::start_mock_server;
use core_test_support::skip_if_no_network;
use core_test_support::streaming_sse::StreamingSseChunk;
use core_test_support::streaming_sse::start_streaming_sse_server;
use core_test_support::test_codex::TestCodex;
use core_test_support::test_codex::TestCodexBuilder;
use core_test_support::test_codex::test_codex;
use core_test_support::wait_for_event;
use core_test_support::wait_for_event_match;
use pretty_assertions::assert_eq;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::Duration;
use tempfile::TempDir;
use tokio::sync::oneshot;
use wiremock::MockServer;

async fn resume_until_initial_messages(
    builder: &mut TestCodexBuilder,
    server: &MockServer,
    home: Arc<TempDir>,
    rollout_path: PathBuf,
    predicate: impl Fn(&[EventMsg]) -> bool,
) -> Result<TestCodex> {
    let deadline = tokio::time::Instant::now() + Duration::from_secs(2);
    let poll_interval = Duration::from_millis(10);
    let mut last_initial_messages = "<missing initial messages>".to_string();

    loop {
        let resumed = builder
            .resume(server, Arc::clone(&home), rollout_path.clone())
            .await?;
        if let Some(initial_messages) = resumed.session_configured.initial_messages.as_ref() {
            if predicate(initial_messages) {
                return Ok(resumed);
            }
            last_initial_messages = format!("{initial_messages:#?}");
        }

        if tokio::time::Instant::now() >= deadline {
            panic!(
                "timed out waiting for rollout resume messages to stabilize: {last_initial_messages}"
            );
        }

        drop(resumed);
        tokio::time::sleep(poll_interval).await;
    }
}

async fn submit_turn_and_capture_chat_tree_node(
    test: &TestCodex,
    server: &MockServer,
    response_id: &str,
    message_id: &str,
    prompt: &str,
    answer: &str,
) -> Result<String> {
    let mock = mount_sse_sequence(
        server,
        vec![
            sse(vec![
                ev_response_created(response_id),
                ev_assistant_message(message_id, answer),
                ev_completed(response_id),
            ]),
            sse(vec![
                ev_response_created(&format!("{response_id}-summary")),
                ev_assistant_message(
                    &format!("{message_id}-summary"),
                    &format!("summary for {prompt}"),
                ),
                ev_completed(&format!("{response_id}-summary")),
            ]),
        ],
    )
    .await;
    test.codex
        .submit(Op::UserInput {
            environments: None,
            items: vec![UserInput::Text {
                text: prompt.into(),
                text_elements: Vec::new(),
            }],
            final_output_json_schema: None,
            responsesapi_client_metadata: None,
            thread_settings: Default::default(),
        })
        .await?;

    let node_id = wait_for_event_match(&test.codex, |event| match event {
        EventMsg::ChatTreeNodeStarted(event) => Some(event.node_id.clone()),
        _ => None,
    })
    .await;
    wait_for_event(&test.codex, |event| {
        matches!(event, EventMsg::TurnComplete(_))
    })
    .await;
    wait_for_event(&test.codex, |event| {
        matches!(
            event,
            EventMsg::ChatTreeNodeSummaryUpdated(event) if event.node_id == node_id
        )
    })
    .await;
    assert_eq!(mock.requests().len(), 2);
    Ok(node_id)
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn completed_chat_tree_turn_generates_llm_summary() -> Result<()> {
    skip_if_no_network!(Ok(()));

    let server = start_mock_server().await;
    let mut builder = test_codex();
    let test = builder.build(&server).await?;
    let rollout_path = test
        .session_configured
        .rollout_path
        .clone()
        .expect("rollout path");
    let mock = mount_sse_sequence(
        &server,
        vec![
            sse(vec![
                ev_response_created("resp-chat-tree-summary-main"),
                ev_assistant_message(
                    "msg-chat-tree-summary-main",
                    "Implemented the frobnicator and added a regression test.",
                ),
                ev_completed("resp-chat-tree-summary-main"),
            ]),
            sse(vec![
                ev_response_created("resp-chat-tree-summary-title"),
                ev_assistant_message(
                    "msg-chat-tree-summary-title",
                    "\"Implement frobnicator with regression coverage\"",
                ),
                ev_completed("resp-chat-tree-summary-title"),
            ]),
        ],
    )
    .await;

    test.codex
        .submit(Op::UserInput {
            environments: None,
            items: vec![UserInput::Text {
                text: "Build the frobnicator".into(),
                text_elements: Vec::new(),
            }],
            final_output_json_schema: None,
            responsesapi_client_metadata: None,
            thread_settings: Default::default(),
        })
        .await?;

    let node_id = wait_for_event_match(&test.codex, |event| match event {
        EventMsg::ChatTreeNodeStarted(event) => Some(event.node_id.clone()),
        _ => None,
    })
    .await;
    wait_for_event(&test.codex, |event| {
        matches!(event, EventMsg::TurnComplete(_))
    })
    .await;
    let summary = wait_for_event_match(&test.codex, |event| match event {
        EventMsg::ChatTreeNodeSummaryUpdated(event) if event.node_id == node_id => {
            event.summary.clone()
        }
        _ => None,
    })
    .await;
    assert_eq!(summary, "Implement frobnicator with regression coverage");

    let requests = mock.requests();
    assert_eq!(requests.len(), 2);
    let summary_request = &requests[1];
    assert_eq!(
        summary_request.instructions_text(),
        "You generate a concise summary label for one completed assistant turn."
    );
    let summary_body = summary_request.body_json();
    assert_eq!(summary_body["tools"].as_array().map(Vec::len), Some(0));
    assert_eq!(summary_body["parallel_tool_calls"].as_bool(), Some(false));
    if let Some(reasoning) = summary_body
        .get("reasoning")
        .and_then(|value| value.as_object())
    {
        assert!(
            !reasoning.contains_key("summary"),
            "summary request should disable reasoning summaries: {reasoning:#?}"
        );
    }
    let summary_input = summary_request.message_input_texts("user").join("\n");
    assert!(summary_input.contains("User message:\nBuild the frobnicator"));
    assert!(
        summary_input.contains(
            "Assistant message:\nImplemented the frobnicator and added a regression test."
        )
    );

    let rollout = tokio::fs::read_to_string(rollout_path).await?;
    assert!(rollout.contains("Implement frobnicator with regression coverage"));

    Ok(())
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn cancelling_summary_before_completion_does_not_persist_update() -> Result<()> {
    skip_if_no_network!(Ok(()));

    let (summary_gate_tx, summary_gate_rx) = oneshot::channel();
    let summary_text = "summary should not persist after cancellation";
    let (server, _completions) = start_streaming_sse_server(vec![
        vec![StreamingSseChunk {
            gate: None,
            body: sse(vec![
                ev_response_created("resp-summary-cancel-main"),
                ev_assistant_message("msg-summary-cancel-main", "Main answer before shutdown."),
                ev_completed("resp-summary-cancel-main"),
            ]),
        }],
        vec![
            StreamingSseChunk {
                gate: None,
                body: sse(vec![ev_response_created("resp-summary-cancel-summary")]),
            },
            StreamingSseChunk {
                gate: Some(summary_gate_rx),
                body: sse(vec![
                    ev_assistant_message("msg-summary-cancel-summary", summary_text),
                    ev_completed("resp-summary-cancel-summary"),
                ]),
            },
        ],
    ])
    .await;
    let mut builder = test_codex();
    let test = builder.build_with_streaming_server(&server).await?;
    let rollout_path = test
        .session_configured
        .rollout_path
        .clone()
        .expect("rollout path");

    test.codex
        .submit(Op::UserInput {
            environments: None,
            items: vec![UserInput::Text {
                text: "Start a cancellable summary".into(),
                text_elements: Vec::new(),
            }],
            final_output_json_schema: None,
            responsesapi_client_metadata: None,
            thread_settings: Default::default(),
        })
        .await?;
    wait_for_event(&test.codex, |event| {
        matches!(event, EventMsg::TurnComplete(_))
    })
    .await;
    server.wait_for_request_count(2).await;

    test.codex.submit(Op::Shutdown).await?;
    wait_for_event(&test.codex, |event| {
        matches!(event, EventMsg::ShutdownComplete)
    })
    .await;
    let _ = summary_gate_tx.send(());

    let rollout = tokio::fs::read_to_string(rollout_path).await?;
    assert!(!rollout.contains(summary_text));
    server.shutdown().await;

    Ok(())
}

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

    let initial_sse = sse(vec![
        ev_response_created("resp-initial"),
        ev_assistant_message("msg-1", "Completed first turn"),
        ev_completed("resp-initial"),
    ]);
    mount_sse_once(&server, initial_sse).await;

    let text_elements = vec![TextElement::new(
        ByteRange { start: 0, end: 6 },
        Some("<note>".into()),
    )];

    codex
        .submit(Op::UserInput {
            environments: None,
            items: vec![UserInput::Text {
                text: "Record some messages".into(),
                text_elements: text_elements.clone(),
            }],
            final_output_json_schema: None,
            responsesapi_client_metadata: None,
            thread_settings: Default::default(),
        })
        .await?;

    wait_for_event(&codex, |event| matches!(event, EventMsg::TurnComplete(_))).await;

    let resumed = resume_until_initial_messages(
        &mut builder,
        &server,
        home,
        rollout_path,
        |initial_messages| {
            matches!(
                initial_messages,
                [
                    EventMsg::TurnStarted(_),
                    EventMsg::UserMessage(_),
                    EventMsg::AgentMessage(_),
                    EventMsg::TokenCount(_),
                    EventMsg::TurnComplete(_),
                ]
            )
        },
    )
    .await?;
    let initial_messages = resumed
        .session_configured
        .initial_messages
        .expect("expected initial messages to be present for resumed session");
    match initial_messages.as_slice() {
        [
            EventMsg::TurnStarted(started),
            EventMsg::UserMessage(first_user),
            EventMsg::AgentMessage(assistant_message),
            EventMsg::TokenCount(_),
            EventMsg::TurnComplete(completed),
        ] => {
            assert_eq!(first_user.message, "Record some messages");
            assert_eq!(first_user.text_elements, text_elements);
            assert_eq!(assistant_message.message, "Completed first turn");
            assert_eq!(completed.turn_id, started.turn_id);
            assert_eq!(
                completed.last_agent_message.as_deref(),
                Some("Completed first turn")
            );
        }
        other => panic!("unexpected initial messages after resume: {other:#?}"),
    }

    Ok(())
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn resume_chat_tree_branch_uses_current_node_history_for_next_request() -> Result<()> {
    skip_if_no_network!(Ok(()));

    let server = start_mock_server().await;
    let mut builder = test_codex();
    let initial = builder.build(&server).await?;
    let home = initial.home.clone();
    let rollout_path = initial
        .session_configured
        .rollout_path
        .clone()
        .expect("rollout path");

    let node_a = submit_turn_and_capture_chat_tree_node(
        &initial,
        &server,
        "resp-a",
        "msg-a",
        "chat tree prompt A",
        "chat tree answer A",
    )
    .await?;
    let _node_b = submit_turn_and_capture_chat_tree_node(
        &initial,
        &server,
        "resp-b",
        "msg-b",
        "chat tree prompt B",
        "chat tree answer B",
    )
    .await?;
    let _node_c = submit_turn_and_capture_chat_tree_node(
        &initial,
        &server,
        "resp-c",
        "msg-c",
        "chat tree prompt C",
        "chat tree answer C",
    )
    .await?;

    initial
        .codex
        .submit(Op::SetCurrentChatTreeNode {
            node_id: node_a.clone(),
            expected_revision: None,
        })
        .await?;
    wait_for_event(&initial.codex, |event| {
        matches!(event, EventMsg::ChatTreeCurrentNodeChanged(event) if event.node_id == node_a)
    })
    .await;

    let _node_d = submit_turn_and_capture_chat_tree_node(
        &initial,
        &server,
        "resp-d",
        "msg-d",
        "chat tree prompt D",
        "chat tree answer D",
    )
    .await?;

    let resumed_mock = mount_sse_once(
        &server,
        sse(vec![
            ev_response_created("resp-resume-chat-tree"),
            ev_assistant_message("msg-resume-chat-tree", "resumed chat tree answer"),
            ev_completed("resp-resume-chat-tree"),
        ]),
    )
    .await;
    let mut resume_builder = test_codex();
    let resumed = resume_builder.resume(&server, home, rollout_path).await?;
    resumed
        .codex
        .submit(Op::UserInput {
            environments: None,
            items: vec![UserInput::Text {
                text: "chat tree prompt after resume".into(),
                text_elements: Vec::new(),
            }],
            final_output_json_schema: None,
            responsesapi_client_metadata: None,
            thread_settings: Default::default(),
        })
        .await?;
    wait_for_event(&resumed.codex, |event| {
        matches!(event, EventMsg::TurnComplete(_))
    })
    .await;

    let request = resumed_mock.single_request();
    assert!(request.body_contains_text("chat tree prompt A"));
    assert!(request.body_contains_text("chat tree answer A"));
    assert!(request.body_contains_text("chat tree prompt D"));
    assert!(request.body_contains_text("chat tree answer D"));
    assert!(request.body_contains_text("chat tree prompt after resume"));
    assert!(!request.body_contains_text("chat tree prompt B"));
    assert!(!request.body_contains_text("chat tree answer B"));
    assert!(!request.body_contains_text("chat tree prompt C"));
    assert!(!request.body_contains_text("chat tree answer C"));

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

    let initial_sse = sse(vec![
        ev_response_created("resp-initial"),
        ev_reasoning_item("reason-1", &["Summarized step"], &["raw detail"]),
        ev_assistant_message("msg-1", "Completed reasoning turn"),
        ev_completed("resp-initial"),
    ]);
    mount_sse_once(&server, initial_sse).await;

    codex
        .submit(Op::UserInput {
            environments: None,
            items: vec![UserInput::Text {
                text: "Record reasoning messages".into(),
                text_elements: Vec::new(),
            }],
            final_output_json_schema: None,
            responsesapi_client_metadata: None,
            thread_settings: Default::default(),
        })
        .await?;

    wait_for_event(&codex, |event| matches!(event, EventMsg::TurnComplete(_))).await;

    let resumed = resume_until_initial_messages(
        &mut builder,
        &server,
        home,
        rollout_path,
        |initial_messages| {
            matches!(
                initial_messages,
                [
                    EventMsg::TurnStarted(_),
                    EventMsg::UserMessage(_),
                    EventMsg::AgentReasoning(_),
                    EventMsg::AgentReasoningRawContent(_),
                    EventMsg::AgentMessage(_),
                    EventMsg::TokenCount(_),
                    EventMsg::TurnComplete(_),
                ]
            )
        },
    )
    .await?;
    let initial_messages = resumed
        .session_configured
        .initial_messages
        .expect("expected initial messages to be present for resumed session");
    match initial_messages.as_slice() {
        [
            EventMsg::TurnStarted(started),
            EventMsg::UserMessage(first_user),
            EventMsg::AgentReasoning(reasoning),
            EventMsg::AgentReasoningRawContent(raw),
            EventMsg::AgentMessage(assistant_message),
            EventMsg::TokenCount(_),
            EventMsg::TurnComplete(completed),
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
        }
        other => panic!("unexpected initial messages after resume: {other:#?}"),
    }

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

    let initial_sse = sse(vec![
        ev_response_created("resp-initial"),
        ev_assistant_message("msg-1", "Completed first turn"),
        ev_completed("resp-initial"),
    ]);
    let initial_mock = mount_sse_once(&server, initial_sse).await;

    codex
        .submit(Op::UserInput {
            environments: None,
            items: vec![UserInput::Text {
                text: "Record initial instructions".into(),
                text_elements: Vec::new(),
            }],
            final_output_json_schema: None,
            responsesapi_client_metadata: None,
            thread_settings: Default::default(),
        })
        .await?;
    wait_for_event(&codex, |event| matches!(event, EventMsg::TurnComplete(_))).await;

    let initial_body = initial_mock.single_request().body_json();
    let initial_instructions = initial_body
        .get("instructions")
        .and_then(|v| v.as_str())
        .unwrap_or_default()
        .to_string();

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
        config.model = Some("gpt-5.3-codex".to_string());
    });
    let resumed = resume_builder.resume(&server, home, rollout_path).await?;
    resumed
        .codex
        .submit(Op::UserInput {
            environments: None,
            items: vec![UserInput::Text {
                text: "Resume with different model".into(),
                text_elements: Vec::new(),
            }],
            final_output_json_schema: None,
            responsesapi_client_metadata: None,
            thread_settings: Default::default(),
        })
        .await?;
    wait_for_event(&resumed.codex, |event| {
        matches!(event, EventMsg::TurnComplete(_))
    })
    .await;

    resumed
        .codex
        .submit(Op::UserInput {
            environments: None,
            items: vec![UserInput::Text {
                text: "Second turn after resume".into(),
                text_elements: Vec::new(),
            }],
            final_output_json_schema: None,
            responsesapi_client_metadata: None,
            thread_settings: Default::default(),
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

    let initial_mock = mount_sse_once(
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
            environments: None,
            items: vec![UserInput::Text {
                text: "Record initial instructions".into(),
                text_elements: Vec::new(),
            }],
            final_output_json_schema: None,
            responsesapi_client_metadata: None,
            thread_settings: Default::default(),
        })
        .await?;
    wait_for_event(&codex, |event| matches!(event, EventMsg::TurnComplete(_))).await;
    let _ = initial_mock.single_request();

    let resumed_mock = mount_sse_once(
        &server,
        sse(vec![
            ev_response_created("resp-resume"),
            ev_assistant_message("msg-2", "Resumed turn"),
            ev_completed("resp-resume"),
        ]),
    )
    .await;

    let mut resume_builder = test_codex().with_config(|config| {
        config.model = Some("gpt-5.3-codex".to_string());
    });
    let resumed = resume_builder.resume(&server, home, rollout_path).await?;
    core_test_support::submit_thread_settings(
        &resumed.codex,
        codex_protocol::protocol::ThreadSettingsOverrides {
            model: Some("gpt-5.4".to_string()),
            ..Default::default()
        },
    )
    .await?;
    resumed
        .codex
        .submit(Op::UserInput {
            environments: None,
            items: vec![UserInput::Text {
                text: "first turn after override".into(),
                text_elements: Vec::new(),
            }],
            final_output_json_schema: None,
            responsesapi_client_metadata: None,
            thread_settings: Default::default(),
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
