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
use core_test_support::test_codex::test_codex;
use core_test_support::wait_for_event;
use core_test_support::wait_for_event_match;
use pretty_assertions::assert_eq;
use std::sync::Arc;
use std::time::Duration;
use tokio::sync::oneshot;

async fn wait_for_request_count(
    mock: &core_test_support::responses::ResponseMock,
    expected: usize,
) {
    tokio::time::timeout(Duration::from_secs(10), async {
        loop {
            if mock.requests().len() >= expected {
                return;
            }
            tokio::time::sleep(Duration::from_millis(10)).await;
        }
    })
    .await
    .expect("timeout waiting for response requests");
}

async fn wait_for_turn_complete_and_summary(test: &TestCodex, node_id: &str) {
    let mut saw_turn_complete = false;
    let mut saw_summary = false;
    wait_for_event(&test.codex, |event| {
        match event {
            EventMsg::TurnComplete(_) => saw_turn_complete = true,
            EventMsg::ChatTreeNodeSummaryUpdated(event) if event.node_id == node_id => {
                saw_summary = true;
            }
            _ => {}
        }
        saw_turn_complete && saw_summary
    })
    .await;
}

async fn submit_turn_and_capture_chat_tree_node(
    test: &TestCodex,
    server: &wiremock::MockServer,
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
            items: vec![UserInput::Text {
                text: prompt.into(),
                text_elements: Vec::new(),
            }],
            final_output_json_schema: None,
            responsesapi_client_metadata: None,
            additional_context: Default::default(),
            thread_settings: Default::default(),
        })
        .await?;

    let node_id = wait_for_event_match(&test.codex, |event| match event {
        EventMsg::ChatTreeNodeStarted(event) => Some(event.node_id.clone()),
        _ => None,
    })
    .await;
    wait_for_turn_complete_and_summary(test, &node_id).await;
    assert_eq!(mock.requests().len(), 2);
    Ok(node_id)
}

async fn assert_summary_response_is_non_fatal(summary_response: String) -> Result<()> {
    let server = start_mock_server().await;
    let mut builder = test_codex();
    let initial = builder.build(&server).await?;
    let home = Arc::clone(&initial.home);
    let rollout_path = initial
        .session_configured
        .rollout_path
        .clone()
        .expect("rollout path");
    let mock = mount_sse_sequence(
        &server,
        vec![
            sse(vec![
                ev_response_created("resp-summary-non-fatal-main"),
                ev_assistant_message("msg-summary-non-fatal-main", "Main answer survives."),
                ev_completed("resp-summary-non-fatal-main"),
            ]),
            summary_response,
        ],
    )
    .await;

    initial
        .codex
        .submit(Op::UserInput {
            items: vec![UserInput::Text {
                text: "Generate a non-fatal summary".into(),
                text_elements: Vec::new(),
            }],
            final_output_json_schema: None,
            responsesapi_client_metadata: None,
            additional_context: Default::default(),
            thread_settings: Default::default(),
        })
        .await?;
    wait_for_event(&initial.codex, |event| {
        matches!(event, EventMsg::TurnComplete(_))
    })
    .await;
    wait_for_request_count(&mock, 2).await;
    initial.codex.shutdown_and_wait().await?;
    assert_eq!(mock.requests().len(), 2);

    let resumed = builder.resume(&server, home, rollout_path).await?;
    let projection = resumed.codex.chat_tree_projection().await;
    assert_eq!(projection.nodes.len(), 1);
    assert_eq!(
        projection.nodes[0].summary.as_deref(),
        Some("Turn 1 · completed")
    );
    Ok(())
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn completed_chat_tree_turn_generates_and_replays_llm_summary() -> Result<()> {
    skip_if_no_network!(Ok(()));

    let server = start_mock_server().await;
    let mut builder = test_codex();
    let initial = builder.build(&server).await?;
    let rollout_path = initial
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

    initial
        .codex
        .submit(Op::UserInput {
            items: vec![UserInput::Text {
                text: "Build the frobnicator".into(),
                text_elements: Vec::new(),
            }],
            final_output_json_schema: None,
            responsesapi_client_metadata: None,
            additional_context: Default::default(),
            thread_settings: Default::default(),
        })
        .await?;

    let node_id = wait_for_event_match(&initial.codex, |event| match event {
        EventMsg::ChatTreeNodeStarted(event) => Some(event.node_id.clone()),
        _ => None,
    })
    .await;
    wait_for_event(&initial.codex, |event| {
        matches!(event, EventMsg::TurnComplete(_))
    })
    .await;
    let summary = wait_for_event_match(&initial.codex, |event| match event {
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
    let summary_input = summary_request.message_input_texts("user").join("\n");
    assert!(summary_input.contains("User message:\nBuild the frobnicator"));
    assert!(
        summary_input.contains(
            "Assistant message:\nImplemented the frobnicator and added a regression test."
        )
    );

    let rollout = tokio::fs::read_to_string(&rollout_path).await?;
    assert!(rollout.contains("Implement frobnicator with regression coverage"));

    let resumed = builder.restart(&server, &initial).await?;
    let projection = resumed.codex.chat_tree_projection().await;
    assert_eq!(
        projection.current_node_id.as_deref(),
        Some(node_id.as_str())
    );
    assert_eq!(projection.nodes.len(), 1);
    assert_eq!(
        projection.nodes[0].summary.as_deref(),
        Some("Implement frobnicator with regression coverage")
    );

    Ok(())
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn failed_and_empty_chat_tree_summaries_do_not_fail_turn() -> Result<()> {
    skip_if_no_network!(Ok(()));

    assert_summary_response_is_non_fatal("data: not-json\n\n".to_string()).await?;
    assert_summary_response_is_non_fatal(sse(vec![
        ev_response_created("resp-summary-empty"),
        ev_assistant_message("msg-summary-empty", "\"\""),
        ev_completed("resp-summary-empty"),
    ]))
    .await?;

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
    let initial = builder.build_with_streaming_server(&server).await?;
    let rollout_path = initial
        .session_configured
        .rollout_path
        .clone()
        .expect("rollout path");

    initial
        .codex
        .submit(Op::UserInput {
            items: vec![UserInput::Text {
                text: "Start a cancellable summary".into(),
                text_elements: Vec::new(),
            }],
            final_output_json_schema: None,
            responsesapi_client_metadata: None,
            additional_context: Default::default(),
            thread_settings: Default::default(),
        })
        .await?;
    wait_for_event(&initial.codex, |event| {
        matches!(event, EventMsg::TurnComplete(_))
    })
    .await;
    server.wait_for_request_count(2).await;

    initial.codex.shutdown_and_wait().await?;
    let _ = summary_gate_tx.send(());

    let rollout = tokio::fs::read_to_string(rollout_path).await?;
    assert!(!rollout.contains(summary_text));
    server.shutdown().await;

    Ok(())
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn resume_chat_tree_branch_uses_current_node_history_for_next_request() -> Result<()> {
    skip_if_no_network!(Ok(()));

    let server = start_mock_server().await;
    let mut builder = test_codex();
    let initial = builder.build(&server).await?;

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
        .set_current_chat_tree_node(&node_a, /*expected_revision*/ None)
        .await
        .map_err(|err| anyhow::anyhow!("failed to select chat tree node: {err:?}"))?;
    assert_eq!(
        initial
            .codex
            .chat_tree_projection()
            .await
            .current_node_id
            .as_deref(),
        Some(node_a.as_str())
    );

    let node_d = submit_turn_and_capture_chat_tree_node(
        &initial,
        &server,
        "resp-d",
        "msg-d",
        "chat tree prompt D",
        "chat tree answer D",
    )
    .await?;

    let resumed = builder.restart(&server, &initial).await?;
    let projection = resumed.codex.chat_tree_projection().await;
    assert_eq!(projection.current_node_id.as_deref(), Some(node_d.as_str()));
    assert_eq!(projection.visible_node_ids, vec![node_a, node_d]);

    let resumed_mock = mount_sse_sequence(
        &server,
        vec![
            sse(vec![
                ev_response_created("resp-resume-chat-tree"),
                ev_assistant_message("msg-resume-chat-tree", "resumed chat tree answer"),
                ev_completed("resp-resume-chat-tree"),
            ]),
            sse(vec![
                ev_response_created("resp-resume-chat-tree-summary"),
                ev_assistant_message("msg-resume-chat-tree-summary", "resumed branch"),
                ev_completed("resp-resume-chat-tree-summary"),
            ]),
        ],
    )
    .await;
    resumed
        .codex
        .submit(Op::UserInput {
            items: vec![UserInput::Text {
                text: "chat tree prompt after resume".into(),
                text_elements: Vec::new(),
            }],
            final_output_json_schema: None,
            responsesapi_client_metadata: None,
            additional_context: Default::default(),
            thread_settings: Default::default(),
        })
        .await?;
    let resumed_node_id = wait_for_event_match(&resumed.codex, |event| match event {
        EventMsg::ChatTreeNodeStarted(event) => Some(event.node_id.clone()),
        _ => None,
    })
    .await;
    wait_for_turn_complete_and_summary(&resumed, &resumed_node_id).await;

    let requests = resumed_mock.requests();
    assert_eq!(requests.len(), 2);
    let request = &requests[0];
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
async fn resume_includes_initial_messages_from_rollout_events() -> Result<()> {
    skip_if_no_network!(Ok(()));

    let server = start_mock_server().await;
    let mut builder = test_codex();
    let initial = builder.build(&server).await?;
    let codex = Arc::clone(&initial.codex);

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
            items: vec![UserInput::Text {
                text: "Record some messages".into(),
                text_elements: text_elements.clone(),
            }],
            final_output_json_schema: None,
            responsesapi_client_metadata: None,
            additional_context: Default::default(),
            thread_settings: Default::default(),
        })
        .await?;

    wait_for_event(&codex, |event| matches!(event, EventMsg::TurnComplete(_))).await;
    let resumed = builder.restart(&server, &initial).await?;
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
async fn resume_includes_initial_messages_from_reasoning_events() -> Result<()> {
    skip_if_no_network!(Ok(()));

    let server = start_mock_server().await;
    let mut builder = test_codex().with_config(|config| {
        config.show_raw_agent_reasoning = true;
    });
    let initial = builder.build(&server).await?;
    let codex = Arc::clone(&initial.codex);

    let initial_sse = sse(vec![
        ev_response_created("resp-initial"),
        ev_reasoning_item("reason-1", &["Summarized step"], &["raw detail"]),
        ev_assistant_message("msg-1", "Completed reasoning turn"),
        ev_completed("resp-initial"),
    ]);
    mount_sse_once(&server, initial_sse).await;

    codex
        .submit(Op::UserInput {
            items: vec![UserInput::Text {
                text: "Record reasoning messages".into(),
                text_elements: Vec::new(),
            }],
            final_output_json_schema: None,
            responsesapi_client_metadata: None,
            additional_context: Default::default(),
            thread_settings: Default::default(),
        })
        .await?;

    wait_for_event(&codex, |event| matches!(event, EventMsg::TurnComplete(_))).await;
    let resumed = builder.restart(&server, &initial).await?;
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

    let initial_sse = sse(vec![
        ev_response_created("resp-initial"),
        ev_assistant_message("msg-1", "Completed first turn"),
        ev_completed("resp-initial"),
    ]);
    let initial_mock = mount_sse_once(&server, initial_sse).await;

    codex
        .submit(Op::UserInput {
            items: vec![UserInput::Text {
                text: "Record initial instructions".into(),
                text_elements: Vec::new(),
            }],
            final_output_json_schema: None,
            responsesapi_client_metadata: None,
            additional_context: Default::default(),
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
        config.model = Some("gpt-5.4".to_string());
    });
    let resumed = resume_builder.restart(&server, &initial).await?;
    resumed
        .codex
        .submit(Op::UserInput {
            items: vec![UserInput::Text {
                text: "Resume with different model".into(),
                text_elements: Vec::new(),
            }],
            final_output_json_schema: None,
            responsesapi_client_metadata: None,
            additional_context: Default::default(),
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
            items: vec![UserInput::Text {
                text: "Second turn after resume".into(),
                text_elements: Vec::new(),
            }],
            final_output_json_schema: None,
            responsesapi_client_metadata: None,
            additional_context: Default::default(),
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
            items: vec![UserInput::Text {
                text: "Record initial instructions".into(),
                text_elements: Vec::new(),
            }],
            final_output_json_schema: None,
            responsesapi_client_metadata: None,
            additional_context: Default::default(),
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
        config.model = Some("gpt-5.5".to_string());
    });
    let resumed = resume_builder.restart(&server, &initial).await?;
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
            items: vec![UserInput::Text {
                text: "first turn after override".into(),
                text_elements: Vec::new(),
            }],
            final_output_json_schema: None,
            responsesapi_client_metadata: None,
            additional_context: Default::default(),
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
