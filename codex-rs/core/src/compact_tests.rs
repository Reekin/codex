use super::*;
use codex_history::CodexHarnessMetadata;
use codex_history::ResponseItemEnvelope;
use codex_protocol::ResponseItemId;
use codex_protocol::models::ContentItemKind;
use codex_protocol::models::ContentItemMetadata;
use codex_protocol::models::DEFAULT_IMAGE_DETAIL;
use codex_protocol::models::ImageReference;
use codex_protocol::models::InternalChatMessageMetadataPassthrough;
use pretty_assertions::assert_eq;
use serde_json::json;
use test_case::test_case;

fn annotated(items: Vec<ResponseItem>) -> Vec<ResponseItemEnvelope> {
    items.into_iter().map(ResponseItemEnvelope::new).collect()
}

fn raw(items: Vec<ResponseItemEnvelope>) -> Vec<ResponseItem> {
    items
        .into_iter()
        .map(ResponseItemEnvelope::into_item)
        .collect()
}

fn user_message(text: &str) -> ResponseItem {
    ResponseItem::Message {
        status: None,
        encrypted_content: None,
        id: None,
        role: "user".to_string(),
        content: vec![ContentItem::InputText {
            text: text.to_string(),
        }],
        phase: None,
        internal_chat_message_metadata_passthrough: None,
    }
}

fn compacted_user_message<'a>(text: &str, original: &'a ResponseItem) -> CompactedUserMessage<'a> {
    CompactedUserMessage {
        message: text.to_string(),
        original,
        harness_metadata: None,
    }
}

#[tokio::test]
async fn extension_window_prefix_precedes_retained_turns_and_is_not_repeated() {
    use codex_extension_api::PreviousWorldStateSection;
    use codex_extension_api::RenderedWorldStateFragment;
    use codex_extension_api::WorldStateSectionContribution;

    let (session, turn) = crate::session::tests::make_session_and_context().await;
    let step = StepContext::for_test(Arc::new(turn));
    let mut world = WorldState::default();
    world.add_extension_section(
        WorldStateSectionContribution::new("example", |previous| {
            let fragment = (!matches!(previous, PreviousWorldStateSection::Known(_))).then(|| {
                RenderedWorldStateFragment::new(
                    "developer",
                    ("<example>", "</example>"),
                    codex_protocol::models::input_text("active policy"),
                )
            });
            (Some(json!("active policy")), fragment)
        })
        .with_window_prefix()
        .with_retained_fragment_matcher(|role, text| {
            role == "developer" && text == "<example>active policy</example>"
        }),
    );
    let retained = vec![user_message("earlier turn"), user_message("latest turn")];
    let (replacement, snapshot) =
        build_compaction_replacement_history(&session, &step, &world, annotated(retained.clone()))
            .await;
    let items = raw(replacement);
    let ResponseItem::Message { role, content, .. } = &items[0] else {
        panic!("expected a separate developer prefix");
    };
    assert_eq!(role, "developer");
    assert_eq!(
        content,
        &[codex_protocol::models::input_text(
            "<example>active policy</example>"
        )]
    );
    assert_eq!(items[1], retained[0]);
    assert!(
        world
            .render_history_diff(Some(&snapshot), &items)
            .1
            .is_empty()
    );
}

#[test]
fn content_items_to_text_joins_non_empty_segments() {
    let items = vec![
        ContentItem::InputText {
            text: "hello".to_string(),
        },
        ContentItem::OutputText {
            annotations: None,
            logprobs: None,
            text: String::new(),
        },
        ContentItem::OutputText {
            annotations: None,
            logprobs: None,
            text: "world".to_string(),
        },
    ];

    let joined = content_items_to_text(&items);

    assert_eq!(Some("hello\nworld".to_string()), joined);
}

#[test]
fn content_items_to_text_ignores_image_only_content() {
    let items = vec![ContentItem::InputImage {
        image: ImageReference::Inline {
            image_url: "file://image.png".to_string(),
        },
        detail: Some(DEFAULT_IMAGE_DETAIL),
    }];

    let joined = content_items_to_text(&items);

    assert_eq!(None, joined);
}

#[test]
fn collect_user_messages_extracts_user_text_only() {
    let items = vec![
        ResponseItem::Message {
            status: None,
            encrypted_content: None,
            id: Some(ResponseItemId::with_suffix("msg", "assistant")),
            role: "assistant".to_string(),
            content: vec![ContentItem::OutputText {
                annotations: None,
                logprobs: None,
                text: "ignored".to_string(),
            }],
            phase: None,
            internal_chat_message_metadata_passthrough: None,
        },
        ResponseItem::Message {
            status: None,
            encrypted_content: None,
            id: Some(ResponseItemId::with_suffix("msg", "user")),
            role: "user".to_string(),
            content: vec![ContentItem::InputText {
                text: "first".to_string(),
            }],
            phase: None,
            internal_chat_message_metadata_passthrough: None,
        },
        ResponseItem::Other,
    ];

    let collected = collect_user_messages(&items);

    assert_eq!(vec![compacted_user_message("first", &items[1])], collected,);
}

#[test_case(json!([{"type": "input_text", "text": "first"}]), "first", true; "unchanged text")]
#[test_case(json!([
    {"type": "input_text", "text": "first"},
    {"type": "input_text", "text": ""},
    {"type": "input_text", "text": "second"}
]), "firstsecond", true; "text boundaries")]
#[test_case(json!([
    {"type": "input_text", "text": "first"},
    {"type": "input_image", "image_url": "file://image.png"}
]), "first", false; "omitted media")]
fn collect_annotated_user_messages_extracts_user_text_only(
    input_content: serde_json::Value,
    expected_text: &str,
    preserve_content: bool,
) {
    let source = json!({
        "id": {"message_id": "source", "turn_id": "turn", "role": "user"},
        "revision": "retained_revision", "complete": true,
    });
    let metadata: CodexHarnessMetadata = serde_json::from_value(json!({
        "retained_source": source, "guardian_sources": [source],
        "guardian_source_order_guidance": true, "user_input_order": 7,
    }))
    .unwrap();
    let mut item = user_message("first");
    if let ResponseItem::Message { content, .. } = &mut item {
        *content = serde_json::from_value(input_content).unwrap();
    }
    let items = vec![
        ResponseItemEnvelope {
            item: item.clone(),
            metadata: Some(metadata.clone()),
        },
        ResponseItemEnvelope::new(ResponseItem::Other),
    ];

    let collected = collect_annotated_user_messages(&items);

    if !preserve_content {
        item = user_message(expected_text);
        if let ResponseItem::Message {
            internal_chat_message_metadata_passthrough,
            ..
        } = &mut item
        {
            *internal_chat_message_metadata_passthrough =
                Some(InternalChatMessageMetadataPassthrough {
                    content_item_kinds: Some(vec![ContentItemKind("unknown".to_string())]),
                    ..Default::default()
                });
        }
    }
    let expected = ResponseItemEnvelope {
        item,
        metadata: Some(metadata),
    };
    assert_eq!(
        collected,
        vec![CompactedUserMessage {
            message: expected_text.to_owned(),
            original: &items[0].item,
            harness_metadata: items[0].metadata.as_ref(),
        }]
    );
    assert_eq!(
        build_compacted_history(Vec::new(), &collected, "summary"),
        vec![
            expected,
            ResponseItemEnvelope::new(ContextualUserFragment::into(CompactionSummary::new(
                "summary"
            ))),
        ],
    );
}

#[test]
fn collect_user_messages_filters_session_prefix_entries() {
    let items = vec![
        ResponseItem::Message {
            status: None,
            encrypted_content: None,
            id: None,
            role: "user".to_string(),
            content: vec![ContentItem::InputText {
                text: r#"# AGENTS.md instructions for project

<INSTRUCTIONS>
do things
</INSTRUCTIONS>"#
                    .to_string(),
            }],
            phase: None,
            internal_chat_message_metadata_passthrough: None,
        },
        ResponseItem::Message {
            status: None,
            encrypted_content: None,
            id: None,
            role: "user".to_string(),
            content: vec![ContentItem::InputText {
                text: "<ENVIRONMENT_CONTEXT>cwd=/tmp</ENVIRONMENT_CONTEXT>".to_string(),
            }],
            phase: None,
            internal_chat_message_metadata_passthrough: None,
        },
        ResponseItem::Message {
            status: None,
            encrypted_content: None,
            id: None,
            role: "user".to_string(),
            content: vec![ContentItem::InputText {
                text: "real user message".to_string(),
            }],
            phase: None,
            internal_chat_message_metadata_passthrough: None,
        },
    ];

    let collected = collect_user_messages(&items);

    assert_eq!(
        vec![compacted_user_message("real user message", &items[2])],
        collected
    );
}

#[test]
fn collect_user_messages_filters_legacy_warnings() {
    let items = vec![
        user_message(
            "Warning: The maximum number of unified exec processes you can keep open is 60 and you currently have 61 processes open. Reuse older processes or close them to prevent automatic pruning of old processes",
        ),
        user_message(
            "Warning: apply_patch was requested via exec_command. Use the apply_patch tool instead of exec_command.",
        ),
        user_message(
            "Warning: Your account was flagged for potentially high-risk cyber activity and this request was routed to gpt-5.2 as a fallback. To regain access to gpt-5.3-codex, apply for trusted access: https://chatgpt.com/cyber or learn more: https://developers.openai.com/codex/concepts/cyber-safety",
        ),
        user_message("real user message"),
    ];

    let collected = collect_user_messages(&items);

    assert_eq!(
        vec![compacted_user_message("real user message", &items[3])],
        collected
    );
}

#[test_case(1, None; "legacy")]
#[test_case(1, Some(vec![ContentItemMetadata::user()]); "user")]
#[test_case(2, Some(vec![ContentItemMetadata::user(); 2]); "multiple user parts")]
#[test_case(2, Some(vec![ContentItemMetadata::user(), ContentItemMetadata::harness()]); "mixed sources")]
#[test_case(1, Some(vec![ContentItemMetadata::default()]); "unknown")]
#[test_case(1, Some(vec![]); "missing entry")]
#[test_case(1, Some(vec![ContentItemMetadata::user(); 2]); "extra entry")]
fn build_token_limited_compacted_history_truncates_overlong_user_messages(
    content_parts: usize,
    attribution: Option<Vec<ContentItemMetadata>>,
) {
    // Use a small truncation limit so the test remains fast while still validating
    // that oversized user content is truncated.
    let max_tokens = 16;
    let big = "word ".repeat(200);
    let mut original = ResponseItemEnvelope::new(user_message(&big));
    if let ResponseItem::Message {
        content,
        internal_chat_message_metadata_passthrough,
        ..
    } = &mut original.item
    {
        *content = vec![ContentItem::InputText { text: big.clone() }; content_parts];
        *internal_chat_message_metadata_passthrough =
            Some(InternalChatMessageMetadataPassthrough {
                content_item_metadata: attribution.clone(),
                ..Default::default()
            });
    }
    original
        .item
        .set_id(Some(ResponseItemId::with_suffix("msg", "long-user")));
    original.metadata = Some(CodexHarnessMetadata::default());
    let history = super::build_compacted_history_with_limit(
        Vec::new(),
        &collect_annotated_user_messages(std::slice::from_ref(&original)),
        "SUMMARY",
        max_tokens,
    );
    assert_eq!(history.len(), 2);

    let truncated_message = &history[0].item;
    let summary_message = &history[1].item;

    let truncated_text = match truncated_message {
        ResponseItem::Message {
            role,
            content,
            internal_chat_message_metadata_passthrough,
            ..
        } if role == "user" => {
            let source_at = |index| {
                attribution
                    .as_ref()
                    .and_then(|sources| sources.get(index))
                    .cloned()
                    .unwrap_or_default()
            };
            assert_eq!(
                content,
                &vec![
                    ContentItem::InputText {
                        text: big[..32].to_string()
                    },
                    ContentItem::InputText {
                        text: format!("…{} tokens truncated…", 250 * content_parts - max_tokens)
                    },
                    ContentItem::InputText {
                        text: big[big.len() - 32..].to_string()
                    },
                ]
            );
            assert_eq!(
                internal_chat_message_metadata_passthrough
                    .as_ref()
                    .unwrap()
                    .content_item_metadata,
                Some(vec![
                    source_at(0),
                    ContentItemMetadata::harness(),
                    source_at(content_parts - 1)
                ])
            );
            content_items_to_text(content).unwrap_or_default()
        }
        other => panic!("unexpected item in history: {other:?}"),
    };

    assert!(
        truncated_text.contains("tokens truncated"),
        "expected truncation marker in truncated user message"
    );
    assert!(
        !truncated_text.contains(&big),
        "truncated user message should not include the full oversized user text"
    );

    let summary_text = match summary_message {
        ResponseItem::Message { role, content, .. } if role == "user" => {
            content_items_to_text(content).unwrap_or_default()
        }
        other => panic!("unexpected item in history: {other:?}"),
    };
    assert_eq!(summary_text, "SUMMARY");
    assert_eq!(history[0].id(), original.id());
    assert_eq!(history[0].metadata, Some(CodexHarnessMetadata::default()));
    assert_eq!(history[1].metadata, None);
}

#[test]
fn build_token_limited_compacted_history_appends_summary_message() {
    let initial_context: Vec<ResponseItemEnvelope> = Vec::new();
    let original = user_message("first user message");
    let user_messages = collect_user_messages(std::slice::from_ref(&original));
    let summary_text = "summary text";

    let history = build_compacted_history(initial_context, &user_messages, summary_text);
    assert!(
        !history.is_empty(),
        "expected compacted history to include summary"
    );

    let last = history.last().expect("history should have a summary entry");
    let summary = match &last.item {
        ResponseItem::Message { role, content, .. } if role == "user" => {
            content_items_to_text(content).unwrap_or_default()
        }
        other => panic!("expected summary message, found {other:?}"),
    };
    assert_eq!(summary, summary_text);
}

#[test_case(false; "legacy")]
#[test_case(true; "attributed text with discarded media")]
fn build_compacted_history_preserves_user_message_passthrough_metadata(attributed: bool) {
    let original = serde_json::from_value(json!({
        "type": "message", "role": "user",
        "id": ResponseItemId::with_suffix("msg", "user"),
        "content": [
            {"type": "input_text", "text": "<image>"},
            {"type": "input_image", "image_url": "file://image.png"},
            {"type": "input_text", "text": "</image>"},
            {"type": "input_text", "text": "first user message"},
            {"type": "input_text", "text": "<audio>"},
            {"type": "input_audio", "audio_url": "file://audio.wav"},
            {"type": "input_text", "text": "</audio>"}
        ],
        "internal_chat_message_metadata_passthrough": {
            "turn_id": "turn-1",
            "content_item_kinds": ["unknown", "user.image", "unknown", "user.text", "unknown", "user.audio", "unknown"],
            "content_item_metadata": attributed.then(|| {
                let mut sources = vec![ContentItemMetadata::harness(); 7];
                sources[3] = ContentItemMetadata::user();
                sources
            })
        }
    }))
    .unwrap();
    let original = ResponseItemEnvelope {
        item: original,
        metadata: Some(CodexHarnessMetadata::default()),
    };
    let history = build_compacted_history(
        Vec::new(),
        &collect_annotated_user_messages(std::slice::from_ref(&original)),
        "summary text",
    );

    assert_eq!(
        history,
        vec![
            ResponseItemEnvelope {
                item: ResponseItem::Message {
                    encrypted_content: None,
                    status: None,
                    id: Some(ResponseItemId::with_suffix("msg", "user")),
                    role: "user".to_string(),
                    content: vec![ContentItem::InputText {
                        text: "first user message".to_string(),
                    }],
                    phase: None,
                    internal_chat_message_metadata_passthrough: Some(
                        InternalChatMessageMetadataPassthrough {
                            turn_id: Some("turn-1".to_string()),
                            content_item_kinds: Some(vec![ContentItemKind(
                                "user.text".to_string()
                            )]),
                            content_item_metadata: attributed
                                .then(|| vec![ContentItemMetadata::user()]),
                            ..Default::default()
                        },
                    ),
                },
                metadata: Some(CodexHarnessMetadata::default()),
            },
            ResponseItemEnvelope::new(ContextualUserFragment::into(CompactionSummary::new(
                "summary text",
            ))),
        ]
    );
}

#[test]
fn assemble_compaction_history_keeps_prefix_first_and_summary_last() {
    let prefix = ContextualUserFragment::into(crate::context::BaseInstructionsFragment(
        "base instructions".to_string(),
    ));
    let agent_completion = ResponseItem::AgentMessage {
        id: None,
        author: "child".to_string(),
        recipient: "parent".to_string(),
        content: vec![AgentMessageInputContent::InputText {
            text: "Message Type: FINAL_ANSWER\nPayload:\nchild completion".to_string(),
        }],
        internal_chat_message_metadata_passthrough: None,
    };
    let compacted_history = vec![
        ResponseItem::Message {
            status: None,
            encrypted_content: None,
            id: None,
            role: "user".to_string(),
            content: vec![ContentItem::InputText {
                text: "older user".to_string(),
            }],
            phase: None,
            internal_chat_message_metadata_passthrough: None,
        },
        ResponseItem::Message {
            status: None,
            encrypted_content: None,
            id: None,
            role: "user".to_string(),
            content: vec![ContentItem::InputText {
                text: "latest user".to_string(),
            }],
            phase: None,
            internal_chat_message_metadata_passthrough: None,
        },
        agent_completion.clone(),
        ResponseItem::Message {
            status: None,
            encrypted_content: None,
            id: None,
            role: "user".to_string(),
            content: vec![ContentItem::InputText {
                text: format!("{SUMMARY_PREFIX}\nsummary text"),
            }],
            phase: None,
            internal_chat_message_metadata_passthrough: None,
        },
    ];
    let initial_context = vec![ResponseItem::Message {
        status: None,
        encrypted_content: None,
        id: None,
        role: "developer".to_string(),
        content: vec![ContentItem::InputText {
            text: "fresh permissions".to_string(),
        }],
        phase: None,
        internal_chat_message_metadata_passthrough: None,
    }];

    let refreshed = raw(assemble_compaction_history(
        annotated(compacted_history),
        vec![prefix.clone()],
        initial_context,
    ));
    let expected = vec![
        prefix,
        ResponseItem::Message {
            status: None,
            encrypted_content: None,
            id: None,
            role: "user".to_string(),
            content: vec![ContentItem::InputText {
                text: "older user".to_string(),
            }],
            phase: None,
            internal_chat_message_metadata_passthrough: None,
        },
        ResponseItem::Message {
            status: None,
            encrypted_content: None,
            id: None,
            role: "developer".to_string(),
            content: vec![ContentItem::InputText {
                text: "fresh permissions".to_string(),
            }],
            phase: None,
            internal_chat_message_metadata_passthrough: None,
        },
        ResponseItem::Message {
            status: None,
            encrypted_content: None,
            id: None,
            role: "user".to_string(),
            content: vec![ContentItem::InputText {
                text: "latest user".to_string(),
            }],
            phase: None,
            internal_chat_message_metadata_passthrough: None,
        },
        agent_completion,
        ResponseItem::Message {
            status: None,
            encrypted_content: None,
            id: None,
            role: "user".to_string(),
            content: vec![ContentItem::InputText {
                text: format!("{SUMMARY_PREFIX}\nsummary text"),
            }],
            phase: None,
            internal_chat_message_metadata_passthrough: None,
        },
    ];
    assert_eq!(refreshed, expected);
}

#[test]
fn assemble_compaction_history_keeps_compaction_last() {
    let agent_task = ResponseItem::AgentMessage {
        id: None,
        author: "parent".to_string(),
        recipient: "child".to_string(),
        content: Vec::new(),
        internal_chat_message_metadata_passthrough: None,
    };
    let compacted_history = vec![
        agent_task.clone(),
        ResponseItem::Compaction {
            id: None,
            encrypted_content: "encrypted".to_string(),
            internal_chat_message_metadata_passthrough: None,
        },
    ];
    let initial_context = vec![ResponseItem::Message {
        status: None,
        encrypted_content: None,
        id: None,
        role: "developer".to_string(),
        content: vec![ContentItem::InputText {
            text: "fresh permissions".to_string(),
        }],
        phase: None,
        internal_chat_message_metadata_passthrough: None,
    }];

    let refreshed = raw(assemble_compaction_history(
        annotated(compacted_history),
        Vec::new(),
        initial_context,
    ));
    let expected = vec![
        ResponseItem::Message {
            status: None,
            encrypted_content: None,
            id: None,
            role: "developer".to_string(),
            content: vec![ContentItem::InputText {
                text: "fresh permissions".to_string(),
            }],
            phase: None,
            internal_chat_message_metadata_passthrough: None,
        },
        agent_task,
        ResponseItem::Compaction {
            id: None,
            encrypted_content: "encrypted".to_string(),
            internal_chat_message_metadata_passthrough: None,
        },
    ];
    assert_eq!(refreshed, expected);
}
