use super::*;
use codex_protocol::context_usage::ContextUsageBreakdown;
use codex_protocol::models::ContentItemKind;
use codex_protocol::models::FunctionCallOutputPayload;
use codex_protocol::models::InternalChatMessageMetadataPassthrough;
use codex_protocol::models::ReasoningItemReasoningSummary;
use pretty_assertions::assert_eq;

fn classified_message(role: &str, parts: &[(&str, &str)]) -> ResponseItem {
    ResponseItem::Message {
        id: None,
        role: role.to_string(),
        content: parts
            .iter()
            .map(|(_, text)| ContentItem::InputText {
                text: (*text).to_string(),
            })
            .collect(),
        phase: None,
        internal_chat_message_metadata_passthrough: Some(InternalChatMessageMetadataPassthrough {
            content_item_kinds: Some(
                parts
                    .iter()
                    .map(|(kind, _)| ContentItemKind((*kind).to_string()))
                    .collect(),
            ),
            ..Default::default()
        }),
    }
}

fn plain_message(role: &str, text: &str) -> ResponseItem {
    ResponseItem::Message {
        id: None,
        role: role.to_string(),
        content: vec![ContentItem::OutputText {
            text: text.to_string(),
        }],
        phase: None,
        internal_chat_message_metadata_passthrough: None,
    }
}

fn nonzero_categories(breakdown: &ContextUsageBreakdown) -> Vec<ContextUsageCategory> {
    ContextUsageCategory::ALL
        .into_iter()
        .filter(|category| breakdown.tokens(*category) > 0)
        .collect()
}

#[test]
fn breakdown_classifies_history_and_sums_to_active_tokens() {
    let long = "x".repeat(4_000);
    let items = vec![
        classified_message(
            "developer",
            &[
                ("generic.developer_instructions", &long),
                ("skills.catalog", &long),
                ("permissions.instructions", &long),
            ],
        ),
        classified_message(
            "user",
            &[
                ("agents_md.instructions", &long),
                ("environments.environment_context", &long),
            ],
        ),
        classified_message("user", &[("user.text", &long)]),
        // Role switches injected by clients carry no classification.
        plain_message("developer", &long),
        ResponseItem::Reasoning {
            id: None,
            summary: vec![ReasoningItemReasoningSummary::SummaryText {
                text: "summary".to_string(),
            }],
            content: None,
            encrypted_content: Some("a".repeat(8_000)),
            internal_chat_message_metadata_passthrough: None,
        },
        ResponseItem::FunctionCall {
            id: None,
            name: "shell".to_string(),
            namespace: None,
            arguments: long.clone(),
            call_id: "call-1".to_string(),
            encrypted_function_args: None,
            internal_chat_message_metadata_passthrough: None,
        },
        ResponseItem::FunctionCallOutput {
            id: None,
            call_id: Some("call-1".to_string()),
            name: None,
            namespace: None,
            output: FunctionCallOutputPayload::from_text(long.clone()),
            internal_chat_message_metadata_passthrough: None,
        },
        plain_message("assistant", &long),
    ];
    let overhead = RequestOverhead::new(&long, /*tools_tokens*/ 1_000);

    let usage = context_usage(&items, overhead, /*tokens*/ 50_000, Some(90_000));

    let sum: i64 = ContextUsageCategory::ALL
        .into_iter()
        .map(|category| usage.breakdown.tokens(category))
        .sum();
    assert_eq!(
        (usage.tokens, usage.auto_compact_token_limit, sum),
        (50_000, Some(90_000), 50_000)
    );
    assert_eq!(
        nonzero_categories(&usage.breakdown),
        vec![
            ContextUsageCategory::BaseInstructions,
            ContextUsageCategory::DeveloperInstructions,
            ContextUsageCategory::AgentsMd,
            ContextUsageCategory::Skills,
            ContextUsageCategory::Tools,
            ContextUsageCategory::Environment,
            ContextUsageCategory::UserMessages,
            ContextUsageCategory::AgentMessages,
            ContextUsageCategory::ToolCalls,
            ContextUsageCategory::Reasoning,
        ]
    );
    // Two developer sources (classified and injected) outweigh one AGENTS.md source.
    assert!(usage.breakdown.developer_instructions > usage.breakdown.agents_md);
}

#[test]
fn unknown_classifications_fall_into_other() {
    let items = vec![classified_message(
        "developer",
        &[("future_feature.instructions", "some text")],
    )];

    let usage = context_usage(&items, RequestOverhead::default(), /*tokens*/ 7, None);

    assert_eq!(
        usage.breakdown,
        ContextUsageBreakdown {
            other: 7,
            ..Default::default()
        }
    );
}

#[test]
fn empty_estimate_assigns_active_tokens_to_other() {
    let usage = context_usage(
        Vec::<ResponseItem>::new().iter(),
        RequestOverhead::default(),
        /*tokens*/ 42,
        None,
    );

    assert_eq!(
        usage.breakdown,
        ContextUsageBreakdown {
            other: 42,
            ..Default::default()
        }
    );
}
