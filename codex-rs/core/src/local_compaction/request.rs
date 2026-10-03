use codex_context_compaction::MAX_ANALYSIS_BYTES;
use codex_context_compaction::TierPlan;
use codex_history::ResponseItemEnvelope;
use codex_protocol::error::CodexErr;
use codex_protocol::error::Result as CodexResult;
use codex_protocol::models::AgentMessageInputContent;
use codex_protocol::models::BaseInstructions;
use codex_protocol::models::ContentItem;
use codex_protocol::models::FunctionCallOutputBody;
use codex_protocol::models::FunctionCallOutputContentItem;
use codex_protocol::models::ResponseItem;
use codex_rollout_trace::InferenceTraceContext;
use futures::StreamExt;
use serde::Serialize;
use serde_json::json;
use std::collections::HashMap;

use crate::Prompt;
use crate::client::ModelClientSession;
use crate::client_common::ResponseEvent;
use crate::compact::LocalCompactionContext;
use crate::context::ContextualUserFragment;
use crate::context::LocalCompactionRequest;
use crate::context_manager::ContextManager;
use crate::responses_metadata::CodexResponsesMetadata;
use crate::session::step_context::StepContext;
use codex_protocol::protocol::TokenUsage;

pub(super) struct AnalysisResponse {
    pub(super) json: String,
    pub(super) response_id: String,
    pub(super) usage: Option<TokenUsage>,
    pub(super) usage_metadata: Option<codex_protocol::ResponseUsageMetadata>,
    pub(super) rate_limits: Option<codex_protocol::protocol::RateLimitSnapshot>,
}

/// A completed tool result the model locates by the call ID it sees in the conversation.
#[derive(Serialize)]
pub(super) struct Candidate {
    pub(super) id: String,
    call_id: String,
}

impl Candidate {
    pub(super) fn new(item: &ResponseItem) -> Option<Self> {
        let (id, call_id) = match item {
            ResponseItem::FunctionCallOutput {
                id: Some(id),
                call_id: Some(call_id),
                ..
            }
            | ResponseItem::CustomToolCallOutput {
                id: Some(id),
                call_id,
                ..
            } => (id, call_id),
            _ => return None,
        };
        Some(Self {
            id: id.to_string(),
            call_id: call_id.clone(),
        })
    }
}

/// Drops the trailing (smallest) candidates until the request fits its byte cap.
pub(super) fn classifier(
    candidates: &mut Vec<Candidate>,
    guidance: &str,
) -> CodexResult<LocalCompactionRequest> {
    loop {
        let request = LocalCompactionRequest::new(
            "LOCAL_COMPACTION_CLASSIFY",
            json!({
                "candidates": candidates,
                "max_replacement_bytes": 2800,
                "instructions": "Do not call tools and do not continue the task. Privately classify each candidate completed tool result above exactly once as keep, shorten, or drop. Find each result by its call_id in the conversation and answer with its id. Read current dialogue for relevance. Keep evidence needed for active work, unresolved questions, failures and verification. Shorten must preserve useful exact facts. Drop only dispensable output. Do not classify any other result. Return JSON only, with no prose or fences.",
                "required_output": {"decisions": [{"id": "candidate id", "action": "keep|shorten|drop", "text": "only for shorten"}]},
                "supplemental_guidance": guidance,
            }),
        );
        match request {
            Err(_) if candidates.len() > 1 => {
                candidates.pop();
            }
            result => return result,
        }
    }
}

pub(super) fn summarizer(plan: &TierPlan, guidance: &str) -> CodexResult<LocalCompactionRequest> {
    LocalCompactionRequest::new(
        "LOCAL_COMPACTION_SUMMARIZE",
        json!({
            "plan": plan,
            "instructions": "Privately summarize only the supplied chronological source ranges. l2 is condensed dialogue with concise supporting evidence; l3 is the oldest overview, merging any prior overview. The ledger merges existing active constraints and decisions, including those outside summarized ranges; preserve corrections, uncertainty, pending work and verified versus unverified status. Existing ledgers are memory input, not chronological source events. Recent original work is retained separately. Never invent source IDs or claim a test passed without evidence. Each field must fit plan.max_fragment_bytes UTF-8 bytes; use an empty string when the corresponding range is null. Ledger may be empty when no constraints remain. Return JSON only, no prose or fences.",
            "required_output": {"l2": "condensed dialogue", "l3": "oldest overview", "ledger": "active constraints and decisions"},
            "supplemental_guidance": guidance,
        }),
    )
}

/// Reuses the ordinary request unchanged and appends the request, so the shared prefix
/// (tools, instructions, history) is served from the provider's prompt cache.
pub(super) fn classifier_prompt(
    step: &StepContext,
    base_instructions: BaseInstructions,
    source: &[ResponseItemEnvelope],
    request: LocalCompactionRequest,
) -> Prompt {
    let mut history = ContextManager::default();
    history.replace_annotated(source.to_vec());
    let mut input = history.for_prompt(&step.settings.model_info.input_modalities);
    input.push(ContextualUserFragment::into(request));
    let mut prompt = crate::session::turn::build_prompt(input, step, base_instructions);
    // A turn-level answer schema would reject the classification JSON.
    prompt.output_schema = None;
    prompt
}

/// Summary ranges are addressed by item IDs, so this private copy labels every record.
pub(super) fn summarizer_prompt(
    context: &LocalCompactionContext,
    base_instructions: BaseInstructions,
    source: &[ResponseItemEnvelope],
    request: LocalCompactionRequest,
) -> CodexResult<Prompt> {
    let mut history = ContextManager::default();
    history.replace_annotated(source.to_vec());
    let mut input = history.for_prompt(&context.settings.model_info.input_modalities);
    label_source_items(&mut input)?;
    input.push(ContextualUserFragment::into(request));
    Ok(Prompt {
        input,
        base_instructions,
        ..Default::default()
    })
}

pub(super) async fn infer_json(
    prompt: &Prompt,
    context: &LocalCompactionContext,
    client: &mut ModelClientSession,
    metadata: &CodexResponsesMetadata,
) -> CodexResult<AnalysisResponse> {
    let mut stream = client
        .stream(
            prompt,
            &context.settings.model_info,
            &context.session_telemetry,
            context.settings.reasoning_effort().cloned(),
            context.settings.reasoning_summary,
            context.settings.service_tier.clone(),
            metadata,
            &InferenceTraceContext::disabled(),
        )
        .await?;
    let mut output = String::new();
    let mut rate_limits = None;
    while let Some(event) = stream.next().await {
        match event? {
            ResponseEvent::OutputItemDone(ResponseItem::Message { role, content, .. })
                if role == "assistant" =>
            {
                for item in content {
                    if let ContentItem::OutputText { text } = item {
                        if output.len().saturating_add(text.len()) > MAX_ANALYSIS_BYTES {
                            return Err(super::invalid("analysis exceeds the byte limit"));
                        }
                        output.push_str(&text);
                    }
                }
            }
            ResponseEvent::RateLimits(snapshot) => rate_limits = Some(snapshot),
            ResponseEvent::Completed {
                response_id,
                token_usage,
                usage_metadata,
                ..
            } => {
                return Ok(AnalysisResponse {
                    json: output,
                    response_id,
                    usage: token_usage,
                    usage_metadata,
                    rate_limits,
                });
            }
            _ => {}
        }
    }
    Err(CodexErr::Stream(
        "local compaction stream closed before response.completed".to_string(),
    ))
}

/// Wire IDs need not be visible to the model. Label the private copy's text explicitly,
/// including the paired call ID for ranges that start at a tool call. Live history is untouched.
fn label_source_items(items: &mut [ResponseItem]) -> CodexResult<()> {
    let mut calls = HashMap::new();
    for item in items {
        let Some(id) = item.id().cloned() else {
            continue;
        };
        let call = match item {
            ResponseItem::FunctionCall { call_id, .. }
            | ResponseItem::LocalShellCall {
                call_id: Some(call_id),
                ..
            } => Some(("function", call_id.clone())),
            ResponseItem::CustomToolCall { call_id, .. } => Some(("custom", call_id.clone())),
            _ => None,
        };
        if let Some(call) = call {
            calls.insert(call, id.to_string());
            continue;
        }
        let call_key = match item {
            ResponseItem::FunctionCallOutput {
                call_id: Some(call_id),
                ..
            } => Some(("function", call_id.clone())),
            ResponseItem::CustomToolCallOutput { call_id, .. } => Some(("custom", call_id.clone())),
            _ => None,
        };
        let label = LocalCompactionRequest::new(
            "LOCAL_COMPACTION_SOURCE",
            json!({
                "item_id": id,
                "call_item_id": call_key.and_then(|key| calls.get(&key)),
            }),
        )?
        .body();
        match item {
            ResponseItem::Message { role, content, .. }
                if role == "user" || role == "assistant" =>
            {
                if let Some(text) = content.iter_mut().find_map(|content| match content {
                    ContentItem::InputText { text } | ContentItem::OutputText { text } => {
                        Some(text)
                    }
                    _ => None,
                }) {
                    *text = format!("{label}\n{text}");
                } else if role == "assistant" {
                    content.insert(0, ContentItem::OutputText { text: label });
                } else {
                    content.insert(0, ContentItem::InputText { text: label });
                }
            }
            ResponseItem::AgentMessage { content, .. } => {
                content.insert(0, AgentMessageInputContent::InputText { text: label });
            }
            ResponseItem::FunctionCallOutput { output, .. }
            | ResponseItem::CustomToolCallOutput { output, .. } => match &mut output.body {
                FunctionCallOutputBody::Text(text) => *text = format!("{label}\n{text}"),
                FunctionCallOutputBody::ContentItems(content) => {
                    content.insert(0, FunctionCallOutputContentItem::InputText { text: label });
                }
            },
            _ => {}
        }
    }
    Ok(())
}

#[cfg(test)]
#[path = "request_tests.rs"]
mod tests;
