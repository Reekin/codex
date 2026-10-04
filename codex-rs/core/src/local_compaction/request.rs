use codex_context_compaction::MAX_ANALYSIS_BYTES;
use codex_context_compaction::MAX_CALL_SUMMARY_BYTES;
use codex_context_compaction::MAX_SUMMARY_BYTES;
use codex_history::ResponseItemEnvelope;
use codex_protocol::error::CodexErr;
use codex_protocol::error::Result as CodexResult;
use codex_protocol::models::BaseInstructions;
use codex_protocol::models::ContentItem;
use codex_protocol::models::ResponseItem;
use codex_protocol::openai_models::ModelInfo;
use codex_rollout_trace::InferenceTraceContext;
use futures::StreamExt;
use serde::Serialize;
use serde_json::json;

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
    /// The paired call carries large arguments that shorten/drop replace with `call_text`.
    #[serde(skip_serializing_if = "std::ops::Not::not")]
    pub(super) summarize_call: bool,
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
            summarize_call: false,
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
                "max_call_text_bytes": MAX_CALL_SUMMARY_BYTES,
                "instructions": "Do not call tools and do not continue the task. Privately classify each candidate completed tool result above exactly once as keep, shorten, or drop. Find each result by its call_id in the conversation and answer with its id. Read current dialogue for relevance. Keep evidence needed for active work, unresolved questions, failures and verification. Shorten must preserve useful exact facts. Drop only dispensable output. Shorten and drop remove any images in the result; keep a result only if its images still need to be looked at, and state in shorten text what they showed. Originals stay retrievable. For candidates with summarize_call, shorten and drop also replace the call's arguments: give call_text, one line saying what the call did (for a patch, which files changed and how). Do not classify any other result. Return JSON only, with no prose or fences.",
                "required_output": {"decisions": [{"id": "candidate id", "action": "keep|shorten|drop", "text": "only for shorten", "call_text": "only with summarize_call, for shorten or drop"}]},
                "supplemental_guidance": guidance,
            }),
            LocalCompactionRequest::MAX_CLASSIFY_BYTES,
        );
        match request {
            Err(_) if candidates.len() > 1 => {
                candidates.pop();
            }
            result => return result,
        }
    }
}

/// One handoff summary of everything sent above it: earlier summaries and older records.
pub(super) fn summary(guidance: &str) -> CodexResult<LocalCompactionRequest> {
    LocalCompactionRequest::new(
        "LOCAL_COMPACTION_SUMMARIZE",
        json!({
            "instructions": "Do not call tools and do not continue the task. You are performing a context checkpoint compaction: create a handoff summary of the whole conversation above, including any earlier summaries in local_compaction blocks, for another LLM that will resume the task. Include the user's goals and current progress, key decisions, important context, constraints and user preferences (including corrections), what remains to be done as clear next steps, and critical data, examples, file paths and references needed to continue. Say what was verified and what was not. Be concise and structured. Answer with the summary text only.",
            "max_bytes": MAX_SUMMARY_BYTES,
            "supplemental_guidance": guidance,
        }),
        LocalCompactionRequest::MAX_BYTES,
    )
}

/// Appends the request to the ordinary request's history. With the sampling step's tools and
/// instructions the shared prefix is served from the provider's prompt cache.
pub(super) fn private_prompt(
    step: Option<&StepContext>,
    base_instructions: BaseInstructions,
    source: &[ResponseItemEnvelope],
    model_info: &ModelInfo,
    request: LocalCompactionRequest,
) -> Prompt {
    let mut history = ContextManager::default();
    history.replace_annotated(source.to_vec());
    let mut input = history.for_prompt(&model_info.input_modalities);
    input.push(ContextualUserFragment::into(request));
    let Some(step) = step else {
        return Prompt {
            input,
            base_instructions,
            ..Default::default()
        };
    };
    let mut prompt = crate::session::turn::build_prompt(input, step, base_instructions);
    // A turn-level answer schema would reject the private answer.
    prompt.output_schema = None;
    prompt
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
