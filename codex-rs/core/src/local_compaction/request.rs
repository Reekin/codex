//! Private marking and summary requests, sent with the ordinary request's prefix.

use std::collections::HashMap;

use codex_context_compaction::LocalCompactionRequest;
use codex_context_compaction::MAX_ANALYSIS_BYTES;
use codex_protocol::ResponseUsageMetadata;
use codex_protocol::error::CodexErr;
use codex_protocol::error::Result as CodexResult;
use codex_protocol::models::ContentItem;
use codex_protocol::models::ResponseItem;
use codex_protocol::openai_models::ReasoningEffort;
use codex_protocol::protocol::RateLimitSnapshot;
use codex_protocol::protocol::TokenUsage;
use codex_rollout_trace::InferenceTraceContext;
use futures::StreamExt;

use super::invalid;
use crate::Prompt;
use crate::client::ModelClientSession;
use crate::client_common::ResponseEvent;
use crate::context::ContextualUserFragment;
use crate::context_manager::ContextManager;
use crate::responses_metadata::CodexResponsesMetadata;
use crate::session::RequestEffortUsage;
use crate::session::session::Session;
use crate::session::step_context::StepContext;

pub(super) struct AnalysisResponse {
    pub(super) text: String,
    pub(super) response_id: String,
    pub(super) usage: Option<TokenUsage>,
    pub(super) usage_metadata: Option<ResponseUsageMetadata>,
    pub(super) rate_limits: Option<RateLimitSnapshot>,
}

/// Appends `request` to the ordinary request built from `history`. With the sampling step's
/// tools and instructions the shared prefix is served from the provider's prompt cache.
pub(super) async fn private_prompt(
    sess: &Session,
    step: &StepContext,
    history: ContextManager,
    request: LocalCompactionRequest,
) -> Prompt {
    let mut input = history
        .for_prompt(
            &step.settings.model_info.input_modalities,
            sess.services.image_store.as_ref(),
        )
        .await;
    sess.services
        .executed_tool_calls
        .attach_to_prompt(&mut input, &mut HashMap::new());
    input.push(ContextualUserFragment::into(request));
    let incremental_tools = sess.current_window_uses_incremental_tools(step).await;
    let mut prompt = crate::session::turn::build_prompt(
        input,
        step,
        sess.get_prompt_base_instructions().await,
        incremental_tools,
    );
    // A turn-level answer schema would reject the private answer.
    prompt.output_schema = None;
    prompt
}

/// The step's model and inference settings, with the effort a compaction request uses.
pub(super) async fn request_effort(sess: &Session, step: &StepContext) -> Option<ReasoningEffort> {
    sess.reasoning_effort_for_request(&step.settings, RequestEffortUsage::Compaction)
        .await
}

/// Streams one private request and returns its assistant text, bounded in size.
pub(super) async fn infer_text(
    prompt: &Prompt,
    step: &StepContext,
    effort: Option<ReasoningEffort>,
    client: &mut ModelClientSession,
    metadata: &CodexResponsesMetadata,
    trace: &InferenceTraceContext,
) -> CodexResult<AnalysisResponse> {
    let settings = &step.settings;
    let mut stream = client
        .stream(
            prompt,
            &settings.model_info,
            &step.session_telemetry,
            effort,
            settings.reasoning_summary,
            settings.service_tier.clone(),
            metadata,
            trace,
        )
        .await?;
    let mut text = String::new();
    let mut rate_limits = None;
    while let Some(event) = stream.next().await {
        match event? {
            ResponseEvent::OutputItemDone(ResponseItem::Message { role, content, .. })
                if role == "assistant" =>
            {
                for item in content {
                    if let ContentItem::OutputText { text: part, .. } = item {
                        if text.len().saturating_add(part.len()) > MAX_ANALYSIS_BYTES {
                            return Err(invalid("analysis exceeds the byte limit"));
                        }
                        text.push_str(&part);
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
                    text,
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
