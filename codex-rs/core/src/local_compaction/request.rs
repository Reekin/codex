use codex_context_compaction::MAX_ANALYSIS_BYTES;
use codex_context_compaction::TierPlan;
use codex_history::ResponseItemEnvelope;
use codex_protocol::error::CodexErr;
use codex_protocol::error::Result as CodexResult;
use codex_protocol::models::ContentItem;
use codex_protocol::models::ResponseItem;
use codex_rollout_trace::InferenceTraceContext;
use futures::StreamExt;
use serde_json::json;

use crate::Prompt;
use crate::client::ModelClientSession;
use crate::client_common::ResponseEvent;
use crate::compact::LocalCompactionContext;
use crate::context::ContextualUserFragment;
use crate::context::LocalCompactionRequest;
use crate::context_manager::ContextManager;
use crate::responses_metadata::CodexResponsesMetadata;
use crate::session::session::Session;

pub(super) fn classifier(ids: &[String], guidance: &str) -> CodexResult<LocalCompactionRequest> {
    LocalCompactionRequest::new(
        "LOCAL_COMPACTION_CLASSIFY",
        json!({
            "eligible_ids": ids,
            "max_replacement_bytes": 2800,
            "instructions": "Privately classify each eligible completed tool result exactly once as keep, shorten, or drop. Read current dialogue for relevance. Keep evidence needed for active work, unresolved questions, failures and verification. Shorten must preserve useful exact facts. Drop only dispensable output. Do not classify any other ID. Return JSON only, with no prose or fences.",
            "required_output": {"decisions": [{"id": "eligible source item ID", "action": "keep|shorten|drop", "text": "only for shorten"}]},
            "supplemental_guidance": guidance,
        }),
    )
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

pub(super) async fn infer_json(
    sess: &Session,
    context: &LocalCompactionContext,
    client: &mut ModelClientSession,
    metadata: &CodexResponsesMetadata,
    source: &[ResponseItemEnvelope],
    request: LocalCompactionRequest,
) -> CodexResult<(String, String)> {
    let mut history = ContextManager::default();
    history.replace_annotated(source.to_vec());
    let mut input = history.for_prompt(&context.settings.model_info.input_modalities);
    input.push(ContextualUserFragment::into(request));
    let prompt = Prompt {
        input,
        base_instructions: sess.get_prompt_base_instructions().await,
        ..Default::default()
    };
    let mut stream = client
        .stream(
            &prompt,
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
            ResponseEvent::RateLimits(snapshot) => {
                sess.update_rate_limits(&context.turn, snapshot).await
            }
            ResponseEvent::Completed {
                response_id,
                token_usage,
                usage_metadata,
                ..
            } => {
                sess.record_observed_response_completed(
                    &context.turn,
                    &response_id,
                    token_usage.as_ref(),
                    usage_metadata.as_ref(),
                )
                .await;
                sess.update_token_usage_info(&context.turn, token_usage.as_ref())
                    .await?;
                // Billing accumulates the private request; active occupancy describes real history.
                sess.recompute_token_usage(&context.turn).await;
                return Ok((output, response_id));
            }
            _ => {}
        }
    }
    Err(CodexErr::Stream(
        "local compaction stream closed before response.completed".to_string(),
    ))
}
