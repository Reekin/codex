//! Estimates which kinds of content make up the active model context.
//!
//! Each history item is sized with the same byte heuristic used for full-history estimates and
//! assigned to a stable category. The per-category estimates are then scaled so they sum exactly
//! to the active context tokens that drive automatic compaction.

use codex_protocol::context_usage::ContextUsage;
use codex_protocol::context_usage::ContextUsageBreakdown;
use codex_protocol::context_usage::ContextUsageCategory;
use codex_protocol::models::ContentItem;
use codex_protocol::models::ResponseItem;
use codex_tools::ToolSpec;
use codex_tools::create_tools_json_for_responses_api;
use codex_utils_output_truncation::approx_token_count;

use super::estimate_image_bytes;
use super::estimate_item_token_count;

/// Request-level context that is not part of conversation history.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) struct RequestOverhead {
    pub(crate) base_instructions_tokens: i64,
    pub(crate) tools_tokens: i64,
}

impl RequestOverhead {
    pub(crate) fn new(base_instructions: &str, tools_tokens: i64) -> Self {
        Self {
            base_instructions_tokens: to_i64(approx_token_count(base_instructions)),
            tools_tokens,
        }
    }
}

/// Estimates the tokens used by tool definitions sent with a sampling request.
pub(crate) fn estimate_tools_tokens(tools: &[ToolSpec]) -> i64 {
    let bytes = create_tools_json_for_responses_api(tools)
        .and_then(|tools| serde_json::to_string(&tools))
        .map_or(0, |json| json.len());
    to_i64(bytes.div_ceil(4))
}

/// Builds the context usage record for `tokens` of active context.
pub(crate) fn context_usage<'a>(
    items: impl IntoIterator<Item = &'a ResponseItem>,
    overhead: RequestOverhead,
    tokens: i64,
    auto_compact_token_limit: Option<i64>,
) -> ContextUsage {
    let mut estimates = ContextUsageBreakdown {
        base_instructions: overhead.base_instructions_tokens,
        tools: overhead.tools_tokens,
        ..Default::default()
    };
    for item in items {
        add_item_estimate(&mut estimates, item);
    }
    ContextUsage {
        tokens,
        auto_compact_token_limit,
        breakdown: scale_breakdown(&estimates, tokens),
    }
}

fn add_item_estimate(breakdown: &mut ContextUsageBreakdown, item: &ResponseItem) {
    let tokens = estimate_item_token_count(item);
    let category = match item {
        ResponseItem::Message {
            role,
            content,
            internal_chat_message_metadata_passthrough,
            ..
        } => {
            let kinds = internal_chat_message_metadata_passthrough
                .as_ref()
                .and_then(|metadata| metadata.content_item_kinds.as_ref())
                .filter(|kinds| kinds.len() == content.len() && !kinds.is_empty());
            if let Some(kinds) = kinds {
                let parts = content
                    .iter()
                    .zip(kinds)
                    .map(|(content, kind)| (category_for_kind(&kind.0), content_weight(content)));
                distribute(breakdown, tokens, parts.collect());
                return;
            }
            category_for_role(role)
        }
        ResponseItem::Reasoning { .. } => ContextUsageCategory::Reasoning,
        ResponseItem::LocalShellCall { .. }
        | ResponseItem::FunctionCall { .. }
        | ResponseItem::FunctionCallOutput { .. }
        | ResponseItem::CustomToolCall { .. }
        | ResponseItem::CustomToolCallOutput { .. }
        | ResponseItem::ToolSearchCall { .. }
        | ResponseItem::ToolSearchOutput { .. }
        | ResponseItem::WebSearchCall { .. }
        | ResponseItem::ImageGenerationCall { .. } => ContextUsageCategory::ToolCalls,
        ResponseItem::AdditionalTools { .. } => ContextUsageCategory::Tools,
        ResponseItem::Compaction { .. }
        | ResponseItem::ContextCompaction { .. }
        | ResponseItem::CompactionTrigger { .. } => ContextUsageCategory::Compaction,
        ResponseItem::AgentMessage { .. }
        | ResponseItem::ConfigurationUpdate { .. }
        | ResponseItem::Other => ContextUsageCategory::Other,
    };
    add(breakdown, category, tokens);
}

fn category_for_role(role: &str) -> ContextUsageCategory {
    match role {
        "assistant" => ContextUsageCategory::AgentMessages,
        "developer" => ContextUsageCategory::DeveloperInstructions,
        "user" => ContextUsageCategory::UserMessages,
        "system" => ContextUsageCategory::BaseInstructions,
        _ => ContextUsageCategory::Other,
    }
}

/// Maps a harness content classification (`<feature>.<name>`) to its category.
fn category_for_kind(kind: &str) -> ContextUsageCategory {
    match kind {
        "generic.developer_instructions"
        | "managed_config.developer_instructions"
        | "multi_agent.role_instructions"
        | "guardian.policy" => return ContextUsageCategory::DeveloperInstructions,
        "multi_agent.mode_instructions"
        | "multi_agent.usage_hint"
        | "multi_agent.subagent_identity" => return ContextUsageCategory::Environment,
        _ => {}
    }
    let feature = kind.split_once('.').map_or(kind, |(feature, _)| feature);
    match feature {
        "model" | "model_switch" | "personality" | "persistent_mode" => {
            ContextUsageCategory::BaseInstructions
        }
        "agents_md" => ContextUsageCategory::AgentsMd,
        "skills"
        | "host_skills"
        | "plugins"
        | "apps"
        | "hosted_plugin_runtime"
        | "selected_executor_plugin_mcp" => ContextUsageCategory::Skills,
        "tools" => ContextUsageCategory::Tools,
        "environments"
        | "permissions"
        | "approved_command_prefixes"
        | "collaboration_mode"
        | "token_budget"
        | "context_window_guidance"
        | "realtime"
        | "realtime_conversation"
        | "current_time"
        | "rollout_budget"
        | "network_proxy"
        | "notes"
        | "multi_agent_mode"
        | "multi_agent_usage_hint"
        | "subagent_identity" => ContextUsageCategory::Environment,
        "user" | "shell" | "additional_content" => ContextUsageCategory::UserMessages,
        "compaction" => ContextUsageCategory::Compaction,
        _ => ContextUsageCategory::Other,
    }
}

fn content_weight(content: &ContentItem) -> i64 {
    match content {
        ContentItem::InputText { text } | ContentItem::OutputText { text } => to_i64(text.len()),
        ContentItem::InputImage { image_url, detail } => estimate_image_bytes(image_url, *detail),
        ContentItem::InputAudio { audio_url } => to_i64(audio_url.len()),
    }
}

/// Splits `tokens` across `parts` in proportion to their weights with an exact sum.
fn distribute(
    breakdown: &mut ContextUsageBreakdown,
    tokens: i64,
    parts: Vec<(ContextUsageCategory, i64)>,
) {
    let total_weight: i64 = parts.iter().map(|(_, weight)| *weight).sum();
    if total_weight <= 0 {
        if let Some((category, _)) = parts.first() {
            add(breakdown, *category, tokens);
        }
        return;
    }
    let mut cumulative_weight = 0;
    let mut allocated = 0;
    for (category, weight) in parts {
        cumulative_weight += weight;
        let target = proportional(tokens, cumulative_weight, total_weight);
        add(breakdown, category, target - allocated);
        allocated = target;
    }
}

/// Scales category estimates so they sum exactly to `tokens`.
fn scale_breakdown(estimates: &ContextUsageBreakdown, tokens: i64) -> ContextUsageBreakdown {
    let tokens = tokens.max(0);
    let total_estimate: i64 = ContextUsageCategory::ALL
        .iter()
        .map(|category| estimates.tokens(*category).max(0))
        .sum();
    let mut scaled = ContextUsageBreakdown::default();
    if total_estimate == 0 {
        scaled.other = tokens;
        return scaled;
    }
    let mut cumulative_estimate = 0;
    let mut allocated = 0;
    for category in ContextUsageCategory::ALL {
        cumulative_estimate += estimates.tokens(category).max(0);
        let target = proportional(tokens, cumulative_estimate, total_estimate);
        *scaled.tokens_mut(category) = target - allocated;
        allocated = target;
    }
    scaled
}

fn proportional(value: i64, numerator: i64, denominator: i64) -> i64 {
    let scaled = i128::from(value) * i128::from(numerator) / i128::from(denominator);
    i64::try_from(scaled).unwrap_or(i64::MAX)
}

fn add(breakdown: &mut ContextUsageBreakdown, category: ContextUsageCategory, tokens: i64) {
    let slot = breakdown.tokens_mut(category);
    *slot = slot.saturating_add(tokens.max(0));
}

fn to_i64(value: usize) -> i64 {
    i64::try_from(value).unwrap_or(i64::MAX)
}

#[cfg(test)]
#[path = "context_usage_tests.rs"]
mod tests;
