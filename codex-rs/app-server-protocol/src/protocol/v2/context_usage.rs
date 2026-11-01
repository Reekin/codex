use super::ThreadTokenUsage;
use crate::JsonSchema;
use crate::TS;
use codex_protocol::context_usage::ContextUsage as CoreContextUsage;
use codex_protocol::context_usage::ContextUsageBreakdown as CoreContextUsageBreakdown;
use serde::Deserialize;
use serde::Serialize;

/// Active model context usage; `tokens` is the value that drives automatic compaction.
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq, Eq, JsonSchema, TS)]
#[serde(rename_all = "camelCase")]
#[ts(export_to = "v2/")]
pub struct ThreadContextUsage {
    #[ts(type = "number")]
    pub tokens: i64,
    /// Active context tokens at which automatic compaction triggers, when it can trigger.
    #[ts(type = "number | null")]
    pub auto_compact_token_limit: Option<i64>,
    /// Estimated composition of `tokens`; categories sum exactly to `tokens`.
    pub breakdown: ThreadContextUsageBreakdown,
}

#[derive(Serialize, Deserialize, Debug, Clone, PartialEq, Eq, JsonSchema, TS)]
#[serde(rename_all = "camelCase")]
#[ts(export_to = "v2/")]
pub struct ThreadContextUsageBreakdown {
    #[ts(type = "number")]
    pub base_instructions: i64,
    #[ts(type = "number")]
    pub developer_instructions: i64,
    #[ts(type = "number")]
    pub agents_md: i64,
    #[ts(type = "number")]
    pub skills: i64,
    #[ts(type = "number")]
    pub tools: i64,
    #[ts(type = "number")]
    pub environment: i64,
    #[ts(type = "number")]
    pub user_messages: i64,
    #[ts(type = "number")]
    pub agent_messages: i64,
    #[ts(type = "number")]
    pub tool_calls: i64,
    #[ts(type = "number")]
    pub reasoning: i64,
    #[ts(type = "number")]
    pub compaction: i64,
    #[ts(type = "number")]
    pub other: i64,
}

impl From<CoreContextUsage> for ThreadContextUsage {
    fn from(value: CoreContextUsage) -> Self {
        Self {
            tokens: value.tokens,
            auto_compact_token_limit: value.auto_compact_token_limit,
            breakdown: value.breakdown.into(),
        }
    }
}

impl From<CoreContextUsageBreakdown> for ThreadContextUsageBreakdown {
    fn from(value: CoreContextUsageBreakdown) -> Self {
        Self {
            base_instructions: value.base_instructions,
            developer_instructions: value.developer_instructions,
            agents_md: value.agents_md,
            skills: value.skills,
            tools: value.tools,
            environment: value.environment,
            user_messages: value.user_messages,
            agent_messages: value.agent_messages,
            tool_calls: value.tool_calls,
            reasoning: value.reasoning,
            compaction: value.compaction,
            other: value.other,
        }
    }
}

#[derive(Serialize, Deserialize, Debug, Clone, PartialEq, JsonSchema, TS)]
#[serde(rename_all = "camelCase")]
#[ts(export_to = "v2/")]
pub struct ThreadTokenUsageReadParams {
    pub thread_id: String,
}

#[derive(Serialize, Deserialize, Debug, Clone, PartialEq, JsonSchema, TS)]
#[serde(rename_all = "camelCase")]
#[ts(export_to = "v2/")]
pub struct ThreadTokenUsageReadResponse {
    /// The last token usage recorded in each turn of the thread's own history, in turn order.
    pub data: Vec<TurnTokenUsage>,
}

#[derive(Serialize, Deserialize, Debug, Clone, PartialEq, JsonSchema, TS)]
#[serde(rename_all = "camelCase")]
#[ts(export_to = "v2/")]
pub struct TurnTokenUsage {
    pub turn_id: String,
    pub token_usage: ThreadTokenUsage,
}
