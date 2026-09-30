//! Active model context usage and its per-category composition.

use schemars::JsonSchema;
use serde::Deserialize;
use serde::Serialize;
use ts_rs::TS;

/// Active model context usage recorded with a token usage snapshot.
#[derive(Debug, Clone, Default, PartialEq, Eq, Deserialize, Serialize, JsonSchema, TS)]
pub struct ContextUsage {
    /// Active context tokens; the same value drives automatic compaction.
    #[ts(type = "number")]
    pub tokens: i64,
    /// Active context tokens at which automatic compaction triggers, when it can trigger.
    #[ts(type = "number | null")]
    pub auto_compact_token_limit: Option<i64>,
    /// Estimated composition of `tokens`; categories sum exactly to `tokens`.
    pub breakdown: ContextUsageBreakdown,
}

/// Stable categories of model-visible context.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ContextUsageCategory {
    BaseInstructions,
    DeveloperInstructions,
    AgentsMd,
    Skills,
    Tools,
    Environment,
    UserMessages,
    AgentMessages,
    ToolCalls,
    Reasoning,
    Compaction,
    Other,
}

impl ContextUsageCategory {
    pub const ALL: [Self; 12] = [
        Self::BaseInstructions,
        Self::DeveloperInstructions,
        Self::AgentsMd,
        Self::Skills,
        Self::Tools,
        Self::Environment,
        Self::UserMessages,
        Self::AgentMessages,
        Self::ToolCalls,
        Self::Reasoning,
        Self::Compaction,
        Self::Other,
    ];
}

/// Token count per [`ContextUsageCategory`].
#[derive(Debug, Clone, Default, PartialEq, Eq, Deserialize, Serialize, JsonSchema, TS)]
pub struct ContextUsageBreakdown {
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

impl ContextUsageBreakdown {
    pub fn tokens(&self, category: ContextUsageCategory) -> i64 {
        match category {
            ContextUsageCategory::BaseInstructions => self.base_instructions,
            ContextUsageCategory::DeveloperInstructions => self.developer_instructions,
            ContextUsageCategory::AgentsMd => self.agents_md,
            ContextUsageCategory::Skills => self.skills,
            ContextUsageCategory::Tools => self.tools,
            ContextUsageCategory::Environment => self.environment,
            ContextUsageCategory::UserMessages => self.user_messages,
            ContextUsageCategory::AgentMessages => self.agent_messages,
            ContextUsageCategory::ToolCalls => self.tool_calls,
            ContextUsageCategory::Reasoning => self.reasoning,
            ContextUsageCategory::Compaction => self.compaction,
            ContextUsageCategory::Other => self.other,
        }
    }

    pub fn tokens_mut(&mut self, category: ContextUsageCategory) -> &mut i64 {
        match category {
            ContextUsageCategory::BaseInstructions => &mut self.base_instructions,
            ContextUsageCategory::DeveloperInstructions => &mut self.developer_instructions,
            ContextUsageCategory::AgentsMd => &mut self.agents_md,
            ContextUsageCategory::Skills => &mut self.skills,
            ContextUsageCategory::Tools => &mut self.tools,
            ContextUsageCategory::Environment => &mut self.environment,
            ContextUsageCategory::UserMessages => &mut self.user_messages,
            ContextUsageCategory::AgentMessages => &mut self.agent_messages,
            ContextUsageCategory::ToolCalls => &mut self.tool_calls,
            ContextUsageCategory::Reasoning => &mut self.reasoning,
            ContextUsageCategory::Compaction => &mut self.compaction,
            ContextUsageCategory::Other => &mut self.other,
        }
    }
}
