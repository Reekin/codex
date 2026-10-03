use super::context_window::context_window_token_status;
use super::session::Session;
use super::turn_context::TurnContext;
use crate::context_manager::context_usage::PriorReasoning;
use crate::context_manager::context_usage::RequestOverhead;
use crate::context_manager::context_usage::context_usage;
use codex_protocol::models::ResponseItem;
use codex_protocol::protocol::TokenUsage;

/// Tracks whether a sampled response's reported input measures the active context.
///
/// Server-side web search runs another model pass inside the same response, and each pass reads
/// the whole context again, so the provider reports the context several times over. Such usage
/// still counts toward totals, but the active context is the request's own active usage plus
/// what the response generated.
#[derive(Default)]
pub(crate) struct ResponseContext {
    request_tokens: i64,
    server_search: bool,
}

impl ResponseContext {
    pub(crate) fn new(request_tokens: i64) -> Self {
        Self {
            request_tokens,
            server_search: false,
        }
    }

    pub(crate) fn observe(&mut self, item: &ResponseItem) {
        self.server_search |= matches!(item, ResponseItem::WebSearchCall { .. });
    }

    /// Usage that measures the active context after this response.
    pub(crate) fn active_usage(&self, usage: &TokenUsage) -> TokenUsage {
        if !self.server_search {
            return usage.clone();
        }
        let input_tokens = usage.input_tokens.min(self.request_tokens);
        TokenUsage {
            input_tokens,
            cached_input_tokens: usage.cached_input_tokens.min(input_tokens),
            cache_write_input_tokens: usage.cache_write_input_tokens.min(input_tokens),
            output_tokens: usage.output_tokens,
            reasoning_output_tokens: usage.reasoning_output_tokens,
            total_tokens: input_tokens.saturating_add(usage.output_tokens),
            codex_rollout_budget_units: None,
        }
    }
}

impl Session {
    /// Remembers the tool-definition size of the prompt being sampled.
    pub(crate) async fn record_request_tools_tokens(&self, tools_tokens: i64) {
        self.state.lock().await.request_tools_tokens = tools_tokens;
    }

    /// Attaches the current active context usage to the latest token usage snapshot.
    pub(crate) async fn refresh_context_usage(&self, turn_context: &TurnContext) {
        if self.state.lock().await.token_info().is_none() {
            return;
        }
        let status = context_window_token_status(self, turn_context).await;
        let base_instructions = self.get_prompt_base_instructions().await;
        let prior_reasoning = if turn_context.model_info().retains_prior_reasoning {
            PriorReasoning::Retained
        } else {
            PriorReasoning::Dropped
        };
        let mut state = self.state.lock().await;
        let Some(mut info) = state.token_info() else {
            return;
        };
        let overhead = RequestOverhead::new(&base_instructions.text, state.request_tools_tokens);
        let items: Vec<_> = state.history.raw_items().collect();
        info.context_usage = Some(context_usage(
            &items,
            overhead,
            prior_reasoning,
            status.active_context_tokens,
            status.auto_compact_token_limit,
        ));
        state.set_token_info(Some(info));
    }
}
