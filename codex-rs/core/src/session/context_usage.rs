use super::context_window::context_window_token_status;
use super::session::Session;
use super::turn_context::TurnContext;
use crate::context_manager::context_usage::RequestOverhead;
use crate::context_manager::context_usage::context_usage;

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
        let mut state = self.state.lock().await;
        let Some(mut info) = state.token_info() else {
            return;
        };
        let overhead = RequestOverhead::new(&base_instructions.text, state.request_tools_tokens);
        info.context_usage = Some(context_usage(
            state.history.raw_items(),
            overhead,
            status.active_context_tokens,
            status.auto_compact_token_limit,
        ));
        state.set_token_info(Some(info));
    }
}
