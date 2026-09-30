//! Serves `thread/tokenUsage/read`: the last token usage recorded in each turn of a thread.

use codex_app_server_protocol::ClientResponsePayload;
use codex_app_server_protocol::JSONRPCErrorError;
use codex_app_server_protocol::ThreadHistoryBuilder;
use codex_app_server_protocol::ThreadTokenUsage;
use codex_app_server_protocol::ThreadTokenUsageReadParams;
use codex_app_server_protocol::ThreadTokenUsageReadResponse;
use codex_app_server_protocol::TurnTokenUsage;
use codex_protocol::ThreadId;
use codex_protocol::protocol::EventMsg;
use codex_rollout::RolloutItem;
use codex_rollout::RolloutRecorder;

use super::ThreadRequestProcessor;
use crate::error_code::internal_error;
use crate::error_code::invalid_request;

impl ThreadRequestProcessor {
    pub(crate) async fn thread_token_usage_read(
        &self,
        params: ThreadTokenUsageReadParams,
    ) -> Result<Option<ClientResponsePayload>, JSONRPCErrorError> {
        let thread_id = ThreadId::from_string(&params.thread_id)
            .map_err(|err| invalid_request(format!("invalid thread id: {err}")))?;
        // The rollout is the persisted source of token usage for both history modes.
        let rollout_path = match self.thread_manager.get_thread(thread_id).await {
            Ok(thread) => {
                thread
                    .flush_rollout()
                    .await
                    .map_err(|err| internal_error(format!("failed to flush rollout: {err}")))?;
                thread.rollout_path()
            }
            Err(_) => codex_rollout::find_thread_path_by_id_str(
                &self.config.codex_home,
                &params.thread_id,
                self.state_db.as_deref(),
            )
            .await
            .map_err(|err| internal_error(format!("failed to locate thread {thread_id}: {err}")))?,
        }
        .ok_or_else(|| invalid_request(format!("thread not found: {thread_id}")))?;
        let (items, _, _) = RolloutRecorder::load_rollout_items(&rollout_path)
            .await
            .map_err(|err| internal_error(format!("failed to read thread {thread_id}: {err}")))?;
        Ok(Some(
            ThreadTokenUsageReadResponse {
                data: turn_token_usages(&items),
            }
            .into(),
        ))
    }
}

/// Attributes each persisted token usage snapshot to the turn that recorded it and keeps the
/// last snapshot per turn, in turn order.
fn turn_token_usages(rollout_items: &[RolloutItem]) -> Vec<TurnTokenUsage> {
    let mut builder = ThreadHistoryBuilder::new();
    let mut last_turn_id: Option<String> = None;
    let mut usages: Vec<TurnTokenUsage> = Vec::new();
    for item in rollout_items {
        builder.handle_rollout_item(item);
        if let Some(turn_id) = builder.active_turn_id() {
            last_turn_id = Some(turn_id.to_string());
        }
        let RolloutItem::EventMsg(EventMsg::TokenCount(event)) = item else {
            continue;
        };
        let (Some(info), Some(turn_id)) = (event.info.as_ref(), last_turn_id.as_ref()) else {
            continue;
        };
        let token_usage = ThreadTokenUsage::from(info.clone());
        match usages.last_mut() {
            Some(last) if &last.turn_id == turn_id => last.token_usage = token_usage,
            _ => usages.push(TurnTokenUsage {
                turn_id: turn_id.clone(),
                token_usage,
            }),
        }
    }
    usages
}

#[cfg(test)]
#[path = "thread_token_usage_read_tests.rs"]
mod tests;
