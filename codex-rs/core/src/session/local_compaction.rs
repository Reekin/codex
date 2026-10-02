use std::sync::Arc;

use super::session::Session;
use super::thread_settings;
use crate::compact::CompactedHistoryMetadata;
use crate::context::GuardianContextMode;
use crate::context::world_state::WorldState;
use crate::context_manager::estimate_item_token_count;
use crate::local_compaction::LocalCompactionState;
use codex_history::CompactedItem;
use codex_history::ResponseItemEnvelope;
use codex_history::RolloutItem;
use codex_protocol::error::Result as CodexResult;
use codex_protocol::models::ResponseItem;
use codex_protocol::protocol::TurnContextItem;
use codex_protocol::protocol::WorldStateItem;

impl Session {
    pub(crate) async fn get_local_compaction_state(&self) -> LocalCompactionState {
        self.state.lock().await.local_compaction.clone()
    }

    pub(crate) async fn set_local_compaction_state(&self, state: LocalCompactionState) {
        self.state.lock().await.local_compaction = state;
    }

    pub(crate) async fn install_local_compaction(
        self: &Arc<Self>,
        source: &[ResponseItemEnvelope],
        user_revision: u64,
        mut replacement: Vec<ResponseItemEnvelope>,
        reference_context_item: Option<TurnContextItem>,
        world_state_baseline: Option<Arc<WorldState>>,
        metadata: CompactedHistoryMetadata,
        max_history_tokens: usize,
    ) -> CodexResult<bool> {
        let session = Arc::clone(self);
        let source = source.to_vec();
        // Dropping the caller on cancellation leaves the persistence transaction running.
        // Its in-memory commit has no await after the successful durability barrier.
        tokio::spawn(async move {
            let _settings_guard = thread_settings::acquire_persistence_lock(&session).await;
            let settings_event = thread_settings::applied_event(&session).await;
            let mut state = session.state.lock().await;
            if state
                .history
                .conversation_history_snapshot()
                .user_message_revision()
                != user_revision
                || !state.history.annotated_items().starts_with(&source)
            {
                return Ok(false);
            }
            replacement.extend_from_slice(&state.history.annotated_items()[source.len()..]);
            let tokens = replacement.iter().fold(0_usize, |tokens, item| {
                tokens.saturating_add(
                    usize::try_from(estimate_item_token_count(&item.item)).unwrap_or(0),
                )
            });
            if tokens > max_history_tokens {
                return Ok(false);
            }
            for item in &mut replacement {
                Self::assign_missing_response_item_id(&mut item.item);
            }
            if session.guardian_context_mode == GuardianContextMode::ThreadOwned
                && let Some(checkpoint) = replacement.iter_mut().rev().find(|item| {
                    matches!(
                        item.item,
                        ResponseItem::Compaction { .. } | ResponseItem::ContextCompaction { .. }
                    )
                })
            {
                checkpoint
                    .metadata
                    .get_or_insert_default()
                    .compaction_model_hash = metadata.compaction_model_hash;
            }

            let baseline = world_state_baseline
                .map(|world_state| world_state.snapshot())
                .or_else(|| state.history.world_state_baseline());
            let mut history = state.history.clone();
            history.replace_compacted(replacement.clone());
            history.set_reference_context_item(reference_context_item.clone());
            if let Some(snapshot) = baseline.as_ref() {
                history.set_world_state_baseline(snapshot.clone());
            }
            let (window_number, window_ids) = state.next_local_compaction_window();
            let checkpoint = CompactedItem {
                message: metadata.message,
                replacement_history: Some(replacement),
                retained_context: Some(history.retained_context().clone()),
                guardian_history: history.guardian_history_checkpoint(),
                mcp_resource_origins: session.services.mcp_runtime.resource_origin_checkpoint(),
                window_number: Some(window_number),
                first_window_id: Some(window_ids.first_window_id.to_string()),
                previous_window_id: window_ids.previous_window_id.map(|id| id.to_string()),
                window_id: Some(window_ids.window_id.to_string()),
                compaction_response_id: metadata.compaction_response_id,
                latest_token_usage_record: state.latest_token_usage_record.clone(),
            };
            let mut items = vec![RolloutItem::Compacted(checkpoint)];
            if let Some(snapshot) = baseline {
                items.push(RolloutItem::WorldState(WorldStateItem::full(
                    snapshot.into_object(),
                )));
            }
            if let Some(reference) = reference_context_item {
                items.push(RolloutItem::TurnContext(reference));
            }
            items.push(RolloutItem::EventMsg(settings_event));

            if let Some(live_thread) = session.live_thread() {
                // Preserve originals before persisting the replacement. Use the fallible
                // writer directly so failed append/flush cannot install a live-only view.
                live_thread.flush().await.map_err(std::io::Error::other)?;
                live_thread
                    .append_items(&items)
                    .await
                    .map_err(std::io::Error::other)?;
                live_thread.flush().await.map_err(std::io::Error::other)?;
            }
            state.install_local_compacted_history(history, window_number, window_ids);
            state.queue_pending_session_start_source(codex_hooks::SessionStartSource::Compact);
            Ok(true)
        })
        .await?
    }
}
