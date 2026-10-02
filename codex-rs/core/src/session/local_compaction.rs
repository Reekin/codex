use std::sync::Arc;

use super::session::Session;
use super::thread_settings;
use super::turn_context::TurnContext;
use crate::compact::CompactedHistoryMetadata;
use crate::compact::LocalCompactionContext;
use crate::context::GuardianContextMode;
use crate::context::world_state::WorldState;
use crate::context_manager::estimate_item_token_count;
use crate::local_compaction::AnalysisKind;
use crate::local_compaction::LocalCompactionState;
use codex_history::CompactedItem;
use codex_history::ResponseItemEnvelope;
use codex_history::RolloutItem;
use codex_protocol::ThreadId;
use codex_protocol::error::Result as CodexResult;
use codex_protocol::models::ResponseItem;
use codex_protocol::protocol::EventMsg;
use codex_protocol::protocol::TokenCountEvent;
use codex_protocol::protocol::TokenUsage;
use codex_protocol::protocol::TokenUsageInfo;
use codex_protocol::protocol::TurnContextItem;
use codex_protocol::protocol::WorldStateItem;

impl Session {
    pub(crate) async fn get_local_compaction_state(&self) -> LocalCompactionState {
        self.state.lock().await.local_compaction.clone()
    }

    pub(crate) async fn set_local_compaction_state(&self, state: LocalCompactionState) {
        let mut current = self.state.lock().await;
        if current.local_compaction.generation == state.generation {
            current.local_compaction = state;
        }
    }

    /// Held across a cleanup cycle so automatic and manual cleanup cannot overwrite each other.
    pub(crate) async fn lock_local_compaction_boundary(&self) -> tokio::sync::OwnedMutexGuard<()> {
        let boundary = self.state.lock().await.local_compaction.boundary();
        boundary.lock_owned().await
    }

    /// The thread this one was forked from, whose validated tool marks it may inherit.
    pub(crate) async fn forked_from_thread_id(&self) -> Option<ThreadId> {
        self.state
            .lock()
            .await
            .session_configuration
            .forked_from_thread_id
    }

    /// Out-of-turn cleanup reports against the running turn when there is one.
    pub(crate) async fn local_compaction_turn(self: &Arc<Self>) -> Arc<TurnContext> {
        match self.active_turn_context_and_cancellation_token().await {
            Some((turn, _)) => turn,
            None => self.new_default_turn().await,
        }
    }

    pub(crate) async fn cancel_local_compaction(&self) {
        let mut state = self.state.lock().await;
        state.local_compaction.reset_window();
        state.background_compaction_usage = None;
    }

    pub(crate) async fn begin_local_compaction_usage(
        &self,
        turn_id: &str,
        generation: u64,
    ) -> bool {
        let mut state = self.state.lock().await;
        if state.local_compaction.generation != generation {
            return false;
        }
        let usage = state
            .latest_token_usage_record
            .as_ref()
            .filter(|record| record.turn_id == turn_id)
            .map(|record| record.turn_token_usage.clone())
            .unwrap_or_default();
        state.background_compaction_usage = Some((turn_id.to_string(), usage));
        true
    }

    pub(crate) async fn local_compaction_hard_limit(
        &self,
        context: &LocalCompactionContext,
    ) -> usize {
        super::context_window::context_window_token_status_for_model(
            self,
            &context.turn.config,
            &context.turn,
            &context.settings.model_info,
        )
        .await
        .auto_compact_token_limit
        .and_then(|limit| usize::try_from(limit).ok())
        .unwrap_or(128_000)
    }

    /// Charge private inference without changing foreground occupancy or completion state.
    pub(crate) async fn record_local_compaction_usage(
        &self,
        turn: &TurnContext,
        response_id: &str,
        usage: Option<&TokenUsage>,
        kind: AnalysisKind,
    ) -> CodexResult<()> {
        let Some(usage) = usage else {
            return Ok(());
        };
        let record = {
            let mut state = self.state.lock().await;
            let mut info = state.token_info().unwrap_or(TokenUsageInfo {
                total_token_usage: TokenUsage::default(),
                last_token_usage: TokenUsage::default(),
                model_context_window: None,
                context_usage: None,
            });
            info.total_token_usage.add_assign(usage);
            state.set_token_info(Some(info));
            let foreground = (kind == AnalysisKind::Background)
                .then(|| state.latest_token_usage_record.clone())
                .flatten()
                .filter(|record| record.turn_id != turn.sub_id);
            let record = state.record_token_usage(
                self.thread_id,
                &turn.sub_id,
                self.session_id(),
                turn.turn_metadata_state
                    .root_turn_id()
                    .unwrap_or_else(|| turn.sub_id.clone()),
                response_id.to_string(),
                usage,
            );
            if let Some(mut foreground) = foreground {
                foreground.thread_token_usage = record.thread_token_usage.clone();
                state.latest_token_usage_record = Some(foreground);
            }
            if kind == AnalysisKind::Background {
                state.background_compaction_usage = None;
            }
            record
        };
        self.persist_rollout_items(&[RolloutItem::TokenUsageRecord(record)])
            .await;
        self.record_rollout_budget_usage(usage)
    }

    pub(crate) async fn send_local_compaction_token_count(&self, turn: &TurnContext) {
        let (info, rate_limits) = self.state.lock().await.token_info_and_rate_limits();
        self.send_event(
            turn,
            EventMsg::TokenCount(TokenCountEvent { info, rate_limits }),
        )
        .await;
    }

    pub(crate) async fn install_local_compaction(
        self: &Arc<Self>,
        source: &[ResponseItemEnvelope],
        user_revision: u64,
        generation: u64,
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
            if state.local_compaction.generation != generation
                || state
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
