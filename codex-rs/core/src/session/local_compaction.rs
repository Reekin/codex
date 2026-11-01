//! Session hooks for local context compaction: marking state, private usage accounting and the
//! durable installation of a replacement view.

use std::sync::Arc;

use codex_extension_api::ConversationHistorySnapshot as _;
use codex_history::CompactedItem;
use codex_history::CompactionResumeMetadata;
use codex_history::DynamicToolsSnapshot;
use codex_history::ResponseItemEnvelope;
use codex_history::RolloutItem;
use codex_protocol::ThreadId;
use codex_protocol::error::CodexErr;
use codex_protocol::error::Result as CodexResult;
use codex_protocol::protocol::EventMsg;
use codex_protocol::protocol::TokenCountEvent;
use codex_protocol::protocol::TokenUsage;
use codex_protocol::protocol::TurnContextItem;
use codex_protocol::protocol::WorldStateItem;
use uuid::Uuid;

use super::TokenUsageContext;
use super::session::Session;
use super::step_settings::ResolvedStepSettings;
use super::thread_settings;
use super::turn_context::TurnContext;
use crate::context::world_state::WorldStateSnapshot;
use crate::context_manager::context_usage::RequestOverhead;
use crate::context_manager::estimate_item_token_count;
use crate::local_compaction::AnalysisKind;
use crate::local_compaction::LocalCompactionState;
use crate::state::AutoCompactWindowIds;
use crate::state::ReasoningEffortPin;

/// A replacement view and the conditions under which it may still be installed.
pub(crate) struct LocalInstall {
    /// History the replacement was computed from; it must still be a prefix of the live history.
    pub(crate) source: Vec<ResponseItemEnvelope>,
    pub(crate) user_revision: u64,
    /// Full window the replacement belongs to.
    pub(crate) generation: u64,
    pub(crate) replacement: Vec<ResponseItemEnvelope>,
    /// The fresh instruction baseline of a full compaction; tool cleanup keeps the current one.
    pub(crate) fresh_baseline: Option<(TurnContextItem, WorldStateSnapshot)>,
    pub(crate) message: String,
    pub(crate) compaction_response_id: Option<String>,
    pub(crate) max_history_tokens: usize,
}

impl Session {
    pub(crate) async fn local_compaction_state(&self) -> LocalCompactionState {
        self.state.lock().await.local_compaction.clone()
    }

    /// Stores the state of a cleanup cycle unless a full-window reset happened meanwhile.
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

    /// Base-instruction and tool-definition size of the latest sampled prompt.
    pub(crate) async fn request_overhead(&self) -> RequestOverhead {
        self.state.lock().await.request_overhead
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

    /// Remembers the originating turn's usage so a batch finishing in a later turn still adds
    /// to it. Returns false when the full window changed since the batch was packed.
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

    /// The hard limit of `model_info` under the turn's configuration, in provider tokens.
    pub(crate) async fn local_compaction_hard_limit(
        &self,
        turn: &TurnContext,
        model_info: &codex_protocol::openai_models::ModelInfo,
    ) -> usize {
        super::context_window::context_window_token_status_for_model(
            self,
            &turn.config,
            turn,
            model_info,
        )
        .await
        .auto_compact_token_limit
        .and_then(|limit| usize::try_from(limit).ok())
        .unwrap_or(128_000)
    }

    /// Charges private inference to the thread's totals without changing foreground
    /// occupancy, the foreground turn's usage record or completion state.
    pub(crate) async fn record_local_compaction_usage(
        &self,
        turn: &TurnContext,
        settings: &ResolvedStepSettings,
        response_id: &str,
        usage: Option<&TokenUsage>,
        kind: AnalysisKind,
    ) -> CodexResult<()> {
        let Some(usage) = usage else {
            return Ok(());
        };
        let record = {
            let mut state = self.state.lock().await;
            let foreground = (kind == AnalysisKind::Background)
                .then(|| state.latest_token_usage_record.clone())
                .flatten();
            let record = state.record_token_usage(
                self.thread_id,
                &turn.sub_id,
                self.session_id(),
                turn.root_turn_id(),
                response_id.to_string(),
                usage,
            );
            if let Some(mut foreground) = foreground {
                foreground.thread_token_usage = record.thread_token_usage.clone();
                if foreground.turn_id == record.turn_id {
                    foreground.turn_token_usage = record.turn_token_usage.clone();
                }
                state.latest_token_usage_record = Some(foreground);
            }
            if kind == AnalysisKind::Background {
                state.background_compaction_usage = None;
            }
            record
        };
        self.persist_rollout_items(&[RolloutItem::TokenUsageRecord(record)])
            .await;
        self.record_token_usage_info(turn, settings, Some(usage), TokenUsageContext::SpendingOnly)
            .await
    }

    /// Reports the charged totals without recomputing the foreground snapshot; the request
    /// may belong to a model used in an earlier turn.
    pub(crate) async fn send_local_compaction_token_count(&self, turn: &TurnContext) {
        let (info, rate_limits) = self.state.lock().await.token_info_and_rate_limits();
        self.send_event(
            turn,
            EventMsg::TokenCount(TokenCountEvent { info, rate_limits }),
        )
        .await;
    }

    /// Persists the originals and a consistent checkpoint, then installs `install` as the live
    /// history. Returns false when the source changed or the result would not fit.
    pub(crate) async fn install_local_compaction(
        self: &Arc<Self>,
        install: LocalInstall,
    ) -> CodexResult<bool> {
        let session = Arc::clone(self);
        // Dropping the caller on cancellation leaves the persistence transaction running.
        // Its in-memory commit has no await after the successful durability barrier.
        tokio::spawn(async move {
            let _settings_guard = thread_settings::acquire_persistence_lock(&session).await;
            let settings_event = thread_settings::applied_event(&session).await;
            let mut state = session.state.lock().await;
            let LocalInstall {
                source,
                user_revision,
                generation,
                mut replacement,
                fresh_baseline,
                message,
                compaction_response_id,
                max_history_tokens,
            } = install;
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
            let mcp_revision = session
                .services
                .executed_tool_calls
                .mcp_attribution_checkpoint(/*force*/ true)
                .and_then(|(attribution, revision)| {
                    replacement.last_mut().map(|envelope| {
                        envelope.metadata.get_or_insert_default().mcp_attribution =
                            Some(attribution);
                        revision
                    })
                });
            let full = fresh_baseline.is_some();
            let (reference, baseline) = match fresh_baseline {
                Some((reference, baseline)) => (Some(reference), Some(baseline)),
                None => (
                    state.reference_context_item(),
                    state
                        .history
                        .world_state_checkpoint()
                        .map(|checkpoint| WorldStateSnapshot::from(&checkpoint.state)),
                ),
            };
            let mut history = state.history.clone();
            let invalidate_reviews = history
                .replace_compacted(replacement.clone(), /*reviewer_compaction_hash*/ None);
            history.set_reference_context_item(reference.clone());
            if let Some(baseline) = baseline.as_ref() {
                history.set_world_state_baseline(baseline.clone());
            }
            let (window_number, window_ids) = next_window(
                state.auto_compact_window_number(),
                state.auto_compact_window_ids(),
            );
            let checkpoint = CompactedItem {
                message,
                replacement_history: Some(replacement),
                guardian_history: history.guardian_history_checkpoint(),
                retained_context: Some(history.retained_context().clone()),
                mcp_resource_origins: session.services.mcp_runtime.resource_origin_checkpoint(),
                window_number: Some(window_number),
                first_window_id: Some(window_ids.first_window_id.to_string()),
                previous_window_id: window_ids.previous_window_id.map(|id| id.to_string()),
                window_id: Some(window_ids.window_id.to_string()),
                compaction_response_id,
                latest_token_usage_record: state.latest_token_usage_record.clone(),
                resume_metadata: Some(CompactionResumeMetadata {
                    dynamic_tools: Some(Box::new(DynamicToolsSnapshot {
                        thread_id: session.thread_id(),
                        tools: state.persisted_dynamic_tools.to_vec(),
                    })),
                    multi_agent_version: session.multi_agent_version(),
                    last_started_turn_id: state.last_started_turn_id.clone(),
                    turn_attribution: state.turn_attribution.clone(),
                    previous_turn_settings: state.previous_turn_settings(),
                }),
            };
            let mut items = vec![RolloutItem::Compacted(Box::new(checkpoint))];
            if let Some(baseline) = baseline {
                items.push(RolloutItem::WorldState(WorldStateItem::full(
                    baseline.into_object(),
                )));
            }
            if let Some(reference) = reference {
                items.push(RolloutItem::TurnContext(reference));
            }
            items.push(RolloutItem::EventMsg(settings_event));
            if let Some(live_thread) = session.live_thread() {
                // Preserve originals before persisting the replacement. Use the fallible
                // writer directly so a failed append or flush cannot install a live-only view.
                live_thread.flush().await.map_err(std::io::Error::other)?;
                live_thread
                    .append_items(&items)
                    .await
                    .map_err(std::io::Error::other)?;
                live_thread.flush().await.map_err(std::io::Error::other)?;
            }
            if let Some(revision) = mcp_revision {
                session
                    .services
                    .executed_tool_calls
                    .mark_mcp_attribution_persisted(revision);
            }
            state.install_local_compacted_history(
                history,
                invalidate_reviews,
                window_number,
                window_ids,
            );
            if full {
                state.reasoning_effort_pin = ReasoningEffortPin::Compacted;
            }
            state.queue_pending_session_start_source(codex_hooks::SessionStartSource::Compact);
            Ok::<_, CodexErr>(true)
        })
        .await?
    }
}

/// The window a replacement view opens: same lineage, a new ID.
fn next_window(number: u64, ids: AutoCompactWindowIds) -> (u64, AutoCompactWindowIds) {
    (
        number.saturating_add(1),
        AutoCompactWindowIds {
            first_window_id: ids.first_window_id,
            previous_window_id: Some(ids.window_id),
            window_id: Uuid::now_v7(),
        },
    )
}
