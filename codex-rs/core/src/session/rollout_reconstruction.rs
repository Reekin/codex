use super::*;
use crate::chat_tree::ReplayedChatTree;
use crate::context_manager::is_user_turn_boundary;
use codex_protocol::chat_tree::ChatTreeEvent;
use codex_protocol::chat_tree::ChatTreeState as DomainChatTreeState;
use codex_protocol::chat_tree_protocol::chat_tree_event_from_protocol_event;
use std::collections::HashSet;
use tracing::warn;
use uuid::Uuid;

// Return value of `Session::reconstruct_history_from_rollout`, bundling the rebuilt history with
// the resume/fork hydration metadata derived from the same replay.
#[derive(Debug)]
pub(super) struct RolloutReconstruction {
    pub(super) history: Vec<ResponseItem>,
    pub(super) previous_turn_settings: Option<PreviousTurnSettings>,
    pub(super) reference_context_item: Option<TurnContextItem>,
    pub(super) window_number: u64,
    pub(super) first_window_id: Option<Uuid>,
    pub(super) previous_window_id: Option<Uuid>,
    pub(super) window_id: Option<Uuid>,
    pub(super) chat_tree: Option<ReplayedChatTree>,
}

#[derive(Debug, Clone, Copy)]
struct ReconstructedWindow {
    number: u64,
    first_id: Option<Uuid>,
    previous_id: Option<Uuid>,
    id: Option<Uuid>,
}

#[derive(Debug, Default)]
enum TurnReferenceContextItem {
    /// No `TurnContextItem` has been seen for this replay span yet.
    ///
    /// This differs from `Cleared`: `NeverSet` means there is no evidence this turn ever
    /// established a baseline, while `Cleared` means a baseline existed and a later compaction
    /// invalidated it. Only the latter must emit an explicit clearing segment for resume/fork
    /// hydration.
    #[default]
    NeverSet,
    /// A previously established baseline was invalidated by later compaction.
    Cleared,
    /// The latest baseline established by this replay span.
    Latest(Box<TurnContextItem>),
}

#[derive(Debug, Default)]
struct ActiveReplaySegment<'a> {
    turn_id: Option<String>,
    counts_as_user_turn: bool,
    previous_turn_settings: Option<PreviousTurnSettings>,
    reference_context_item: TurnReferenceContextItem,
    base_replacement_history: Option<&'a [ResponseItem]>,
    window: Option<ReconstructedWindow>,
}

fn turn_ids_are_compatible(active_turn_id: Option<&str>, item_turn_id: Option<&str>) -> bool {
    active_turn_id
        .is_none_or(|turn_id| item_turn_id.is_none_or(|item_turn_id| item_turn_id == turn_id))
}

fn finalize_active_segment<'a>(
    active_segment: ActiveReplaySegment<'a>,
    base_replacement_history: &mut Option<&'a [ResponseItem]>,
    previous_turn_settings: &mut Option<PreviousTurnSettings>,
    reference_context_item: &mut TurnReferenceContextItem,
    window: &mut Option<ReconstructedWindow>,
    pending_rollback_turns: &mut usize,
) {
    // Thread rollback drops the newest surviving real user-message boundaries. In replay, that
    // means skipping the next finalized segments that contain a non-contextual
    // `EventMsg::UserMessage`.
    if *pending_rollback_turns > 0 {
        if active_segment.counts_as_user_turn {
            *pending_rollback_turns -= 1;
        }
        return;
    }

    // A surviving replacement-history checkpoint is a complete history base. Once we
    // know the newest surviving one, older rollout items do not affect rebuilt history.
    if base_replacement_history.is_none()
        && let Some(segment_base_replacement_history) = active_segment.base_replacement_history
    {
        *base_replacement_history = Some(segment_base_replacement_history);
    }

    if window.is_none() {
        *window = active_segment.window;
    }

    // `previous_turn_settings` come from the newest surviving user turn that established them.
    if previous_turn_settings.is_none() && active_segment.counts_as_user_turn {
        *previous_turn_settings = active_segment.previous_turn_settings;
    }

    // `reference_context_item` comes from the newest surviving user turn baseline, or
    // from a surviving compaction that explicitly cleared that baseline.
    if matches!(reference_context_item, TurnReferenceContextItem::NeverSet)
        && (active_segment.counts_as_user_turn
            || matches!(
                active_segment.reference_context_item,
                TurnReferenceContextItem::Cleared
            ))
    {
        *reference_context_item = active_segment.reference_context_item;
    }
}

impl Session {
    pub(super) async fn reconstruct_history_from_rollout(
        &self,
        turn_context: &TurnContext,
        rollout_items: &[RolloutItem],
    ) -> RolloutReconstruction {
        // Replay metadata should already match the shape of the future lazy reverse loader, even
        // while history materialization still uses an eager bridge. Scan newest-to-oldest,
        // stopping once a surviving replacement-history checkpoint and the required resume metadata
        // are both known; then replay only the buffered surviving tail forward to preserve exact
        // history semantics.
        let mut base_replacement_history: Option<&[ResponseItem]> = None;
        let mut previous_turn_settings = None;
        let mut reference_context_item = TurnReferenceContextItem::NeverSet;
        let mut window = None;
        // Rollback is "drop the newest N user turns". While scanning in reverse, that becomes
        // "skip the next N user-turn segments we finalize".
        let mut pending_rollback_turns = 0usize;
        // Borrowed suffix of rollout items newer than the newest surviving replacement-history
        // checkpoint. If no such checkpoint exists, this remains the full rollout.
        let mut rollout_suffix = rollout_items;
        // Reverse replay accumulates rollout items into the newest in-progress turn segment until
        // we hit its matching `TurnStarted`, at which point the segment can be finalized.
        let mut active_segment: Option<ActiveReplaySegment<'_>> = None;

        for (index, item) in rollout_items.iter().enumerate().rev() {
            match item {
                RolloutItem::Compacted(compacted) => {
                    let active_segment =
                        active_segment.get_or_insert_with(ActiveReplaySegment::default);
                    if active_segment.window.is_none()
                        && let Some(window_number) = compacted.window_number
                    {
                        active_segment.window = Some(ReconstructedWindow {
                            number: window_number,
                            first_id: compacted.first_window_id.as_deref().and_then(parse_uuid_v7),
                            previous_id: compacted
                                .previous_window_id
                                .as_deref()
                                .and_then(parse_uuid_v7),
                            id: compacted.window_id.as_deref().and_then(parse_uuid_v7),
                        });
                    }
                    // Looking backward, compaction clears any older baseline unless a newer
                    // `TurnContextItem` in this same segment has already re-established it.
                    if matches!(
                        active_segment.reference_context_item,
                        TurnReferenceContextItem::NeverSet
                    ) {
                        active_segment.reference_context_item = TurnReferenceContextItem::Cleared;
                    }
                    if active_segment.base_replacement_history.is_none()
                        && let Some(replacement_history) = &compacted.replacement_history
                    {
                        active_segment.base_replacement_history = Some(replacement_history);
                        rollout_suffix = &rollout_items[index + 1..];
                    }
                }
                RolloutItem::EventMsg(EventMsg::ThreadRolledBack(rollback)) => {
                    pending_rollback_turns = pending_rollback_turns
                        .saturating_add(usize::try_from(rollback.num_turns).unwrap_or(usize::MAX));
                }
                RolloutItem::EventMsg(EventMsg::TurnComplete(event)) => {
                    let active_segment =
                        active_segment.get_or_insert_with(ActiveReplaySegment::default);
                    // Reverse replay often sees `TurnComplete` before any turn-scoped metadata.
                    // Capture the turn id early so later `TurnContext` / abort items can match it.
                    if active_segment.turn_id.is_none() {
                        active_segment.turn_id = Some(event.turn_id.clone());
                    }
                }
                RolloutItem::EventMsg(EventMsg::TurnAborted(event)) => {
                    if let Some(active_segment) = active_segment.as_mut() {
                        if active_segment.turn_id.is_none()
                            && let Some(turn_id) = &event.turn_id
                        {
                            active_segment.turn_id = Some(turn_id.clone());
                        }
                    } else if let Some(turn_id) = &event.turn_id {
                        active_segment = Some(ActiveReplaySegment {
                            turn_id: Some(turn_id.clone()),
                            ..Default::default()
                        });
                    }
                }
                RolloutItem::EventMsg(EventMsg::UserMessage(_)) => {
                    let active_segment =
                        active_segment.get_or_insert_with(ActiveReplaySegment::default);
                    active_segment.counts_as_user_turn = true;
                }
                RolloutItem::TurnContext(ctx) => {
                    let active_segment =
                        active_segment.get_or_insert_with(ActiveReplaySegment::default);
                    // `TurnContextItem` can attach metadata to an existing segment, but only a
                    // real `UserMessage` event should make the segment count as a user turn.
                    if active_segment.turn_id.is_none() {
                        active_segment.turn_id = ctx.turn_id.clone();
                    }
                    if turn_ids_are_compatible(
                        active_segment.turn_id.as_deref(),
                        ctx.turn_id.as_deref(),
                    ) {
                        active_segment.previous_turn_settings = Some(PreviousTurnSettings {
                            model: ctx.model.clone(),
                            comp_hash: ctx.comp_hash.clone(),
                            realtime_active: ctx.realtime_active,
                        });
                        if matches!(
                            active_segment.reference_context_item,
                            TurnReferenceContextItem::NeverSet
                        ) {
                            active_segment.reference_context_item =
                                TurnReferenceContextItem::Latest(Box::new(ctx.clone()));
                        }
                    }
                }
                RolloutItem::EventMsg(EventMsg::TurnStarted(event)) => {
                    // `TurnStarted` is the oldest boundary of the active reverse segment.
                    if active_segment.as_ref().is_some_and(|active_segment| {
                        turn_ids_are_compatible(
                            active_segment.turn_id.as_deref(),
                            Some(event.turn_id.as_str()),
                        )
                    }) && let Some(active_segment) = active_segment.take()
                    {
                        finalize_active_segment(
                            active_segment,
                            &mut base_replacement_history,
                            &mut previous_turn_settings,
                            &mut reference_context_item,
                            &mut window,
                            &mut pending_rollback_turns,
                        );
                    }
                }
                RolloutItem::ResponseItem(response_item) => {
                    let active_segment =
                        active_segment.get_or_insert_with(ActiveReplaySegment::default);
                    active_segment.counts_as_user_turn |= is_user_turn_boundary(response_item);
                }
                RolloutItem::InterAgentCommunication(_) => {
                    let active_segment =
                        active_segment.get_or_insert_with(ActiveReplaySegment::default);
                    active_segment.counts_as_user_turn = true;
                }
                RolloutItem::EventMsg(_) | RolloutItem::SessionMeta(_) => {}
            }

            if base_replacement_history.is_some()
                && previous_turn_settings.is_some()
                && !matches!(reference_context_item, TurnReferenceContextItem::NeverSet)
            {
                // At this point we have both eager resume metadata values and the replacement-
                // history base for the surviving tail, so older rollout items cannot affect this
                // result.
                break;
            }
        }

        if let Some(active_segment) = active_segment.take() {
            finalize_active_segment(
                active_segment,
                &mut base_replacement_history,
                &mut previous_turn_settings,
                &mut reference_context_item,
                &mut window,
                &mut pending_rollback_turns,
            );
        }

        let fallback_window_number = u64::try_from(
            rollout_items
                .iter()
                .filter(|item| matches!(item, RolloutItem::Compacted(_)))
                .count(),
        )
        .unwrap_or(u64::MAX);

        let mut history = ContextManager::new();
        let mut saw_legacy_compaction_without_replacement_history = false;
        if let Some(base_replacement_history) = base_replacement_history {
            history.replace(base_replacement_history.to_vec());
        }
        // Materialize exact history semantics from the replay-derived suffix. The eventual lazy
        // design should keep this same replay shape, but drive it from a resumable reverse source
        // instead of an eagerly loaded `&[RolloutItem]`.
        for item in rollout_suffix {
            match item {
                RolloutItem::ResponseItem(response_item) => {
                    history.record_items(
                        std::iter::once(response_item),
                        turn_context.model_info.truncation_policy.into(),
                    );
                }
                RolloutItem::InterAgentCommunication(communication) => {
                    let response_item = communication.to_model_input_item();
                    history.record_items(
                        std::iter::once(&response_item),
                        turn_context.model_info.truncation_policy.into(),
                    );
                }
                RolloutItem::Compacted(compacted) => {
                    if let Some(replacement_history) = &compacted.replacement_history {
                        // This should actually never happen, because the reverse loop above (to build rollout_suffix)
                        // should stop before any compaction that has Some replacement_history
                        history.replace(replacement_history.clone());
                    } else {
                        saw_legacy_compaction_without_replacement_history = true;
                        // Legacy rollouts without `replacement_history` should rebuild the
                        // historical TurnContext at the correct insertion point from persisted
                        // `TurnContextItem`s. These are rare enough that we currently just clear
                        // `reference_context_item`, reinject canonical context at the end of the
                        // resumed conversation, and accept the temporary out-of-distribution
                        // prompt shape.
                        // TODO(ccunningham): if we drop support for None replacement_history compaction items,
                        // we can get rid of this second loop entirely and just build `history` directly in the first loop.
                        let user_messages = compact::collect_user_messages(history.raw_items());
                        let rebuilt = compact::build_compacted_history(
                            Vec::new(),
                            &user_messages,
                            &compacted.message,
                        );
                        history.replace(rebuilt);
                    }
                }
                RolloutItem::EventMsg(EventMsg::ThreadRolledBack(rollback)) => {
                    history.drop_last_n_user_turns(rollback.num_turns);
                }
                RolloutItem::EventMsg(_)
                | RolloutItem::TurnContext(_)
                | RolloutItem::SessionMeta(_) => {}
            }
        }

        let reference_context_item = match reference_context_item {
            TurnReferenceContextItem::NeverSet | TurnReferenceContextItem::Cleared => None,
            TurnReferenceContextItem::Latest(turn_reference_context_item) => {
                Some(*turn_reference_context_item)
            }
        };
        let reference_context_item = if saw_legacy_compaction_without_replacement_history {
            None
        } else {
            reference_context_item
        };

        let window = window.unwrap_or(ReconstructedWindow {
            number: fallback_window_number,
            first_id: None,
            previous_id: None,
            id: None,
        });
        RolloutReconstruction {
            history: history.raw_items().to_vec(),
            previous_turn_settings,
            reference_context_item,
            window_number: window.number,
            first_window_id: window.first_id,
            previous_window_id: window.previous_id,
            window_id: window.id,
            chat_tree: self.reconstruct_chat_tree_from_rollout(turn_context, rollout_items),
        }
    }

    fn reconstruct_chat_tree_from_rollout(
        &self,
        turn_context: &TurnContext,
        rollout_items: &[RolloutItem],
    ) -> Option<ReplayedChatTree> {
        let mut domain = DomainChatTreeState::default();
        let mut history_snapshots = HashMap::<String, ContextManager>::new();
        let mut active_node_id = None::<String>;
        let mut active_turn_id = None::<String>;
        let mut active_history = ContextManager::new();
        let mut legacy_history = ContextManager::new();
        let mut saw_chat_tree_event = false;
        let mut saw_current_node_change = false;

        for item in rollout_items {
            match item {
                RolloutItem::EventMsg(EventMsg::ChatTreeNodeStarted(payload)) => {
                    let parent_history = payload
                        .parent_node_id
                        .as_ref()
                        .and_then(|parent_node_id| history_snapshots.get(parent_node_id))
                        .cloned()
                        .unwrap_or_else(|| legacy_history.clone());
                    active_history = parent_history.clone();
                    let chat_tree_event = ChatTreeEvent::NodeStarted {
                        revision: payload.revision,
                        node_id: payload.node_id.clone(),
                        parent_node_id: payload.parent_node_id.clone(),
                        turn_id: payload.turn_id.clone(),
                        order: payload.order,
                    };
                    if let Err(err) = domain.apply_event(&chat_tree_event) {
                        warn!(
                            ?err,
                            "invalid durable chat tree event ignored while replaying rollout"
                        );
                        continue;
                    }
                    history_snapshots.insert(payload.node_id.clone(), parent_history);
                    saw_chat_tree_event = true;
                    active_node_id = payload
                        .turn_id
                        .as_deref()
                        .is_none_or(|turn_id| {
                            active_turn_id
                                .as_deref()
                                .is_none_or(|active_turn_id| active_turn_id == turn_id)
                        })
                        .then(|| payload.node_id.clone());
                }
                RolloutItem::ResponseItem(response_item) => {
                    if let Some(active_node_id) = active_node_id.as_deref() {
                        active_history.record_items(
                            std::iter::once(response_item),
                            turn_context.model_info.truncation_policy.into(),
                        );
                        history_snapshots
                            .insert(active_node_id.to_string(), active_history.clone());
                    } else if !saw_chat_tree_event {
                        legacy_history.record_items(
                            std::iter::once(response_item),
                            turn_context.model_info.truncation_policy.into(),
                        );
                    }
                }
                RolloutItem::TurnContext(turn_context_item) => {
                    if let Some(active_node_id) = active_node_id.as_deref() {
                        active_history.set_reference_context_item(Some(turn_context_item.clone()));
                        history_snapshots
                            .insert(active_node_id.to_string(), active_history.clone());
                    } else if saw_chat_tree_event
                        && let Some(current_node_id) = domain.current_node_id()
                        && let Some(history) = history_snapshots.get_mut(current_node_id)
                    {
                        history.set_reference_context_item(Some(turn_context_item.clone()));
                    } else if !saw_chat_tree_event {
                        legacy_history.set_reference_context_item(Some(turn_context_item.clone()));
                    }
                }
                RolloutItem::Compacted(compacted) => {
                    if let Some(active_node_id) = active_node_id.as_deref()
                        && let Some(replacement_history) = &compacted.replacement_history
                    {
                        active_history.replace(replacement_history.clone());
                        history_snapshots
                            .insert(active_node_id.to_string(), active_history.clone());
                    } else if saw_chat_tree_event
                        && let Some(replacement_history) = &compacted.replacement_history
                        && let Some(current_node_id) = domain.current_node_id()
                    {
                        let mut compacted_history = ContextManager::new();
                        compacted_history.replace(replacement_history.clone());
                        history_snapshots.insert(current_node_id.to_string(), compacted_history);
                    } else if !saw_chat_tree_event
                        && let Some(replacement_history) = &compacted.replacement_history
                    {
                        legacy_history.replace(replacement_history.clone());
                    }
                }
                RolloutItem::EventMsg(EventMsg::ChatTreeNodeFinalized(payload)) => {
                    let chat_tree_event = ChatTreeEvent::NodeFinalized {
                        revision: payload.revision,
                        node_id: payload.node_id.clone(),
                        status: payload.status,
                    };
                    if let Err(err) = domain.apply_event(&chat_tree_event) {
                        warn!(
                            ?err,
                            "invalid durable chat tree event ignored while replaying rollout"
                        );
                        continue;
                    }
                    saw_chat_tree_event = true;
                    if active_node_id.as_deref() == Some(payload.node_id.as_str()) {
                        history_snapshots.insert(payload.node_id.clone(), active_history.clone());
                        active_node_id = None;
                    }
                }
                RolloutItem::EventMsg(EventMsg::ChatTreeNodeSummaryUpdated(payload)) => {
                    let chat_tree_event = ChatTreeEvent::NodeSummaryUpdated {
                        revision: payload.revision,
                        node_id: payload.node_id.clone(),
                        summary: payload.summary.clone(),
                    };
                    if let Err(err) = domain.apply_event(&chat_tree_event) {
                        warn!(
                            ?err,
                            "invalid durable chat tree event ignored while replaying rollout"
                        );
                    }
                    saw_chat_tree_event = true;
                }
                RolloutItem::EventMsg(EventMsg::ChatTreeCurrentNodeChanged(payload)) => {
                    let chat_tree_event = ChatTreeEvent::CurrentNodeChanged {
                        revision: payload.revision,
                        node_id: payload.node_id.clone(),
                    };
                    if let Err(err) = domain.apply_event(&chat_tree_event) {
                        warn!(
                            ?err,
                            "invalid durable chat tree event ignored while replaying rollout"
                        );
                    }
                    saw_chat_tree_event = true;
                    saw_current_node_change = true;
                }
                RolloutItem::EventMsg(EventMsg::TurnStarted(payload)) => {
                    active_turn_id = Some(payload.turn_id.clone());
                    if let Err(err) = domain.apply_event(&ChatTreeEvent::LegacyTurnStarted {
                        turn_id: payload.turn_id.clone(),
                    }) {
                        warn!(
                            ?err,
                            "invalid durable chat tree event ignored while replaying rollout"
                        );
                    }
                }
                RolloutItem::EventMsg(EventMsg::TurnComplete(payload)) => {
                    if active_turn_id.as_deref() == Some(payload.turn_id.as_str()) {
                        active_turn_id = None;
                        active_node_id = None;
                    }
                }
                RolloutItem::EventMsg(EventMsg::TurnAborted(payload)) => {
                    if payload.turn_id.as_deref() == active_turn_id.as_deref() {
                        active_turn_id = None;
                        active_node_id = None;
                    }
                }
                RolloutItem::EventMsg(event_msg @ EventMsg::ThreadRolledBack(_)) => {
                    if let Some(event) = chat_tree_event_from_protocol_event(event_msg)
                        && let Err(err) = domain.apply_event(&event)
                    {
                        warn!(
                            ?err,
                            "invalid durable chat tree event ignored while replaying rollout"
                        );
                    }
                    let surviving_node_ids = domain
                        .projection()
                        .nodes
                        .into_iter()
                        .map(|node| node.node_id)
                        .collect::<HashSet<_>>();
                    history_snapshots.retain(|node_id, _| surviving_node_ids.contains(node_id));
                    active_turn_id = None;
                    active_node_id = None;
                    saw_chat_tree_event = !surviving_node_ids.is_empty();
                }
                RolloutItem::EventMsg(_)
                | RolloutItem::SessionMeta(_)
                | RolloutItem::InterAgentCommunication(_) => {}
            }
        }

        if domain.projection().nodes.is_empty() {
            return None;
        }

        let current_history = domain
            .current_node_id()
            .and_then(|node_id| history_snapshots.get(node_id))
            .cloned();
        let current_reference_context_item = current_history
            .as_ref()
            .and_then(ContextManager::reference_context_item);
        Some(ReplayedChatTree {
            domain,
            history_snapshots,
            current_history,
            current_reference_context_item,
            current_node_was_explicitly_selected: saw_current_node_change,
        })
    }
}

fn parse_uuid_v7(value: &str) -> Option<Uuid> {
    Uuid::parse_str(value)
        .ok()
        .filter(|uuid| uuid.get_version_num() == 7)
}
