//! Session-wide mutable state.

use codex_protocol::models::PermissionProfile;
use codex_protocol::models::ResponseItem;
use std::collections::HashMap;
use std::collections::HashSet;

use crate::codex::PreviousTurnSettings;
use crate::codex::SessionConfiguration;
use crate::compact;
use crate::context_manager::ContextManager;
use crate::protocol::ChatTreeTurnInfo;
use crate::protocol::EventMsg;
use crate::protocol::RateLimitSnapshot;
use crate::protocol::TokenUsage;
use crate::protocol::TokenUsageInfo;
use crate::protocol::TurnAbortReason;
use crate::sandboxing::merge_permission_profiles;
use crate::session_startup_prewarm::SessionStartupPrewarmHandle;
use crate::truncate::TruncationPolicy;
use codex_protocol::protocol::RolloutItem;
use codex_protocol::protocol::TurnAbortedEvent;
use codex_protocol::protocol::TurnContextItem;

const CHAT_TREE_ROOT_NODE_ID: &str = "__root__";

#[derive(Debug, Clone)]
struct ChatTreeNodeState {
    parent_node_id: Option<String>,
    summary: Option<String>,
    history_snapshot: Option<ContextManager>,
}

#[derive(Debug, Clone)]
struct ChatTreeState {
    current_node_id: String,
    nodes: HashMap<String, ChatTreeNodeState>,
}

impl ChatTreeState {
    fn new(history: &ContextManager) -> Self {
        let mut nodes = HashMap::new();
        nodes.insert(
            CHAT_TREE_ROOT_NODE_ID.to_string(),
            ChatTreeNodeState {
                parent_node_id: None,
                summary: Some("root".to_string()),
                history_snapshot: Some(history.clone()),
            },
        );
        Self {
            current_node_id: CHAT_TREE_ROOT_NODE_ID.to_string(),
            nodes,
        }
    }

    fn create_child_and_set_current(&mut self, node_id: String) -> Option<ChatTreeTurnInfo> {
        if self.nodes.contains_key(&node_id) {
            return None;
        }

        let parent_node_id = Some(self.current_node_id.clone());
        self.nodes.insert(
            node_id.clone(),
            ChatTreeNodeState {
                parent_node_id: parent_node_id.clone(),
                summary: None,
                history_snapshot: None,
            },
        );
        self.current_node_id = node_id.clone();

        Some(ChatTreeTurnInfo {
            node_id,
            parent_node_id: parent_node_id.filter(|id| id != CHAT_TREE_ROOT_NODE_ID),
            summary: None,
        })
    }

    fn set_current_snapshot(&mut self, history_snapshot: ContextManager) {
        if let Some(node) = self.nodes.get_mut(&self.current_node_id) {
            node.history_snapshot = Some(history_snapshot);
        }
    }

    fn finalize_node(
        &mut self,
        node_id: &str,
        summary: Option<String>,
        history_snapshot: ContextManager,
    ) -> Option<ChatTreeTurnInfo> {
        let node = self.nodes.get_mut(node_id)?;
        if let Some(summary) = summary {
            node.summary = Some(summary);
        }
        if node.history_snapshot.is_none() {
            node.history_snapshot = Some(history_snapshot);
        }
        Some(ChatTreeTurnInfo {
            node_id: node_id.to_string(),
            parent_node_id: node
                .parent_node_id
                .clone()
                .filter(|id| id != CHAT_TREE_ROOT_NODE_ID),
            summary: node.summary.clone(),
        })
    }

    fn update_node_summary(&mut self, node_id: &str, summary: String) -> Option<ChatTreeTurnInfo> {
        let node = self.nodes.get_mut(node_id)?;
        node.summary = Some(summary);
        Some(ChatTreeTurnInfo {
            node_id: node_id.to_string(),
            parent_node_id: node
                .parent_node_id
                .clone()
                .filter(|id| id != CHAT_TREE_ROOT_NODE_ID),
            summary: node.summary.clone(),
        })
    }

    fn restore_node(
        &mut self,
        node_id: &str,
        parent_node_id: Option<String>,
        summary: Option<String>,
        history_snapshot: ContextManager,
    ) -> ChatTreeTurnInfo {
        let stored_parent_node_id =
            parent_node_id.unwrap_or_else(|| CHAT_TREE_ROOT_NODE_ID.to_string());
        self.nodes.insert(
            node_id.to_string(),
            ChatTreeNodeState {
                parent_node_id: Some(stored_parent_node_id.clone()),
                summary: summary.clone(),
                history_snapshot: Some(history_snapshot),
            },
        );
        self.current_node_id = node_id.to_string();
        ChatTreeTurnInfo {
            node_id: node_id.to_string(),
            parent_node_id: normalize_parent_node_id(stored_parent_node_id),
            summary,
        }
    }

    fn snapshot_for_node(&self, node_id: &str) -> Option<ContextManager> {
        self.nodes
            .get(node_id)
            .and_then(|node| node.history_snapshot.clone())
    }

    fn current_info(&self) -> Option<ChatTreeTurnInfo> {
        let node = self.nodes.get(&self.current_node_id)?;
        Some(ChatTreeTurnInfo {
            node_id: self.current_node_id.clone(),
            parent_node_id: node
                .parent_node_id
                .clone()
                .filter(|id| id != CHAT_TREE_ROOT_NODE_ID),
            summary: node.summary.clone(),
        })
    }

    fn set_current_and_get_snapshot(&mut self, node_id: &str) -> Result<ContextManager, String> {
        let node = self
            .nodes
            .get(node_id)
            .ok_or_else(|| format!("unknown chat tree node id: {node_id}"))?;
        let snapshot = node
            .history_snapshot
            .clone()
            .ok_or_else(|| format!("chat tree node has no snapshot yet: {node_id}"))?;
        self.current_node_id = node_id.to_string();
        Ok(snapshot)
    }
}

/// Persistent, session-scoped state previously stored directly on `Session`.
pub(crate) struct SessionState {
    pub(crate) session_configuration: SessionConfiguration,
    pub(crate) history: ContextManager,
    pub(crate) latest_rate_limits: Option<RateLimitSnapshot>,
    pub(crate) server_reasoning_included: bool,
    pub(crate) dependency_env: HashMap<String, String>,
    pub(crate) mcp_dependency_prompted: HashSet<String>,
    /// Settings used by the latest regular user turn, used for turn-to-turn
    /// model/realtime handling on subsequent regular turns (including full-context
    /// reinjection after resume or `/compact`).
    previous_turn_settings: Option<PreviousTurnSettings>,
    /// Startup prewarmed session prepared during session initialization.
    pub(crate) startup_prewarm: Option<SessionStartupPrewarmHandle>,
    pub(crate) active_connector_selection: HashSet<String>,
    pub(crate) pending_session_start_source: Option<codex_hooks::SessionStartSource>,
    granted_permissions: Option<PermissionProfile>,
    chat_tree: ChatTreeState,
}

impl SessionState {
    /// Create a new session state mirroring previous `State::default()` semantics.
    pub(crate) fn new(session_configuration: SessionConfiguration) -> Self {
        let history = ContextManager::new();
        Self {
            session_configuration,
            chat_tree: ChatTreeState::new(&history),
            history,
            latest_rate_limits: None,
            server_reasoning_included: false,
            dependency_env: HashMap::new(),
            mcp_dependency_prompted: HashSet::new(),
            previous_turn_settings: None,
            startup_prewarm: None,
            active_connector_selection: HashSet::new(),
            pending_session_start_source: None,
            granted_permissions: None,
        }
    }

    // History helpers
    pub(crate) fn record_items<I>(&mut self, items: I, policy: TruncationPolicy)
    where
        I: IntoIterator,
        I::Item: std::ops::Deref<Target = ResponseItem>,
    {
        self.history.record_items(items, policy);
    }

    pub(crate) fn previous_turn_settings(&self) -> Option<PreviousTurnSettings> {
        self.previous_turn_settings.clone()
    }
    pub(crate) fn set_previous_turn_settings(
        &mut self,
        previous_turn_settings: Option<PreviousTurnSettings>,
    ) {
        self.previous_turn_settings = previous_turn_settings;
    }

    pub(crate) fn clone_history(&self) -> ContextManager {
        self.history.clone()
    }

    pub(crate) fn replace_history(
        &mut self,
        items: Vec<ResponseItem>,
        reference_context_item: Option<TurnContextItem>,
    ) {
        self.history.replace(items);
        self.history
            .set_reference_context_item(reference_context_item);
    }

    #[cfg(test)]
    pub(crate) fn sync_current_chat_tree_snapshot(&mut self) {
        self.chat_tree.set_current_snapshot(self.history.clone());
    }

    pub(crate) fn create_chat_tree_child_node(
        &mut self,
        node_id: String,
    ) -> Option<ChatTreeTurnInfo> {
        self.chat_tree.set_current_snapshot(self.history.clone());
        self.chat_tree.create_child_and_set_current(node_id)
    }

    pub(crate) fn finalize_chat_tree_node(
        &mut self,
        node_id: &str,
        summary: Option<String>,
    ) -> Option<ChatTreeTurnInfo> {
        let snapshot = self.history.clone();
        self.chat_tree.finalize_node(node_id, summary, snapshot)
    }

    pub(crate) fn set_current_chat_tree_node(
        &mut self,
        node_id: &str,
    ) -> Result<ChatTreeTurnInfo, String> {
        let snapshot = self.chat_tree.set_current_and_get_snapshot(node_id)?;
        self.history = snapshot;
        self.chat_tree
            .current_info()
            .ok_or_else(|| format!("unknown chat tree node id: {node_id}"))
    }

    pub(crate) fn current_chat_tree_node_id(&self) -> Option<String> {
        (self.chat_tree.current_node_id != CHAT_TREE_ROOT_NODE_ID)
            .then(|| self.chat_tree.current_node_id.clone())
    }

    pub(crate) fn restore_from_rollout(
        &mut self,
        rollout_items: &[RolloutItem],
        initial_context: &[ResponseItem],
        policy: TruncationPolicy,
    ) {
        let reference_context_item = self.history.reference_context_item();
        let mut history = ContextManager::new();
        let mut chat_tree = ChatTreeState::new(&history);
        let mut active_turn: Option<PendingChatTreeTurnRestore> = None;

        for item in rollout_items {
            match item {
                RolloutItem::ResponseItem(_)
                | RolloutItem::Compacted(_)
                | RolloutItem::EventMsg(EventMsg::ThreadRolledBack(_)) => {
                    if let Some(active_turn) = active_turn.as_mut() {
                        active_turn.history_mutations.push(item.clone());
                    } else {
                        apply_rollout_item_to_history(&mut history, item, initial_context, policy);
                        chat_tree.set_current_snapshot(history.clone());
                    }
                }
                RolloutItem::EventMsg(EventMsg::TurnStarted(event)) => {
                    active_turn = Some(PendingChatTreeTurnRestore {
                        turn_id: event.turn_id.clone(),
                        parent_node_id: chat_tree.current_node_id.clone(),
                        history_mutations: Vec::new(),
                    });
                }
                RolloutItem::EventMsg(EventMsg::TurnComplete(event)) => {
                    let pending_turn = active_turn.take();
                    if let Some(chat_tree_turn) = &event.chat_tree {
                        let mut snapshot = build_chat_tree_snapshot(
                            &chat_tree,
                            pending_turn.as_ref(),
                            chat_tree_turn.parent_node_id.as_deref(),
                        );
                        if let Some(pending_turn) = pending_turn.as_ref()
                            && pending_turn.turn_id == chat_tree_turn.node_id
                        {
                            for mutation in &pending_turn.history_mutations {
                                apply_rollout_item_to_history(
                                    &mut snapshot,
                                    mutation,
                                    initial_context,
                                    policy,
                                );
                            }
                        }
                        history = snapshot.clone();
                        chat_tree.restore_node(
                            &chat_tree_turn.node_id,
                            chat_tree_turn.parent_node_id.clone(),
                            chat_tree_turn.summary.clone(),
                            snapshot,
                        );
                    } else if let Some(pending_turn) = pending_turn {
                        for mutation in &pending_turn.history_mutations {
                            apply_rollout_item_to_history(
                                &mut history,
                                mutation,
                                initial_context,
                                policy,
                            );
                        }
                        chat_tree.set_current_snapshot(history.clone());
                    }
                }
                RolloutItem::EventMsg(EventMsg::ChatTreeNodeUpdated(event)) => {
                    let chat_tree_turn = &event.chat_tree;
                    if let Some(summary) = &chat_tree_turn.summary {
                        let _ =
                            chat_tree.update_node_summary(&chat_tree_turn.node_id, summary.clone());
                    }
                }
                RolloutItem::EventMsg(EventMsg::TurnAborted(TurnAbortedEvent {
                    turn_id: Some(turn_id),
                    reason,
                    chat_tree: event_chat_tree,
                })) => {
                    let pending_turn = active_turn.take();
                    let mut snapshot = build_chat_tree_snapshot(
                        &chat_tree,
                        pending_turn.as_ref(),
                        event_chat_tree
                            .as_ref()
                            .and_then(|turn| turn.parent_node_id.as_deref())
                            .or_else(|| {
                                pending_turn
                                    .as_ref()
                                    .map(|turn| turn.parent_node_id.as_str())
                            }),
                    );
                    if let Some(pending_turn) = pending_turn.as_ref()
                        && pending_turn.turn_id == *turn_id
                    {
                        for mutation in &pending_turn.history_mutations {
                            apply_rollout_item_to_history(
                                &mut snapshot,
                                mutation,
                                initial_context,
                                policy,
                            );
                        }
                    }
                    history = snapshot.clone();
                    let parent_node_id = event_chat_tree
                        .as_ref()
                        .and_then(|turn| turn.parent_node_id.clone())
                        .and_then(normalize_parent_node_id)
                        .or_else(|| {
                            pending_turn
                                .and_then(|turn| normalize_parent_node_id(turn.parent_node_id))
                        });
                    chat_tree.restore_node(
                        event_chat_tree
                            .as_ref()
                            .map(|turn| turn.node_id.as_str())
                            .unwrap_or(turn_id),
                        parent_node_id,
                        event_chat_tree
                            .as_ref()
                            .and_then(|turn| turn.summary.clone())
                            .or_else(|| Some(abort_reason_summary(reason))),
                        snapshot,
                    );
                }
                RolloutItem::TurnContext(_)
                | RolloutItem::SessionMeta(_)
                | RolloutItem::EventMsg(_) => {}
            }
        }

        if let Some(pending_turn) = active_turn {
            for mutation in &pending_turn.history_mutations {
                apply_rollout_item_to_history(&mut history, mutation, initial_context, policy);
            }
            chat_tree.set_current_snapshot(history.clone());
        }

        self.history = history;
        self.history
            .set_reference_context_item(reference_context_item);
        self.chat_tree = chat_tree;
    }

    pub(crate) fn append_to_all_chat_tree_snapshots(
        &mut self,
        items: &[ResponseItem],
        policy: TruncationPolicy,
    ) {
        if items.is_empty() {
            return;
        }

        for node in self.chat_tree.nodes.values_mut() {
            if let Some(snapshot) = node.history_snapshot.as_mut() {
                snapshot.record_items(items.iter(), policy);
            }
        }
    }

    pub(crate) fn set_token_info(&mut self, info: Option<TokenUsageInfo>) {
        self.history.set_token_info(info);
    }

    pub(crate) fn set_reference_context_item(&mut self, item: Option<TurnContextItem>) {
        self.history.set_reference_context_item(item);
    }

    pub(crate) fn reference_context_item(&self) -> Option<TurnContextItem> {
        self.history.reference_context_item()
    }

    // Token/rate limit helpers
    pub(crate) fn update_token_info_from_usage(
        &mut self,
        usage: &TokenUsage,
        model_context_window: Option<i64>,
    ) {
        self.history.update_token_info(usage, model_context_window);
    }

    pub(crate) fn token_info(&self) -> Option<TokenUsageInfo> {
        self.history.token_info()
    }

    pub(crate) fn set_rate_limits(&mut self, snapshot: RateLimitSnapshot) {
        self.latest_rate_limits = Some(merge_rate_limit_fields(
            self.latest_rate_limits.as_ref(),
            snapshot,
        ));
    }

    pub(crate) fn token_info_and_rate_limits(
        &self,
    ) -> (Option<TokenUsageInfo>, Option<RateLimitSnapshot>) {
        (self.token_info(), self.latest_rate_limits.clone())
    }

    pub(crate) fn set_token_usage_full(&mut self, context_window: i64) {
        self.history.set_token_usage_full(context_window);
    }

    pub(crate) fn get_total_token_usage(&self, server_reasoning_included: bool) -> i64 {
        self.history
            .get_total_token_usage(server_reasoning_included)
    }

    pub(crate) fn set_server_reasoning_included(&mut self, included: bool) {
        self.server_reasoning_included = included;
    }

    pub(crate) fn server_reasoning_included(&self) -> bool {
        self.server_reasoning_included
    }

    pub(crate) fn record_mcp_dependency_prompted<I>(&mut self, names: I)
    where
        I: IntoIterator<Item = String>,
    {
        self.mcp_dependency_prompted.extend(names);
    }

    pub(crate) fn mcp_dependency_prompted(&self) -> HashSet<String> {
        self.mcp_dependency_prompted.clone()
    }

    pub(crate) fn set_dependency_env(&mut self, values: HashMap<String, String>) {
        for (key, value) in values {
            self.dependency_env.insert(key, value);
        }
    }

    pub(crate) fn dependency_env(&self) -> HashMap<String, String> {
        self.dependency_env.clone()
    }

    pub(crate) fn set_session_startup_prewarm(
        &mut self,
        startup_prewarm: SessionStartupPrewarmHandle,
    ) {
        self.startup_prewarm = Some(startup_prewarm);
    }

    pub(crate) fn take_session_startup_prewarm(&mut self) -> Option<SessionStartupPrewarmHandle> {
        self.startup_prewarm.take()
    }

    // Adds connector IDs to the active set and returns the merged selection.
    pub(crate) fn merge_connector_selection<I>(&mut self, connector_ids: I) -> HashSet<String>
    where
        I: IntoIterator<Item = String>,
    {
        self.active_connector_selection.extend(connector_ids);
        self.active_connector_selection.clone()
    }

    // Returns the current connector selection tracked on session state.
    pub(crate) fn get_connector_selection(&self) -> HashSet<String> {
        self.active_connector_selection.clone()
    }

    // Removes all currently tracked connector selections.
    pub(crate) fn clear_connector_selection(&mut self) {
        self.active_connector_selection.clear();
    }

    pub(crate) fn set_pending_session_start_source(
        &mut self,
        value: Option<codex_hooks::SessionStartSource>,
    ) {
        self.pending_session_start_source = value;
    }

    pub(crate) fn take_pending_session_start_source(
        &mut self,
    ) -> Option<codex_hooks::SessionStartSource> {
        self.pending_session_start_source.take()
    }

    pub(crate) fn record_granted_permissions(&mut self, permissions: PermissionProfile) {
        self.granted_permissions =
            merge_permission_profiles(self.granted_permissions.as_ref(), Some(&permissions));
    }

    pub(crate) fn granted_permissions(&self) -> Option<PermissionProfile> {
        self.granted_permissions.clone()
    }
}

fn abort_reason_summary(reason: &TurnAbortReason) -> String {
    match reason {
        TurnAbortReason::Interrupted => "turn interrupted".to_string(),
        TurnAbortReason::Replaced => "turn replaced".to_string(),
        TurnAbortReason::ReviewEnded => "turn review ended".to_string(),
    }
}

fn normalize_parent_node_id(parent_node_id: String) -> Option<String> {
    if parent_node_id == CHAT_TREE_ROOT_NODE_ID {
        None
    } else {
        Some(parent_node_id)
    }
}

fn build_chat_tree_snapshot(
    chat_tree: &ChatTreeState,
    pending_turn: Option<&PendingChatTreeTurnRestore>,
    persisted_parent_node_id: Option<&str>,
) -> ContextManager {
    let parent_node_id = persisted_parent_node_id
        .or_else(|| pending_turn.map(|turn| turn.parent_node_id.as_str()))
        .unwrap_or(CHAT_TREE_ROOT_NODE_ID);
    chat_tree
        .snapshot_for_node(parent_node_id)
        .unwrap_or_default()
}

fn apply_rollout_item_to_history(
    history: &mut ContextManager,
    item: &RolloutItem,
    initial_context: &[ResponseItem],
    policy: TruncationPolicy,
) {
    match item {
        RolloutItem::ResponseItem(response_item) => {
            history.record_items(std::iter::once(response_item), policy);
        }
        RolloutItem::Compacted(compacted) => {
            if let Some(replacement) = &compacted.replacement_history {
                history.replace(replacement.clone());
            } else {
                let user_messages = compact::collect_user_messages(history.raw_items());
                let rebuilt = compact::build_compacted_history(
                    initial_context.to_vec(),
                    &user_messages,
                    &compacted.message,
                );
                history.replace(rebuilt);
            }
        }
        RolloutItem::EventMsg(EventMsg::ThreadRolledBack(rollback)) => {
            history.drop_last_n_user_turns(rollback.num_turns);
        }
        RolloutItem::TurnContext(_) | RolloutItem::SessionMeta(_) | RolloutItem::EventMsg(_) => {}
    }
}

#[derive(Debug, Clone)]
struct PendingChatTreeTurnRestore {
    turn_id: String,
    parent_node_id: String,
    history_mutations: Vec<RolloutItem>,
}

// Sometimes new snapshots don't include credits or plan information.
// Preserve those from the previous snapshot when missing. For `limit_id`, treat
// missing values as the default `"codex"` bucket.
fn merge_rate_limit_fields(
    previous: Option<&RateLimitSnapshot>,
    mut snapshot: RateLimitSnapshot,
) -> RateLimitSnapshot {
    if snapshot.limit_id.is_none() {
        snapshot.limit_id = Some("codex".to_string());
    }
    if snapshot.credits.is_none() {
        snapshot.credits = previous.and_then(|prior| prior.credits.clone());
    }
    if snapshot.plan_type.is_none() {
        snapshot.plan_type = previous.and_then(|prior| prior.plan_type);
    }
    snapshot
}

#[cfg(test)]
#[path = "session_tests.rs"]
mod tests;
