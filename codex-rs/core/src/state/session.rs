//! Session-wide mutable state.

use codex_protocol::models::PermissionProfile;
use codex_protocol::models::ResponseItem;
use std::collections::HashMap;
use std::collections::HashSet;
use tokio::task::JoinHandle;

use crate::codex::PreviousTurnSettings;
use crate::codex::SessionConfiguration;
use crate::compact;
use crate::context_manager::ContextManager;
use crate::error::Result as CodexResult;
use crate::protocol::ChatTreeTurnInfo;
use crate::protocol::EventMsg;
use crate::protocol::RateLimitSnapshot;
use crate::protocol::TokenUsage;
use crate::protocol::TokenUsageInfo;
use crate::protocol::TurnAbortReason;
use crate::tasks::RegularTask;
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
    /// Startup regular task pre-created during session initialization.
    pub(crate) startup_regular_task: Option<JoinHandle<CodexResult<RegularTask>>>,
    pub(crate) active_mcp_tool_selection: Option<Vec<String>>,
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
            startup_regular_task: None,
            active_mcp_tool_selection: None,
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

    pub(crate) fn set_current_chat_tree_node(&mut self, node_id: &str) -> Result<(), String> {
        let snapshot = self.chat_tree.set_current_and_get_snapshot(node_id)?;
        self.history = snapshot;
        Ok(())
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

    pub(crate) fn set_startup_regular_task(&mut self, task: JoinHandle<CodexResult<RegularTask>>) {
        self.startup_regular_task = Some(task);
    }

    pub(crate) fn take_startup_regular_task(
        &mut self,
    ) -> Option<JoinHandle<CodexResult<RegularTask>>> {
        self.startup_regular_task.take()
    }

    pub(crate) fn merge_mcp_tool_selection(&mut self, tool_names: Vec<String>) -> Vec<String> {
        if tool_names.is_empty() {
            return self.active_mcp_tool_selection.clone().unwrap_or_default();
        }

        let mut merged = self.active_mcp_tool_selection.take().unwrap_or_default();
        let mut seen: HashSet<String> = merged.iter().cloned().collect();

        for tool_name in tool_names {
            if seen.insert(tool_name.clone()) {
                merged.push(tool_name);
            }
        }

        self.active_mcp_tool_selection = Some(merged.clone());
        merged
    }

    pub(crate) fn set_mcp_tool_selection(&mut self, tool_names: Vec<String>) {
        if tool_names.is_empty() {
            self.active_mcp_tool_selection = None;
            return;
        }

        let mut selected = Vec::new();
        let mut seen = HashSet::new();
        for tool_name in tool_names {
            if seen.insert(tool_name.clone()) {
                selected.push(tool_name);
            }
        }

        self.active_mcp_tool_selection = if selected.is_empty() {
            None
        } else {
            Some(selected)
        };
    }

    pub(crate) fn get_mcp_tool_selection(&self) -> Option<Vec<String>> {
        self.active_mcp_tool_selection.clone()
    }

    pub(crate) fn clear_mcp_tool_selection(&mut self) {
        self.active_mcp_tool_selection = None;
    }

    pub(crate) fn record_granted_permissions(&mut self, permissions: PermissionProfile) {
        self.granted_permissions = crate::sandboxing::merge_permission_profiles(
            self.granted_permissions.as_ref(),
            Some(&permissions),
        );
    }

    pub(crate) fn granted_permissions(&self) -> Option<PermissionProfile> {
        self.granted_permissions.clone()
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
mod tests {
    use super::*;
    use crate::codex::make_session_configuration_for_tests;
    use crate::protocol::EventMsg;
    use crate::protocol::RateLimitWindow;
    use crate::protocol::TurnCompleteEvent;
    use crate::protocol::TurnStartedEvent;
    use codex_protocol::config_types::ModeKind;
    use codex_protocol::models::ContentItem;
    use codex_protocol::models::ResponseItem;
    use codex_protocol::protocol::RolloutItem;
    use pretty_assertions::assert_eq;

    fn user_message(text: &str) -> ResponseItem {
        ResponseItem::Message {
            id: None,
            role: "user".to_string(),
            content: vec![ContentItem::InputText {
                text: text.to_string(),
            }],
            end_turn: None,
            phase: None,
        }
    }

    fn assistant_message(text: &str) -> ResponseItem {
        ResponseItem::Message {
            id: None,
            role: "assistant".to_string(),
            content: vec![ContentItem::OutputText {
                text: text.to_string(),
            }],
            end_turn: None,
            phase: None,
        }
    }

    #[tokio::test]
    async fn chat_tree_switch_restores_branch_snapshot_without_diverged_turns() {
        let session_configuration = make_session_configuration_for_tests().await;
        let mut state = SessionState::new(session_configuration);

        let node_a = state
            .create_chat_tree_child_node("node-a".to_string())
            .expect("create node-a");
        assert_eq!(node_a.parent_node_id, None);

        let items_a = [
            user_message("turn-a-user"),
            assistant_message("turn-a-assistant"),
        ];
        state.record_items(items_a.iter(), TruncationPolicy::Bytes(usize::MAX));
        state.finalize_chat_tree_node("node-a", Some("summary-a".to_string()));

        let snapshot_a = state.clone_history().raw_items().to_vec();

        let node_b = state
            .create_chat_tree_child_node("node-b".to_string())
            .expect("create node-b");
        assert_eq!(node_b.parent_node_id, Some("node-a".to_string()));
        let items_b = [
            user_message("turn-b-user"),
            assistant_message("turn-b-assistant"),
        ];
        state.record_items(items_b.iter(), TruncationPolicy::Bytes(usize::MAX));
        state.finalize_chat_tree_node("node-b", Some("summary-b".to_string()));

        state
            .set_current_chat_tree_node("node-a")
            .expect("switch to node-a");
        assert_eq!(state.clone_history().raw_items(), snapshot_a.as_slice());

        let node_c = state
            .create_chat_tree_child_node("node-c".to_string())
            .expect("create node-c");
        assert_eq!(node_c.parent_node_id, Some("node-a".to_string()));
        let items_c = vec![
            user_message("turn-c-user"),
            assistant_message("turn-c-assistant"),
        ];
        state.record_items(items_c.iter(), TruncationPolicy::Bytes(usize::MAX));
        state.finalize_chat_tree_node("node-c", Some("summary-c".to_string()));

        let mut expected = snapshot_a;
        expected.extend(items_c);
        assert_eq!(state.clone_history().raw_items(), expected.as_slice());
    }

    #[tokio::test]
    async fn chat_tree_late_summary_keeps_original_snapshot() {
        let session_configuration = make_session_configuration_for_tests().await;
        let mut state = SessionState::new(session_configuration);

        state
            .create_chat_tree_child_node("node-a".to_string())
            .expect("create node-a");
        let items_a = [
            user_message("turn-a-user"),
            assistant_message("turn-a-assistant"),
        ];
        state.record_items(items_a.iter(), TruncationPolicy::Bytes(usize::MAX));
        state.finalize_chat_tree_node("node-a", None);
        let snapshot_a = state.clone_history().raw_items().to_vec();

        state
            .create_chat_tree_child_node("node-b".to_string())
            .expect("create node-b");
        let items_b = [
            user_message("turn-b-user"),
            assistant_message("turn-b-assistant"),
        ];
        state.record_items(items_b.iter(), TruncationPolicy::Bytes(usize::MAX));
        state.finalize_chat_tree_node("node-b", Some("summary-b".to_string()));

        let updated = state
            .finalize_chat_tree_node("node-a", Some("summary-a".to_string()))
            .expect("update node-a");
        assert_eq!(updated.node_id, "node-a");
        assert_eq!(updated.parent_node_id, None);
        assert_eq!(updated.summary, Some("summary-a".to_string()));

        state
            .set_current_chat_tree_node("node-a")
            .expect("switch to node-a");
        assert_eq!(state.clone_history().raw_items(), snapshot_a.as_slice());
    }

    #[tokio::test]
    async fn sync_current_chat_tree_snapshot_keeps_compacted_branch_after_switching_away_and_back()
    {
        let session_configuration = make_session_configuration_for_tests().await;
        let mut state = SessionState::new(session_configuration);

        state
            .create_chat_tree_child_node("node-a".to_string())
            .expect("create node-a");
        let items_a = [
            user_message("turn-a-user"),
            assistant_message("turn-a-assistant"),
        ];
        state.record_items(items_a.iter(), TruncationPolicy::Bytes(usize::MAX));
        state.finalize_chat_tree_node("node-a", Some("summary-a".to_string()));

        state
            .create_chat_tree_child_node("node-b".to_string())
            .expect("create node-b");
        let items_b = [
            user_message("turn-b-user"),
            assistant_message("turn-b-assistant"),
        ];
        state.record_items(items_b.iter(), TruncationPolicy::Bytes(usize::MAX));
        state.finalize_chat_tree_node("node-b", Some("summary-b".to_string()));

        state
            .set_current_chat_tree_node("node-a")
            .expect("switch to node-a");
        state
            .create_chat_tree_child_node("node-c".to_string())
            .expect("create node-c");
        let items_c = [
            user_message("turn-c-user"),
            assistant_message("turn-c-assistant"),
        ];
        state.record_items(items_c.iter(), TruncationPolicy::Bytes(usize::MAX));
        state.finalize_chat_tree_node("node-c", Some("summary-c".to_string()));

        let compacted_history = vec![
            user_message("compacted-user"),
            assistant_message("compacted-summary"),
        ];
        state.replace_history(compacted_history.clone());
        state.sync_current_chat_tree_snapshot();

        state
            .set_current_chat_tree_node("node-b")
            .expect("switch to node-b");
        state
            .set_current_chat_tree_node("node-c")
            .expect("switch to node-c");
        assert_eq!(
            state.clone_history().raw_items(),
            compacted_history.as_slice()
        );
    }

    #[tokio::test]
    async fn restore_from_rollout_uses_persisted_parent_node_id() {
        let session_configuration = make_session_configuration_for_tests().await;
        let mut state = SessionState::new(session_configuration);
        let policy = TruncationPolicy::Bytes(usize::MAX);

        let turn_started = |turn_id: &str| {
            RolloutItem::EventMsg(EventMsg::TurnStarted(TurnStartedEvent {
                turn_id: turn_id.to_string(),
                model_context_window: None,
                collaboration_mode_kind: ModeKind::Default,
            }))
        };
        let turn_complete = |turn_id: &str, parent_node_id: Option<&str>, summary: &str| {
            RolloutItem::EventMsg(EventMsg::TurnComplete(TurnCompleteEvent {
                turn_id: turn_id.to_string(),
                last_agent_message: None,
                chat_tree: Some(ChatTreeTurnInfo {
                    node_id: turn_id.to_string(),
                    parent_node_id: parent_node_id.map(std::string::ToString::to_string),
                    summary: Some(summary.to_string()),
                }),
            }))
        };

        let rollout_items = vec![
            turn_started("node-a"),
            RolloutItem::ResponseItem(user_message("turn-a-user")),
            RolloutItem::ResponseItem(assistant_message("turn-a-assistant")),
            turn_complete("node-a", None, "summary-a"),
            turn_started("node-b"),
            RolloutItem::ResponseItem(user_message("turn-b-user")),
            RolloutItem::ResponseItem(assistant_message("turn-b-assistant")),
            turn_complete("node-b", Some("node-a"), "summary-b"),
            turn_started("node-c"),
            RolloutItem::ResponseItem(user_message("turn-c-user")),
            RolloutItem::ResponseItem(assistant_message("turn-c-assistant")),
            turn_complete("node-c", Some("node-a"), "summary-c"),
        ];

        state.restore_from_rollout(&rollout_items, &[], policy);

        state
            .set_current_chat_tree_node("node-b")
            .expect("switch to node-b");
        assert_eq!(
            state.clone_history().raw_items(),
            &[
                user_message("turn-a-user"),
                assistant_message("turn-a-assistant"),
                user_message("turn-b-user"),
                assistant_message("turn-b-assistant"),
            ]
        );

        state
            .set_current_chat_tree_node("node-c")
            .expect("switch to node-c");
        assert_eq!(
            state.clone_history().raw_items(),
            &[
                user_message("turn-a-user"),
                assistant_message("turn-a-assistant"),
                user_message("turn-c-user"),
                assistant_message("turn-c-assistant"),
            ]
        );
    }

    #[tokio::test]
    async fn merge_mcp_tool_selection_deduplicates_and_preserves_order() {
        let session_configuration = make_session_configuration_for_tests().await;
        let mut state = SessionState::new(session_configuration);

        let merged = state.merge_mcp_tool_selection(vec![
            "mcp__rmcp__echo".to_string(),
            "mcp__rmcp__image".to_string(),
            "mcp__rmcp__echo".to_string(),
        ]);
        assert_eq!(
            merged,
            vec![
                "mcp__rmcp__echo".to_string(),
                "mcp__rmcp__image".to_string(),
            ]
        );

        let merged = state.merge_mcp_tool_selection(vec![
            "mcp__rmcp__image".to_string(),
            "mcp__rmcp__search".to_string(),
        ]);
        assert_eq!(
            merged,
            vec![
                "mcp__rmcp__echo".to_string(),
                "mcp__rmcp__image".to_string(),
                "mcp__rmcp__search".to_string(),
            ]
        );
    }

    #[tokio::test]
    async fn merge_mcp_tool_selection_empty_input_is_noop() {
        let session_configuration = make_session_configuration_for_tests().await;
        let mut state = SessionState::new(session_configuration);
        state.merge_mcp_tool_selection(vec![
            "mcp__rmcp__echo".to_string(),
            "mcp__rmcp__image".to_string(),
        ]);

        let merged = state.merge_mcp_tool_selection(Vec::new());
        assert_eq!(
            merged,
            vec![
                "mcp__rmcp__echo".to_string(),
                "mcp__rmcp__image".to_string(),
            ]
        );
        assert_eq!(
            state.get_mcp_tool_selection(),
            Some(vec![
                "mcp__rmcp__echo".to_string(),
                "mcp__rmcp__image".to_string(),
            ])
        );
    }

    #[tokio::test]
    async fn clear_mcp_tool_selection_removes_selection() {
        let session_configuration = make_session_configuration_for_tests().await;
        let mut state = SessionState::new(session_configuration);
        state.merge_mcp_tool_selection(vec!["mcp__rmcp__echo".to_string()]);

        state.clear_mcp_tool_selection();

        assert_eq!(state.get_mcp_tool_selection(), None);
    }

    #[tokio::test]
    async fn set_mcp_tool_selection_deduplicates_and_preserves_order() {
        let session_configuration = make_session_configuration_for_tests().await;
        let mut state = SessionState::new(session_configuration);
        state.merge_mcp_tool_selection(vec!["mcp__rmcp__old".to_string()]);

        state.set_mcp_tool_selection(vec![
            "mcp__rmcp__echo".to_string(),
            "mcp__rmcp__image".to_string(),
            "mcp__rmcp__echo".to_string(),
            "mcp__rmcp__search".to_string(),
        ]);

        assert_eq!(
            state.get_mcp_tool_selection(),
            Some(vec![
                "mcp__rmcp__echo".to_string(),
                "mcp__rmcp__image".to_string(),
                "mcp__rmcp__search".to_string(),
            ])
        );
    }

    #[tokio::test]
    async fn set_mcp_tool_selection_empty_input_clears_selection() {
        let session_configuration = make_session_configuration_for_tests().await;
        let mut state = SessionState::new(session_configuration);
        state.merge_mcp_tool_selection(vec!["mcp__rmcp__echo".to_string()]);

        state.set_mcp_tool_selection(Vec::new());

        assert_eq!(state.get_mcp_tool_selection(), None);
    }

    #[tokio::test]
    // Verifies connector merging deduplicates repeated IDs.
    async fn merge_connector_selection_deduplicates_entries() {
        let session_configuration = make_session_configuration_for_tests().await;
        let mut state = SessionState::new(session_configuration);
        let merged = state.merge_connector_selection([
            "calendar".to_string(),
            "calendar".to_string(),
            "drive".to_string(),
        ]);

        assert_eq!(
            merged,
            HashSet::from(["calendar".to_string(), "drive".to_string()])
        );
    }

    #[tokio::test]
    // Verifies clearing connector selection removes all saved IDs.
    async fn clear_connector_selection_removes_entries() {
        let session_configuration = make_session_configuration_for_tests().await;
        let mut state = SessionState::new(session_configuration);
        state.merge_connector_selection(["calendar".to_string()]);

        state.clear_connector_selection();

        assert_eq!(state.get_connector_selection(), HashSet::new());
    }

    #[tokio::test]
    async fn set_rate_limits_defaults_limit_id_to_codex_when_missing() {
        let session_configuration = make_session_configuration_for_tests().await;
        let mut state = SessionState::new(session_configuration);

        state.set_rate_limits(RateLimitSnapshot {
            limit_id: None,
            limit_name: None,
            primary: Some(RateLimitWindow {
                used_percent: 12.0,
                window_minutes: Some(60),
                resets_at: Some(100),
            }),
            secondary: None,
            credits: None,
            plan_type: None,
        });

        assert_eq!(
            state
                .latest_rate_limits
                .as_ref()
                .and_then(|v| v.limit_id.clone()),
            Some("codex".to_string())
        );
    }

    #[tokio::test]
    async fn set_rate_limits_defaults_to_codex_when_limit_id_missing_after_other_bucket() {
        let session_configuration = make_session_configuration_for_tests().await;
        let mut state = SessionState::new(session_configuration);

        state.set_rate_limits(RateLimitSnapshot {
            limit_id: Some("codex_other".to_string()),
            limit_name: Some("codex_other".to_string()),
            primary: Some(RateLimitWindow {
                used_percent: 20.0,
                window_minutes: Some(60),
                resets_at: Some(200),
            }),
            secondary: None,
            credits: None,
            plan_type: None,
        });
        state.set_rate_limits(RateLimitSnapshot {
            limit_id: None,
            limit_name: None,
            primary: Some(RateLimitWindow {
                used_percent: 30.0,
                window_minutes: Some(60),
                resets_at: Some(300),
            }),
            secondary: None,
            credits: None,
            plan_type: None,
        });

        assert_eq!(
            state
                .latest_rate_limits
                .as_ref()
                .and_then(|v| v.limit_id.clone()),
            Some("codex".to_string())
        );
    }

    #[tokio::test]
    async fn set_rate_limits_carries_credits_and_plan_type_from_codex_to_codex_other() {
        let session_configuration = make_session_configuration_for_tests().await;
        let mut state = SessionState::new(session_configuration);

        state.set_rate_limits(RateLimitSnapshot {
            limit_id: Some("codex".to_string()),
            limit_name: Some("codex".to_string()),
            primary: Some(RateLimitWindow {
                used_percent: 10.0,
                window_minutes: Some(60),
                resets_at: Some(100),
            }),
            secondary: None,
            credits: Some(crate::protocol::CreditsSnapshot {
                has_credits: true,
                unlimited: false,
                balance: Some("50".to_string()),
            }),
            plan_type: Some(codex_protocol::account::PlanType::Plus),
        });

        state.set_rate_limits(RateLimitSnapshot {
            limit_id: Some("codex_other".to_string()),
            limit_name: None,
            primary: Some(RateLimitWindow {
                used_percent: 30.0,
                window_minutes: Some(120),
                resets_at: Some(200),
            }),
            secondary: None,
            credits: None,
            plan_type: None,
        });

        assert_eq!(
            state.latest_rate_limits,
            Some(RateLimitSnapshot {
                limit_id: Some("codex_other".to_string()),
                limit_name: None,
                primary: Some(RateLimitWindow {
                    used_percent: 30.0,
                    window_minutes: Some(120),
                    resets_at: Some(200),
                }),
                secondary: None,
                credits: Some(crate::protocol::CreditsSnapshot {
                    has_credits: true,
                    unlimited: false,
                    balance: Some("50".to_string()),
                }),
                plan_type: Some(codex_protocol::account::PlanType::Plus),
            })
        );
    }
}
