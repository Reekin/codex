//! Session-wide mutable state.

use codex_protocol::models::ResponseItem;
use std::collections::HashMap;
use std::collections::HashSet;

use crate::codex::SessionConfiguration;
use crate::context_manager::ContextManager;
use crate::protocol::ChatTreeTurnInfo;
use crate::protocol::RateLimitSnapshot;
use crate::protocol::TokenUsage;
use crate::protocol::TokenUsageInfo;
use crate::tasks::RegularTask;
use crate::truncate::TruncationPolicy;
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
        node.summary = summary.clone();
        node.history_snapshot = Some(history_snapshot);
        Some(ChatTreeTurnInfo {
            node_id: node_id.to_string(),
            parent_node_id: node
                .parent_node_id
                .clone()
                .filter(|id| id != CHAT_TREE_ROOT_NODE_ID),
            summary,
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
    /// Whether the session's initial context has been seeded into history.
    ///
    /// TODO(owen): This is a temporary solution to avoid updating a thread's updated_at
    /// timestamp when resuming a session. Remove this once SQLite is in place.
    pub(crate) initial_context_seeded: bool,
    /// Previous model seen by the session, used for model-switch handling on task start.
    previous_model: Option<String>,
    /// Startup regular task pre-created during session initialization.
    pub(crate) startup_regular_task: Option<RegularTask>,
    pub(crate) active_mcp_tool_selection: Option<Vec<String>>,
    pub(crate) active_connector_selection: HashSet<String>,
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
            initial_context_seeded: false,
            previous_model: None,
            startup_regular_task: None,
            active_mcp_tool_selection: None,
            active_connector_selection: HashSet::new(),
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

    pub(crate) fn previous_model(&self) -> Option<String> {
        self.previous_model.clone()
    }
    pub(crate) fn set_previous_model(&mut self, previous_model: Option<String>) {
        self.previous_model = previous_model;
    }

    pub(crate) fn clone_history(&self) -> ContextManager {
        self.history.clone()
    }

    pub(crate) fn replace_history(&mut self, items: Vec<ResponseItem>) {
        self.history.replace(items);
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

    pub(crate) fn set_token_info(&mut self, info: Option<TokenUsageInfo>) {
        self.history.set_token_info(info);
    }

    pub(crate) fn set_previous_context_item(&mut self, item: Option<TurnContextItem>) {
        self.history.set_previous_context_item(item);
    }

    pub(crate) fn previous_context_item(&self) -> Option<TurnContextItem> {
        self.history.previous_context_item()
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

    pub(crate) fn set_startup_regular_task(&mut self, task: RegularTask) {
        self.startup_regular_task = Some(task);
    }

    pub(crate) fn take_startup_regular_task(&mut self) -> Option<RegularTask> {
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
    use crate::protocol::RateLimitWindow;
    use codex_protocol::models::ContentItem;
    use codex_protocol::models::ResponseItem;
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

        let items_a = vec![
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
        let items_b = vec![
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

        let mut expected = snapshot_a.clone();
        expected.extend(items_c);
        assert_eq!(state.clone_history().raw_items(), expected.as_slice());
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
