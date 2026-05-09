use crate::protocol::ChatTreeNodeStatus;
use std::collections::HashMap;
use std::collections::HashSet;
use tracing::warn;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ChatTreeNodeStarted {
    pub revision: u64,
    pub node_id: String,
    pub parent_node_id: Option<String>,
    pub turn_id: Option<String>,
    pub order: u64,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ChatTreeNodeFinalized {
    pub revision: u64,
    pub node_id: String,
    pub status: ChatTreeNodeStatus,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ChatTreeCurrentNodeChanged {
    pub revision: u64,
    pub node_id: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ChatTreeNode {
    pub node_id: String,
    pub parent_node_id: Option<String>,
    pub turn_id: Option<String>,
    pub order: u64,
    pub status: ChatTreeNodeStatus,
    pub summary: Option<String>,
}

impl ChatTreeNode {
    pub fn default_summary(&self) -> String {
        format!(
            "Turn {} · {}",
            self.order.saturating_add(1),
            self.status.summary_label()
        )
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ChatTreeProjection {
    pub version: u32,
    pub revision: u64,
    pub current_node_id: Option<String>,
    pub visible_node_ids: Vec<String>,
    pub visible_turn_ids: Vec<String>,
    pub nodes: Vec<ChatTreeNode>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ChatTreeOverlayEntry {
    pub node_id: String,
    pub depth: usize,
    pub summary: String,
    pub is_current: bool,
}

#[derive(Debug, Clone, Default)]
pub struct ChatTreeState {
    revision: u64,
    current_node_id: Option<String>,
    legacy_visible_turn_ids: Vec<String>,
    nodes: HashMap<String, ChatTreeNode>,
    ordered_node_ids: Vec<String>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ChatTreeApplyError {
    DuplicateNode(String),
    MissingParent {
        node_id: String,
        parent_node_id: String,
    },
    UnknownNode(String),
    RevisionConflict {
        expected: u64,
        actual: u64,
    },
    NonIncreasingRevision {
        current: u64,
        incoming: u64,
    },
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ChatTreeEvent {
    NodeStarted {
        revision: u64,
        node_id: String,
        parent_node_id: Option<String>,
        turn_id: Option<String>,
        order: u64,
    },
    NodeFinalized {
        revision: u64,
        node_id: String,
        status: ChatTreeNodeStatus,
    },
    NodeSummaryUpdated {
        revision: u64,
        node_id: String,
        summary: Option<String>,
    },
    CurrentNodeChanged {
        revision: u64,
        node_id: String,
    },
    LegacyTurnStarted {
        turn_id: String,
    },
    RollbackVisibleTurns {
        num_turns: u32,
    },
}

impl ChatTreeState {
    pub fn start_node(&mut self, turn_id: String) -> Option<ChatTreeNodeStarted> {
        let node_id = turn_id.clone();
        if self.nodes.contains_key(&node_id) {
            warn!(
                node_id,
                "duplicate chat tree turn id ignored while starting node"
            );
            return None;
        }
        let parent_node_id = self.current_node_id.clone();
        let order = self.ordered_node_ids.len() as u64;
        self.revision = self.revision.saturating_add(1);
        self.insert_node(ChatTreeNode {
            node_id: node_id.clone(),
            parent_node_id: parent_node_id.clone(),
            turn_id: Some(turn_id.clone()),
            order,
            status: ChatTreeNodeStatus::Pending,
            summary: None,
        });
        self.current_node_id = Some(node_id.clone());
        Some(ChatTreeNodeStarted {
            revision: self.revision,
            node_id,
            parent_node_id,
            turn_id: Some(turn_id),
            order,
        })
    }

    pub fn finalize_node(
        &mut self,
        node_id: &str,
        status: ChatTreeNodeStatus,
    ) -> Option<ChatTreeNodeFinalized> {
        let node = self.nodes.get_mut(node_id)?;
        node.status = status;
        if node.summary.is_none() {
            node.summary = Some(node.default_summary());
        }
        self.revision = self.revision.saturating_add(1);
        Some(ChatTreeNodeFinalized {
            revision: self.revision,
            node_id: node_id.to_string(),
            status,
        })
    }

    pub fn set_current_node(
        &mut self,
        node_id: &str,
        expected_revision: Option<u64>,
    ) -> Result<ChatTreeCurrentNodeChanged, ChatTreeApplyError> {
        if let Some(expected) = expected_revision
            && expected != self.revision
        {
            return Err(ChatTreeApplyError::RevisionConflict {
                expected,
                actual: self.revision,
            });
        }
        if !self.nodes.contains_key(node_id) {
            return Err(ChatTreeApplyError::UnknownNode(node_id.to_string()));
        }
        self.current_node_id = Some(node_id.to_string());
        self.revision = self.revision.saturating_add(1);
        Ok(ChatTreeCurrentNodeChanged {
            revision: self.revision,
            node_id: node_id.to_string(),
        })
    }

    pub fn rollback_current_node_change(
        &mut self,
        event: &ChatTreeCurrentNodeChanged,
        previous_node_id: Option<String>,
        previous_revision: u64,
    ) {
        if self.current_node_id.as_deref() == Some(event.node_id.as_str())
            && self.revision == event.revision
        {
            self.current_node_id = previous_node_id;
            self.revision = previous_revision;
        }
    }

    pub fn apply_event(&mut self, event: &ChatTreeEvent) -> Result<(), ChatTreeApplyError> {
        match event {
            ChatTreeEvent::NodeStarted {
                revision,
                node_id,
                parent_node_id,
                turn_id,
                order,
            } => self.apply_node_started(
                *revision,
                node_id,
                parent_node_id.as_deref(),
                turn_id.as_deref(),
                *order,
            ),
            ChatTreeEvent::NodeFinalized {
                revision,
                node_id,
                status,
            } => self.apply_node_finalized(*revision, node_id, *status),
            ChatTreeEvent::NodeSummaryUpdated {
                revision,
                node_id,
                summary,
            } => self.apply_node_summary_updated(*revision, node_id, summary.clone()),
            ChatTreeEvent::CurrentNodeChanged { revision, node_id } => {
                self.apply_current_node_changed(*revision, node_id)
            }
            ChatTreeEvent::LegacyTurnStarted { turn_id } => {
                if self.nodes.is_empty() {
                    self.legacy_visible_turn_ids.push(turn_id.clone());
                }
                Ok(())
            }
            ChatTreeEvent::RollbackVisibleTurns { num_turns } => {
                self.rollback_visible_turns(*num_turns);
                Ok(())
            }
        }
    }

    pub fn rollback_started_node(
        &mut self,
        event: &ChatTreeNodeStarted,
        previous_current_node_id: Option<String>,
        previous_revision: u64,
    ) {
        if self.current_node_id.as_deref() == Some(event.node_id.as_str())
            && self.revision == event.revision
        {
            self.nodes.remove(&event.node_id);
            self.ordered_node_ids
                .retain(|node_id| node_id != &event.node_id);
            self.current_node_id = previous_current_node_id;
            self.revision = previous_revision;
        }
    }

    pub fn restore_node_for_rollback(&mut self, node: ChatTreeNode, previous_revision: u64) {
        if self.nodes.contains_key(&node.node_id) {
            self.nodes.insert(node.node_id.clone(), node);
            self.revision = previous_revision;
        }
    }

    pub fn projection(&self) -> ChatTreeProjection {
        let current_node_id = self.current_node_id.as_ref().and_then(|node_id| {
            if self.nodes.contains_key(node_id) {
                Some(node_id.clone())
            } else {
                warn!(
                    node_id,
                    "missing current chat tree node while building projection"
                );
                None
            }
        });
        let visible_node_ids = self.visible_node_ids();
        let visible_turn_ids = self
            .legacy_visible_turn_ids
            .iter()
            .cloned()
            .chain(visible_node_ids.iter().filter_map(|node_id| {
                self.nodes
                    .get(node_id)
                    .and_then(|node| node.turn_id.clone())
            }))
            .collect();
        let nodes = self
            .ordered_node_ids
            .iter()
            .filter_map(|node_id| self.nodes.get(node_id).cloned())
            .collect();

        ChatTreeProjection {
            version: 1,
            revision: self.revision,
            current_node_id,
            visible_node_ids,
            visible_turn_ids,
            nodes,
        }
    }

    pub fn overlay_entries(&self) -> Vec<ChatTreeOverlayEntry> {
        overlay_entries_from_projection(&self.projection())
    }

    pub fn replace_from_parts(
        &mut self,
        nodes: HashMap<String, ChatTreeNode>,
        ordered_node_ids: Vec<String>,
        current_node_id: Option<String>,
        legacy_visible_turn_ids: Vec<String>,
        revision: u64,
    ) {
        self.nodes = nodes;
        self.ordered_node_ids = ordered_node_ids;
        self.current_node_id = current_node_id;
        self.legacy_visible_turn_ids = legacy_visible_turn_ids;
        self.revision = revision;
    }

    pub fn from_projection(projection: ChatTreeProjection) -> Self {
        let node_turn_ids = projection
            .nodes
            .iter()
            .filter_map(|node| node.turn_id.as_deref())
            .collect::<HashSet<_>>();
        let legacy_visible_turn_ids = projection
            .visible_turn_ids
            .into_iter()
            .filter(|turn_id| !node_turn_ids.contains(turn_id.as_str()))
            .collect();
        let mut state = Self::default();
        state.replace_from_parts(
            projection
                .nodes
                .iter()
                .map(|node| (node.node_id.clone(), node.clone()))
                .collect(),
            ordered_node_ids_from_nodes(&projection.nodes),
            projection.current_node_id,
            legacy_visible_turn_ids,
            projection.revision,
        );
        state
    }

    pub fn node(&self, node_id: &str) -> Option<&ChatTreeNode> {
        self.nodes.get(node_id)
    }

    pub fn revision(&self) -> u64 {
        self.revision
    }

    pub fn current_node_id(&self) -> Option<&str> {
        self.current_node_id.as_deref()
    }

    fn apply_node_started(
        &mut self,
        revision: u64,
        node_id: &str,
        parent_node_id: Option<&str>,
        turn_id: Option<&str>,
        order: u64,
    ) -> Result<(), ChatTreeApplyError> {
        self.reject_non_increasing_revision(revision)?;
        if self.nodes.contains_key(node_id) {
            return Err(ChatTreeApplyError::DuplicateNode(node_id.to_string()));
        }
        if let Some(parent_node_id) = parent_node_id
            && !self.nodes.contains_key(parent_node_id)
        {
            return Err(ChatTreeApplyError::MissingParent {
                node_id: node_id.to_string(),
                parent_node_id: parent_node_id.to_string(),
            });
        }
        if self.nodes.is_empty()
            && turn_id == self.legacy_visible_turn_ids.last().map(String::as_str)
        {
            self.legacy_visible_turn_ids.pop();
        }
        self.insert_node(ChatTreeNode {
            node_id: node_id.to_string(),
            parent_node_id: parent_node_id.map(str::to_string),
            turn_id: turn_id.map(str::to_string),
            order,
            status: ChatTreeNodeStatus::Pending,
            summary: None,
        });
        self.current_node_id = Some(node_id.to_string());
        self.revision = revision;
        Ok(())
    }

    fn apply_node_finalized(
        &mut self,
        revision: u64,
        node_id: &str,
        status: ChatTreeNodeStatus,
    ) -> Result<(), ChatTreeApplyError> {
        self.reject_non_increasing_revision(revision)?;
        let Some(node) = self.nodes.get_mut(node_id) else {
            return Err(ChatTreeApplyError::UnknownNode(node_id.to_string()));
        };
        node.status = status;
        if node.summary.is_none() {
            node.summary = Some(node.default_summary());
        }
        self.revision = revision;
        Ok(())
    }

    fn apply_node_summary_updated(
        &mut self,
        revision: u64,
        node_id: &str,
        summary: Option<String>,
    ) -> Result<(), ChatTreeApplyError> {
        self.reject_non_increasing_revision(revision)?;
        let Some(node) = self.nodes.get_mut(node_id) else {
            return Err(ChatTreeApplyError::UnknownNode(node_id.to_string()));
        };
        node.summary = summary;
        self.revision = revision;
        Ok(())
    }

    fn apply_current_node_changed(
        &mut self,
        revision: u64,
        node_id: &str,
    ) -> Result<(), ChatTreeApplyError> {
        self.reject_non_increasing_revision(revision)?;
        if !self.nodes.contains_key(node_id) {
            return Err(ChatTreeApplyError::UnknownNode(node_id.to_string()));
        }
        self.current_node_id = Some(node_id.to_string());
        self.revision = revision;
        Ok(())
    }

    fn reject_non_increasing_revision(&self, incoming: u64) -> Result<(), ChatTreeApplyError> {
        if incoming <= self.revision {
            return Err(ChatTreeApplyError::NonIncreasingRevision {
                current: self.revision,
                incoming,
            });
        }
        Ok(())
    }

    fn rollback_visible_turns(&mut self, num_turns: u32) -> Vec<String> {
        let count = usize::try_from(num_turns).unwrap_or(usize::MAX);
        if count == 0 || self.nodes.is_empty() {
            return Vec::new();
        }

        let visible_node_ids = self.visible_node_ids();
        let remove_root_ids = visible_node_ids
            .iter()
            .rev()
            .take(count)
            .cloned()
            .collect::<HashSet<_>>();
        if remove_root_ids.is_empty() {
            return Vec::new();
        }

        let mut removed_node_ids = HashSet::<String>::new();
        let mut changed = true;
        while changed {
            changed = false;
            for node in self.nodes.values() {
                if (remove_root_ids.contains(&node.node_id)
                    || node
                        .parent_node_id
                        .as_ref()
                        .is_some_and(|parent_node_id| removed_node_ids.contains(parent_node_id)))
                    && removed_node_ids.insert(node.node_id.clone())
                {
                    changed = true;
                }
            }
        }

        let fallback_current_node_id = visible_node_ids
            .iter()
            .rev()
            .find(|node_id| !removed_node_ids.contains(*node_id))
            .cloned();

        for node_id in &removed_node_ids {
            self.nodes.remove(node_id);
        }
        self.ordered_node_ids
            .retain(|node_id| !removed_node_ids.contains(node_id));
        self.current_node_id = fallback_current_node_id;
        self.revision = self.revision.saturating_add(1);

        let mut removed_node_ids = removed_node_ids.into_iter().collect::<Vec<_>>();
        removed_node_ids.sort();
        removed_node_ids
    }

    fn insert_node(&mut self, node: ChatTreeNode) {
        if !self.nodes.contains_key(&node.node_id) {
            self.ordered_node_ids.push(node.node_id.clone());
        }
        self.nodes.insert(node.node_id.clone(), node);
    }

    fn visible_node_ids(&self) -> Vec<String> {
        let mut visible_node_ids = Vec::new();
        let mut next_node_id = self.current_node_id.clone();
        let mut visited_node_ids = HashSet::new();
        while let Some(node_id) = next_node_id {
            if !visited_node_ids.insert(node_id.clone()) {
                warn!(node_id, "cycle detected while projecting chat tree");
                break;
            }
            let Some(node) = self.nodes.get(&node_id) else {
                warn!(node_id, "missing current chat tree node while projecting");
                break;
            };
            visible_node_ids.push(node_id);
            next_node_id = node.parent_node_id.clone();
        }
        visible_node_ids.reverse();
        visible_node_ids
    }
}

pub fn overlay_entries_from_projection(
    projection: &ChatTreeProjection,
) -> Vec<ChatTreeOverlayEntry> {
    let mut children_by_parent = HashMap::<Option<String>, Vec<&ChatTreeNode>>::new();
    for node in nodes_by_order(&projection.nodes) {
        children_by_parent
            .entry(node.parent_node_id.clone())
            .or_default()
            .push(node);
    }
    let mut entries = Vec::new();
    append_overlay_entries(
        None,
        0,
        &children_by_parent,
        projection.current_node_id.as_deref(),
        &mut entries,
    );
    entries
}

fn append_overlay_entries(
    parent_node_id: Option<&str>,
    depth: usize,
    children_by_parent: &HashMap<Option<String>, Vec<&ChatTreeNode>>,
    current_node_id: Option<&str>,
    entries: &mut Vec<ChatTreeOverlayEntry>,
) {
    let key = parent_node_id.map(str::to_string);
    let Some(children) = children_by_parent.get(&key) else {
        return;
    };
    for node in children {
        entries.push(ChatTreeOverlayEntry {
            node_id: node.node_id.clone(),
            depth,
            summary: node
                .summary
                .clone()
                .unwrap_or_else(|| node.default_summary()),
            is_current: current_node_id == Some(node.node_id.as_str()),
        });
        append_overlay_entries(
            Some(node.node_id.as_str()),
            depth + 1,
            children_by_parent,
            current_node_id,
            entries,
        );
    }
}

fn nodes_by_order(nodes: &[ChatTreeNode]) -> Vec<&ChatTreeNode> {
    let mut nodes = nodes.iter().collect::<Vec<_>>();
    nodes.sort_by_key(|node| node.order);
    nodes
}

fn ordered_node_ids_from_nodes(nodes: &[ChatTreeNode]) -> Vec<String> {
    nodes_by_order(nodes)
        .into_iter()
        .map(|node| node.node_id.clone())
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn projection_includes_legacy_visible_turn_ids() {
        let mut state = ChatTreeState::default();
        state
            .apply_event(&ChatTreeEvent::LegacyTurnStarted {
                turn_id: "legacy-1".to_string(),
            })
            .unwrap();
        state
            .apply_event(&ChatTreeEvent::LegacyTurnStarted {
                turn_id: "legacy-2".to_string(),
            })
            .unwrap();
        state
            .start_node("turn-a".to_string())
            .expect("node should start");

        assert_eq!(
            state.projection().visible_turn_ids,
            vec![
                "legacy-1".to_string(),
                "legacy-2".to_string(),
                "turn-a".to_string(),
            ]
        );
    }

    #[test]
    fn projection_stops_on_parent_cycle() {
        let mut state = ChatTreeState::default();
        state.replace_from_parts(
            [
                (
                    "turn-a".to_string(),
                    ChatTreeNode {
                        node_id: "turn-a".to_string(),
                        parent_node_id: Some("turn-b".to_string()),
                        turn_id: Some("turn-a".to_string()),
                        order: 0,
                        status: ChatTreeNodeStatus::Completed,
                        summary: None,
                    },
                ),
                (
                    "turn-b".to_string(),
                    ChatTreeNode {
                        node_id: "turn-b".to_string(),
                        parent_node_id: Some("turn-a".to_string()),
                        turn_id: Some("turn-b".to_string()),
                        order: 1,
                        status: ChatTreeNodeStatus::Completed,
                        summary: None,
                    },
                ),
            ]
            .into_iter()
            .collect(),
            vec!["turn-a".to_string(), "turn-b".to_string()],
            Some("turn-a".to_string()),
            Vec::new(),
            1,
        );

        assert_eq!(
            state.projection().visible_turn_ids,
            vec!["turn-b".to_string(), "turn-a".to_string()]
        );
    }

    #[test]
    fn overlay_entries_use_chronological_sibling_order_and_depth() {
        let mut state = ChatTreeState::default();
        state.start_node("turn-a".to_string()).unwrap();
        state.start_node("turn-b".to_string()).unwrap();
        state.set_current_node("turn-a", None).unwrap();
        state.start_node("turn-c".to_string()).unwrap();

        let entries = state.overlay_entries();

        assert_eq!(
            entries
                .into_iter()
                .map(|entry| (entry.node_id, entry.depth, entry.is_current))
                .collect::<Vec<_>>(),
            vec![
                ("turn-a".to_string(), 0, false),
                ("turn-b".to_string(), 1, false),
                ("turn-c".to_string(), 1, true),
            ]
        );
    }

    #[test]
    fn replay_rejects_missing_parent_and_unknown_current() {
        let mut state = ChatTreeState::default();

        assert_eq!(
            state.apply_event(&ChatTreeEvent::NodeStarted {
                revision: 1,
                node_id: "turn-b".to_string(),
                parent_node_id: Some("turn-a".to_string()),
                turn_id: Some("turn-b".to_string()),
                order: 1,
            }),
            Err(ChatTreeApplyError::MissingParent {
                node_id: "turn-b".to_string(),
                parent_node_id: "turn-a".to_string(),
            })
        );
        assert_eq!(
            state.apply_event(&ChatTreeEvent::CurrentNodeChanged {
                revision: 2,
                node_id: "turn-missing".to_string(),
            }),
            Err(ChatTreeApplyError::UnknownNode("turn-missing".to_string()))
        );
        assert_eq!(state.projection().current_node_id, None);
        assert_eq!(state.projection().nodes, Vec::new());
    }

    #[test]
    fn replay_rejects_non_increasing_revision_without_rewinding_projection() {
        let mut state = ChatTreeState::default();
        state.start_node("turn-a".to_string()).unwrap();
        state.start_node("turn-b".to_string()).unwrap();
        let before = state.projection();

        assert_eq!(
            state.apply_event(&ChatTreeEvent::CurrentNodeChanged {
                revision: 1,
                node_id: "turn-a".to_string(),
            }),
            Err(ChatTreeApplyError::NonIncreasingRevision {
                current: before.revision,
                incoming: 1,
            })
        );
        assert_eq!(state.projection(), before);
    }

    #[test]
    fn replay_rejects_duplicate_revision_without_mutating_projection() {
        let mut state = ChatTreeState::default();
        state.start_node("turn-a".to_string()).unwrap();
        let before = state.projection();

        assert_eq!(
            state.apply_event(&ChatTreeEvent::NodeSummaryUpdated {
                revision: before.revision,
                node_id: "turn-a".to_string(),
                summary: Some("duplicate revision summary".to_string()),
            }),
            Err(ChatTreeApplyError::NonIncreasingRevision {
                current: before.revision,
                incoming: before.revision,
            })
        );
        assert_eq!(state.projection(), before);
    }

    #[test]
    fn rollback_visible_turns_prunes_current_path_and_descendants() {
        let mut state = ChatTreeState::default();
        state.start_node("turn-a".to_string()).unwrap();
        state.start_node("turn-b".to_string()).unwrap();
        state.set_current_node("turn-a", None).unwrap();
        state.start_node("turn-c".to_string()).unwrap();
        state.start_node("turn-d".to_string()).unwrap();

        state
            .apply_event(&ChatTreeEvent::RollbackVisibleTurns { num_turns: 1 })
            .unwrap();

        let projection = state.projection();
        assert_eq!(projection.current_node_id, Some("turn-c".to_string()));
        assert_eq!(
            projection.visible_node_ids,
            vec!["turn-a".to_string(), "turn-c".to_string()]
        );
        assert_eq!(
            projection
                .nodes
                .into_iter()
                .map(|node| node.node_id)
                .collect::<Vec<_>>(),
            vec![
                "turn-a".to_string(),
                "turn-b".to_string(),
                "turn-c".to_string(),
            ]
        );
    }
}
