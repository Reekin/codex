use crate::context_manager::ContextManager;
use codex_protocol::chat_tree::ChatTreeApplyError;
use codex_protocol::chat_tree::ChatTreeCurrentNodeChanged;
use codex_protocol::chat_tree::ChatTreeNode as DomainChatTreeNode;
use codex_protocol::chat_tree::ChatTreeNodeFinalized;
use codex_protocol::chat_tree::ChatTreeNodeStarted;
use codex_protocol::chat_tree::ChatTreeProjection as DomainChatTreeProjection;
use codex_protocol::chat_tree::ChatTreeState as DomainChatTreeState;
use codex_protocol::protocol::ChatTreeNodeStatus;
use codex_protocol::protocol::TurnContextItem;
use std::collections::HashMap;

#[derive(Debug, Clone, Default)]
pub(crate) struct ChatTreeState {
    domain: DomainChatTreeState,
    history_snapshots: HashMap<String, ContextManager>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ChatTreeError {
    UnknownNode(String),
    MissingSnapshot(String),
    RevisionConflict { expected: u64, actual: u64 },
    Persistence(String),
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ChatTreeNodeSnapshot {
    pub node_id: String,
    pub parent_node_id: Option<String>,
    pub turn_id: Option<String>,
    pub order: u64,
    pub status: ChatTreeNodeStatus,
    pub summary: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ChatTreeProjectionSnapshot {
    pub version: u32,
    pub revision: u64,
    pub current_node_id: Option<String>,
    pub visible_node_ids: Vec<String>,
    pub visible_turn_ids: Vec<String>,
    pub nodes: Vec<ChatTreeNodeSnapshot>,
}

pub(crate) struct ChatTreeSelection {
    pub(crate) event: ChatTreeCurrentNodeChanged,
    pub(crate) history: ContextManager,
    pub(crate) previous_node_id: Option<String>,
    pub(crate) previous_revision: u64,
}

pub(crate) struct ChatTreeNodeStart {
    pub(crate) event: ChatTreeNodeStarted,
    previous_node_id: Option<String>,
    previous_revision: u64,
}

pub(crate) struct ChatTreeNodeFinalization {
    pub(crate) event: ChatTreeNodeFinalized,
    previous_node: DomainChatTreeNode,
    previous_revision: u64,
    previous_history_snapshot: Option<ContextManager>,
}

impl ChatTreeState {
    pub(crate) fn start_node(
        &mut self,
        turn_id: String,
        parent_history: ContextManager,
    ) -> Option<ChatTreeNodeStart> {
        let previous_node_id = self.domain.current_node_id().map(str::to_string);
        let previous_revision = self.domain.revision();
        let event = self.domain.start_node(turn_id)?;
        self.history_snapshots
            .insert(event.node_id.clone(), parent_history);
        Some(ChatTreeNodeStart {
            event,
            previous_node_id,
            previous_revision,
        })
    }

    pub(crate) fn finalize_node(
        &mut self,
        node_id: &str,
        status: ChatTreeNodeStatus,
        history_snapshot: ContextManager,
    ) -> Option<ChatTreeNodeFinalization> {
        let previous_node = self.domain.node(node_id)?.clone();
        let previous_revision = self.domain.revision();
        let previous_history_snapshot = self.history_snapshots.get(node_id).cloned();
        let event = self.domain.finalize_node(node_id, status)?;
        self.history_snapshots
            .insert(node_id.to_string(), history_snapshot);
        Some(ChatTreeNodeFinalization {
            event,
            previous_node,
            previous_revision,
            previous_history_snapshot,
        })
    }

    pub(crate) fn rollback_node_start(&mut self, start: &ChatTreeNodeStart) {
        self.domain.rollback_started_node(
            &start.event,
            start.previous_node_id.clone(),
            start.previous_revision,
        );
        self.history_snapshots.remove(&start.event.node_id);
    }

    pub(crate) fn rollback_node_finalization(&mut self, finalization: &ChatTreeNodeFinalization) {
        self.domain.restore_node_for_rollback(
            finalization.previous_node.clone(),
            finalization.previous_revision,
        );
        if let Some(history_snapshot) = &finalization.previous_history_snapshot {
            self.history_snapshots
                .insert(finalization.event.node_id.clone(), history_snapshot.clone());
        } else {
            self.history_snapshots.remove(&finalization.event.node_id);
        }
    }

    pub(crate) fn update_current_history_snapshot(
        &mut self,
        history_snapshot: ContextManager,
    ) -> bool {
        let Some(current_node_id) = self.domain.current_node_id() else {
            return false;
        };
        self.history_snapshots
            .insert(current_node_id.to_string(), history_snapshot);
        true
    }

    pub(crate) fn set_current_node(
        &mut self,
        node_id: &str,
        expected_revision: Option<u64>,
    ) -> Result<ChatTreeSelection, ChatTreeError> {
        if let Some(expected) = expected_revision
            && expected != self.domain.revision()
        {
            return Err(ChatTreeError::RevisionConflict {
                expected,
                actual: self.domain.revision(),
            });
        }
        if self.domain.node(node_id).is_none() {
            return Err(ChatTreeError::UnknownNode(node_id.to_string()));
        }
        let Some(history) = self.history_snapshots.get(node_id).cloned() else {
            return Err(ChatTreeError::MissingSnapshot(node_id.to_string()));
        };
        let previous_node_id = self.domain.current_node_id().map(str::to_string);
        let previous_revision = self.domain.revision();
        let event = self
            .domain
            .set_current_node(node_id, expected_revision)
            .map_err(ChatTreeError::from)?;
        Ok(ChatTreeSelection {
            event,
            history,
            previous_node_id,
            previous_revision,
        })
    }

    pub(crate) fn rollback_current_node_change(&mut self, selection: &ChatTreeSelection) {
        self.domain.rollback_current_node_change(
            &selection.event,
            selection.previous_node_id.clone(),
            selection.previous_revision,
        );
    }

    pub(crate) fn replace_from_replay(
        &mut self,
        domain: DomainChatTreeState,
        history_snapshots: HashMap<String, ContextManager>,
    ) {
        self.domain = domain;
        self.history_snapshots = history_snapshots;
    }

    pub(crate) fn projection(&self) -> ChatTreeProjectionSnapshot {
        ChatTreeProjectionSnapshot::from(self.domain.projection())
    }
}

impl From<ChatTreeApplyError> for ChatTreeError {
    fn from(value: ChatTreeApplyError) -> Self {
        match value {
            ChatTreeApplyError::DuplicateNode(node_id) => ChatTreeError::Persistence(format!(
                "duplicate chat tree node in domain event: {node_id}"
            )),
            ChatTreeApplyError::MissingParent {
                node_id,
                parent_node_id,
            } => ChatTreeError::Persistence(format!(
                "chat tree node {node_id} references missing parent {parent_node_id}"
            )),
            ChatTreeApplyError::UnknownNode(node_id) => ChatTreeError::UnknownNode(node_id),
            ChatTreeApplyError::RevisionConflict { expected, actual } => {
                ChatTreeError::RevisionConflict { expected, actual }
            }
            ChatTreeApplyError::NonIncreasingRevision { current, incoming } => {
                ChatTreeError::Persistence(format!(
                    "stale chat tree event revision {incoming} ignored at revision {current}"
                ))
            }
        }
    }
}

impl From<DomainChatTreeProjection> for ChatTreeProjectionSnapshot {
    fn from(value: DomainChatTreeProjection) -> Self {
        Self {
            version: value.version,
            revision: value.revision,
            current_node_id: value.current_node_id,
            visible_node_ids: value.visible_node_ids,
            visible_turn_ids: value.visible_turn_ids,
            nodes: value.nodes.into_iter().map(Into::into).collect(),
        }
    }
}

impl From<DomainChatTreeNode> for ChatTreeNodeSnapshot {
    fn from(value: DomainChatTreeNode) -> Self {
        Self {
            node_id: value.node_id,
            parent_node_id: value.parent_node_id,
            turn_id: value.turn_id,
            order: value.order,
            status: value.status,
            summary: value.summary,
        }
    }
}

#[derive(Debug)]
pub(crate) struct ReplayedChatTree {
    pub(crate) domain: DomainChatTreeState,
    pub(crate) history_snapshots: HashMap<String, ContextManager>,
    pub(crate) current_history: Option<ContextManager>,
    pub(crate) current_reference_context_item: Option<TurnContextItem>,
}

#[cfg(test)]
mod tests {
    use super::*;
    use pretty_assertions::assert_eq;

    #[test]
    fn start_node_uses_current_node_as_parent_and_makes_new_node_current() {
        let mut chat_tree = ChatTreeState::default();

        let root_event = chat_tree
            .start_node("turn-a".to_string(), ContextManager::default())
            .expect("root node should start");
        let child_event = chat_tree
            .start_node("turn-b".to_string(), ContextManager::default())
            .expect("child node should start");

        assert_eq!(
            root_event.event,
            ChatTreeNodeStarted {
                revision: 1,
                node_id: "turn-a".to_string(),
                parent_node_id: None,
                turn_id: Some("turn-a".to_string()),
                order: 0,
            }
        );
        assert_eq!(
            child_event.event,
            ChatTreeNodeStarted {
                revision: 2,
                node_id: "turn-b".to_string(),
                parent_node_id: Some("turn-a".to_string()),
                turn_id: Some("turn-b".to_string()),
                order: 1,
            }
        );
        assert_eq!(
            chat_tree.projection().current_node_id,
            Some("turn-b".to_string())
        );
    }

    #[test]
    fn start_node_ignores_duplicate_turn_ids() {
        let mut chat_tree = ChatTreeState::default();

        assert!(
            chat_tree
                .start_node("turn-a".to_string(), ContextManager::default())
                .is_some()
        );
        assert!(
            chat_tree
                .start_node("turn-a".to_string(), ContextManager::default())
                .is_none()
        );

        let projection = chat_tree.projection();
        assert_eq!(projection.revision, 1);
        assert_eq!(projection.nodes[0].node_id, "turn-a");
    }

    #[test]
    fn set_current_node_rejects_stale_revision_before_switching() {
        let mut chat_tree = ChatTreeState::default();
        chat_tree
            .start_node("turn-a".to_string(), ContextManager::default())
            .expect("node should start");

        let result = chat_tree.set_current_node("turn-a", Some(0));

        assert_eq!(
            result.map(|selection| selection.event),
            Err(ChatTreeError::RevisionConflict {
                expected: 0,
                actual: 1,
            })
        );
        let projection = chat_tree.projection();
        assert_eq!(projection.current_node_id, Some("turn-a".to_string()));
        assert_eq!(projection.revision, 1);
    }

    #[test]
    fn finalize_node_sets_default_summary_and_status() {
        let mut chat_tree = ChatTreeState::default();
        chat_tree
            .start_node("turn-a".to_string(), ContextManager::default())
            .expect("node should start");

        let event = chat_tree
            .finalize_node(
                "turn-a",
                ChatTreeNodeStatus::Interrupted,
                ContextManager::default(),
            )
            .expect("node should finalize");

        assert_eq!(
            event.event,
            ChatTreeNodeFinalized {
                revision: 2,
                node_id: "turn-a".to_string(),
                status: ChatTreeNodeStatus::Interrupted,
            }
        );
        assert_eq!(
            chat_tree.projection().nodes,
            vec![ChatTreeNodeSnapshot {
                node_id: "turn-a".to_string(),
                parent_node_id: None,
                turn_id: Some("turn-a".to_string()),
                order: 0,
                status: ChatTreeNodeStatus::Interrupted,
                summary: Some("Turn 1 · interrupted".to_string()),
            }]
        );
        assert!(chat_tree.history_snapshots.contains_key("turn-a"));
    }
}
