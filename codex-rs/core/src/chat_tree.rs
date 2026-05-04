use crate::context_manager::ContextManager;
use codex_protocol::protocol::ChatTreeChangeKind;
use codex_protocol::protocol::ChatTreeCurrentNodeChangedEvent;
use codex_protocol::protocol::ChatTreeNodeFinalizedEvent;
use codex_protocol::protocol::ChatTreeNodeStartedEvent;
use codex_protocol::protocol::ChatTreeNodeStatus;
use codex_protocol::protocol::TurnContextItem;
use std::collections::HashMap;

#[derive(Debug, Clone)]
pub(crate) struct ChatTreeNodeState {
    pub(crate) parent_node_id: Option<String>,
    pub(crate) turn_id: Option<String>,
    pub(crate) order: u64,
    pub(crate) status: ChatTreeNodeStatus,
    pub(crate) summary: Option<String>,
    pub(crate) history_snapshot: Option<ContextManager>,
}

#[derive(Debug, Clone, Default)]
pub(crate) struct ChatTreeState {
    revision: u64,
    current_node_id: Option<String>,
    nodes: HashMap<String, ChatTreeNodeState>,
    ordered_node_ids: Vec<String>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ChatTreeError {
    UnknownNode(String),
    MissingSnapshot(String),
    RevisionConflict { expected: u64, actual: u64 },
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
    pub(crate) event: ChatTreeCurrentNodeChangedEvent,
    pub(crate) history: ContextManager,
}

impl ChatTreeState {
    pub(crate) fn start_node(
        &mut self,
        turn_id: String,
        parent_history: ContextManager,
    ) -> Option<ChatTreeNodeStartedEvent> {
        let node_id = turn_id.clone();
        let event_turn_id = turn_id.clone();
        if self.nodes.contains_key(&node_id) {
            return None;
        }
        let parent_node_id = self.current_node_id.clone();
        let order = self.ordered_node_ids.len() as u64;
        self.revision = self.revision.saturating_add(1);
        let node = ChatTreeNodeState {
            parent_node_id: parent_node_id.clone(),
            turn_id: Some(turn_id),
            order,
            status: ChatTreeNodeStatus::Pending,
            summary: None,
            history_snapshot: Some(parent_history),
        };
        self.nodes.insert(node_id.clone(), node);
        self.ordered_node_ids.push(node_id.clone());
        self.current_node_id = Some(node_id.clone());
        Some(ChatTreeNodeStartedEvent {
            revision: self.revision,
            node_id,
            parent_node_id,
            turn_id: Some(event_turn_id),
            order,
        })
    }

    pub(crate) fn finalize_node(
        &mut self,
        node_id: &str,
        status: ChatTreeNodeStatus,
        history_snapshot: ContextManager,
    ) -> Option<ChatTreeNodeFinalizedEvent> {
        let node = self.nodes.get_mut(node_id)?;
        node.status = status;
        if node.summary.is_none() {
            node.summary = Some(default_summary(status));
        }
        node.history_snapshot = Some(history_snapshot);
        self.revision = self.revision.saturating_add(1);
        Some(ChatTreeNodeFinalizedEvent {
            revision: self.revision,
            node_id: node_id.to_string(),
            status,
        })
    }

    pub(crate) fn set_current_node(
        &mut self,
        node_id: &str,
        expected_revision: Option<u64>,
    ) -> Result<ChatTreeSelection, ChatTreeError> {
        if let Some(expected) = expected_revision
            && expected != self.revision
        {
            return Err(ChatTreeError::RevisionConflict {
                expected,
                actual: self.revision,
            });
        }
        let Some(node) = self.nodes.get(node_id) else {
            return Err(ChatTreeError::UnknownNode(node_id.to_string()));
        };
        let Some(history) = node.history_snapshot.clone() else {
            return Err(ChatTreeError::MissingSnapshot(node_id.to_string()));
        };
        self.current_node_id = Some(node_id.to_string());
        self.revision = self.revision.saturating_add(1);
        Ok(ChatTreeSelection {
            event: ChatTreeCurrentNodeChangedEvent {
                revision: self.revision,
                node_id: node_id.to_string(),
                change_kind: ChatTreeChangeKind::CurrentNodeChanged,
            },
            history,
        })
    }

    pub(crate) fn replace_from_replay(
        &mut self,
        nodes: HashMap<String, ChatTreeNodeState>,
        ordered_node_ids: Vec<String>,
        current_node_id: Option<String>,
        revision: u64,
    ) {
        self.nodes = nodes;
        self.ordered_node_ids = ordered_node_ids;
        self.current_node_id = current_node_id;
        self.revision = revision;
    }

    pub(crate) fn projection(&self) -> ChatTreeProjectionSnapshot {
        let mut visible_node_ids = Vec::new();
        let mut next_node_id = self.current_node_id.clone();
        while let Some(node_id) = next_node_id {
            let Some(node) = self.nodes.get(&node_id) else {
                break;
            };
            visible_node_ids.push(node_id);
            next_node_id = node.parent_node_id.clone();
        }
        visible_node_ids.reverse();
        let visible_turn_ids = visible_node_ids
            .iter()
            .filter_map(|node_id| self.nodes.get(node_id))
            .filter_map(|node| node.turn_id.clone())
            .collect();
        let nodes = self
            .ordered_node_ids
            .iter()
            .filter_map(|node_id| {
                let node = self.nodes.get(node_id)?;
                Some(ChatTreeNodeSnapshot {
                    node_id: node_id.clone(),
                    parent_node_id: node.parent_node_id.clone(),
                    turn_id: node.turn_id.clone(),
                    order: node.order,
                    status: node.status,
                    summary: node.summary.clone(),
                })
            })
            .collect();

        ChatTreeProjectionSnapshot {
            version: 1,
            revision: self.revision,
            current_node_id: self.current_node_id.clone(),
            visible_node_ids,
            visible_turn_ids,
            nodes,
        }
    }
}

pub(crate) fn default_summary(status: ChatTreeNodeStatus) -> String {
    match status {
        ChatTreeNodeStatus::Pending => "turn pending",
        ChatTreeNodeStatus::Completed => "turn completed",
        ChatTreeNodeStatus::Interrupted => "turn interrupted",
        ChatTreeNodeStatus::Replaced => "turn replaced",
        ChatTreeNodeStatus::ReviewEnded => "turn review ended",
    }
    .to_string()
}

#[derive(Debug)]
pub(crate) struct ReplayedChatTree {
    pub(crate) nodes: HashMap<String, ChatTreeNodeState>,
    pub(crate) ordered_node_ids: Vec<String>,
    pub(crate) current_node_id: Option<String>,
    pub(crate) revision: u64,
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
            root_event,
            ChatTreeNodeStartedEvent {
                revision: 1,
                node_id: "turn-a".to_string(),
                parent_node_id: None,
                turn_id: Some("turn-a".to_string()),
                order: 0,
            }
        );
        assert_eq!(
            child_event,
            ChatTreeNodeStartedEvent {
                revision: 2,
                node_id: "turn-b".to_string(),
                parent_node_id: Some("turn-a".to_string()),
                turn_id: Some("turn-b".to_string()),
                order: 1,
            }
        );
        assert_eq!(chat_tree.current_node_id, Some("turn-b".to_string()));
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

        assert_eq!(chat_tree.revision, 1);
        assert_eq!(chat_tree.ordered_node_ids, vec!["turn-a".to_string()]);
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
        assert_eq!(chat_tree.current_node_id, Some("turn-a".to_string()));
        assert_eq!(chat_tree.revision, 1);
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
            event,
            ChatTreeNodeFinalizedEvent {
                revision: 2,
                node_id: "turn-a".to_string(),
                status: ChatTreeNodeStatus::Interrupted,
            }
        );
        assert_eq!(
            chat_tree.nodes.get("turn-a").map(|node| (
                node.status,
                node.summary.as_deref(),
                node.history_snapshot.is_some()
            )),
            Some((
                ChatTreeNodeStatus::Interrupted,
                Some("turn interrupted"),
                true
            ))
        );
    }
}
