use crate::chat_tree::ChatTreeCurrentNodeChanged;
use crate::chat_tree::ChatTreeEvent;
use crate::chat_tree::ChatTreeNodeFinalized;
use crate::chat_tree::ChatTreeNodeStarted;
use crate::chat_tree::ChatTreeNodeSummaryUpdated;
use crate::protocol::ChatTreeCurrentNodeChangedEvent;
use crate::protocol::ChatTreeNodeFinalizedEvent;
use crate::protocol::ChatTreeNodeStartedEvent;
use crate::protocol::ChatTreeNodeSummaryUpdatedEvent;
use crate::protocol::EventMsg;
use crate::protocol::ThreadRolledBackEvent;
use crate::protocol::TurnStartedEvent;

impl From<ChatTreeNodeStarted> for ChatTreeNodeStartedEvent {
    fn from(value: ChatTreeNodeStarted) -> Self {
        Self {
            revision: value.revision,
            node_id: value.node_id,
            parent_node_id: value.parent_node_id,
            turn_id: value.turn_id,
            order: value.order,
        }
    }
}

impl From<ChatTreeNodeFinalized> for ChatTreeNodeFinalizedEvent {
    fn from(value: ChatTreeNodeFinalized) -> Self {
        Self {
            revision: value.revision,
            node_id: value.node_id,
            status: value.status,
        }
    }
}

impl From<ChatTreeNodeSummaryUpdated> for ChatTreeNodeSummaryUpdatedEvent {
    fn from(value: ChatTreeNodeSummaryUpdated) -> Self {
        Self {
            revision: value.revision,
            node_id: value.node_id,
            summary: value.summary,
        }
    }
}

impl From<ChatTreeCurrentNodeChanged> for ChatTreeCurrentNodeChangedEvent {
    fn from(value: ChatTreeCurrentNodeChanged) -> Self {
        Self {
            revision: value.revision,
            node_id: value.node_id,
        }
    }
}

pub fn chat_tree_event_from_protocol_event(event: &EventMsg) -> Option<ChatTreeEvent> {
    match event {
        EventMsg::ChatTreeNodeStarted(payload) => Some(ChatTreeEvent::NodeStarted {
            revision: payload.revision,
            node_id: payload.node_id.clone(),
            parent_node_id: payload.parent_node_id.clone(),
            turn_id: payload.turn_id.clone(),
            order: payload.order,
        }),
        EventMsg::ChatTreeNodeFinalized(payload) => Some(ChatTreeEvent::NodeFinalized {
            revision: payload.revision,
            node_id: payload.node_id.clone(),
            status: payload.status,
        }),
        EventMsg::ChatTreeNodeSummaryUpdated(payload) => Some(ChatTreeEvent::NodeSummaryUpdated {
            revision: payload.revision,
            node_id: payload.node_id.clone(),
            summary: payload.summary.clone(),
        }),
        EventMsg::ChatTreeCurrentNodeChanged(payload) => Some(ChatTreeEvent::CurrentNodeChanged {
            revision: payload.revision,
            node_id: payload.node_id.clone(),
        }),
        EventMsg::TurnStarted(TurnStartedEvent { turn_id, .. }) => {
            Some(ChatTreeEvent::LegacyTurnStarted {
                turn_id: turn_id.clone(),
            })
        }
        EventMsg::ThreadRolledBack(ThreadRolledBackEvent { num_turns }) => {
            Some(ChatTreeEvent::RollbackVisibleTurns {
                num_turns: *num_turns,
            })
        }
        _ => None,
    }
}
