use codex_app_server_protocol::ChatTreeChange;
use codex_app_server_protocol::ChatTreeChangeKind;
use codex_app_server_protocol::ChatTreeNode;
use codex_app_server_protocol::ChatTreeNodeStatus;
use codex_app_server_protocol::ChatTreeProjection;
use codex_core::ChatTreeProjectionSnapshot as CoreChatTreeProjectionSnapshot;

pub(crate) fn chat_tree_projection_from_core(
    projection: CoreChatTreeProjectionSnapshot,
) -> ChatTreeProjection {
    ChatTreeProjection {
        version: projection.version,
        revision: projection.revision,
        current_node_id: projection.current_node_id,
        visible_node_ids: projection.visible_node_ids,
        visible_turn_ids: projection.visible_turn_ids,
        nodes: projection
            .nodes
            .into_iter()
            .map(|node| ChatTreeNode {
                node_id: node.node_id,
                parent_node_id: node.parent_node_id,
                turn_id: node.turn_id,
                order: node.order,
                status: node.status.into(),
                summary: node.summary,
            })
            .collect(),
    }
}

pub(crate) fn chat_tree_projection_contains_change(
    chat_tree: &ChatTreeProjection,
    change: &ChatTreeChange,
    expected_summary: Option<Option<&str>>,
) -> bool {
    let Some(node_id) = change.node_id.as_deref() else {
        return true;
    };
    match change.r#type {
        ChatTreeChangeKind::NodeStarted => {
            chat_tree.nodes.iter().any(|node| node.node_id == node_id)
        }
        ChatTreeChangeKind::NodeSummaryUpdated => chat_tree.nodes.iter().any(|node| {
            node.node_id == node_id
                && expected_summary
                    .is_none_or(|expected_summary| node.summary.as_deref() == expected_summary)
        }),
        ChatTreeChangeKind::NodeFinalized => chat_tree
            .nodes
            .iter()
            .any(|node| node.node_id == node_id && node.status != ChatTreeNodeStatus::Pending),
        ChatTreeChangeKind::CurrentNodeChanged => {
            chat_tree.current_node_id.as_deref() == Some(node_id)
        }
        ChatTreeChangeKind::TreeRebuilt => true,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn projection_with_summary(summary: Option<&str>) -> ChatTreeProjection {
        ChatTreeProjection {
            version: 1,
            revision: 1,
            current_node_id: Some("node-a".to_string()),
            visible_node_ids: vec!["node-a".to_string()],
            visible_turn_ids: vec!["turn-a".to_string()],
            nodes: vec![ChatTreeNode {
                node_id: "node-a".to_string(),
                parent_node_id: None,
                turn_id: Some("turn-a".to_string()),
                order: 0,
                status: ChatTreeNodeStatus::Completed,
                summary: summary.map(str::to_string),
            }],
        }
    }

    #[test]
    fn node_summary_updated_requires_expected_summary_when_available() {
        let change = ChatTreeChange {
            r#type: ChatTreeChangeKind::NodeSummaryUpdated,
            node_id: Some("node-a".to_string()),
        };

        assert!(chat_tree_projection_contains_change(
            &projection_with_summary(Some("new summary")),
            &change,
            Some(Some("new summary")),
        ));
        assert!(chat_tree_projection_contains_change(
            &projection_with_summary(None),
            &change,
            Some(None),
        ));
        assert!(!chat_tree_projection_contains_change(
            &projection_with_summary(Some("old summary")),
            &change,
            Some(Some("new summary")),
        ));
    }
}
