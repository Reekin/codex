use codex_app_server_protocol::ChatTreeNode;
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
