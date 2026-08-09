use super::*;
use codex_app_server_protocol::ChatTreeNode;
use codex_app_server_protocol::ChatTreeNodeStatus;
use insta::assert_snapshot;
use tokio::sync::mpsc::unbounded_channel;

fn projection() -> ChatTreeProjection {
    ChatTreeProjection {
        version: 1,
        revision: 8,
        current_node_id: Some("node-d".to_string()),
        visible_node_ids: vec!["node-a".to_string(), "node-d".to_string()],
        visible_turn_ids: vec!["turn-a".to_string(), "turn-d".to_string()],
        nodes: vec![
            ChatTreeNode {
                node_id: "node-a".to_string(),
                parent_node_id: None,
                turn_id: Some("turn-a".to_string()),
                order: 0,
                status: ChatTreeNodeStatus::Completed,
                summary: Some("Inspect the project and identify the migration surface".to_string()),
            },
            ChatTreeNode {
                node_id: "node-b".to_string(),
                parent_node_id: Some("node-a".to_string()),
                turn_id: Some("turn-b".to_string()),
                order: 1,
                status: ChatTreeNodeStatus::Completed,
                summary: Some("Implement protocol and core adapters".to_string()),
            },
            ChatTreeNode {
                node_id: "node-c".to_string(),
                parent_node_id: Some("node-b".to_string()),
                turn_id: Some("turn-c".to_string()),
                order: 2,
                status: ChatTreeNodeStatus::Interrupted,
                summary: Some("Turn 3 · interrupted".to_string()),
            },
            ChatTreeNode {
                node_id: "node-d".to_string(),
                parent_node_id: Some("node-a".to_string()),
                turn_id: Some("turn-d".to_string()),
                order: 3,
                status: ChatTreeNodeStatus::Completed,
                summary: Some("Add app-server RPC and TUI validation".to_string()),
            },
        ],
    }
}

fn render_view(width: u16) -> Buffer {
    let (tx, _rx) = unbounded_channel::<AppEvent>();
    let state = ChatTreeUiState {
        projection: projection(),
    };
    let view = state
        .view(AppEventSender::new(tx))
        .expect("projection should create a chat tree view");
    let area = Rect::new(0, 0, width, view.desired_height(width));
    let mut buffer = Buffer::empty(area);
    view.render(area, &mut buffer);
    buffer
}

#[test]
fn chat_tree_overlay_renders_branches_and_current_marker() {
    assert_snapshot!(
        "chat_tree_overlay_branched",
        format!("{:?}", render_view(72))
    );
}

#[test]
fn chat_tree_overlay_wraps_long_summaries() {
    assert_snapshot!("chat_tree_overlay_narrow", format!("{:?}", render_view(36)));
}
