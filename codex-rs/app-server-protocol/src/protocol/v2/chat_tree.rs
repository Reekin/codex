use super::shared::v2_enum_from_core;
use codex_protocol::protocol::ChatTreeChangeKind as CoreChatTreeChangeKind;
use codex_protocol::protocol::ChatTreeNodeStatus as CoreChatTreeNodeStatus;
use schemars::JsonSchema;
use serde::Deserialize;
use serde::Serialize;
use ts_rs::TS;

#[derive(Serialize, Deserialize, Debug, Clone, PartialEq, JsonSchema, TS)]
#[serde(rename_all = "camelCase")]
#[ts(export_to = "v2/")]
pub struct ChatTreeReadParams {
    pub thread_id: String,
}

#[derive(Serialize, Deserialize, Debug, Clone, PartialEq, JsonSchema, TS)]
#[serde(rename_all = "camelCase")]
#[ts(export_to = "v2/")]
pub struct ChatTreeReadResponse {
    pub thread_id: String,
    pub chat_tree: Box<ChatTreeProjection>,
}

#[derive(Serialize, Deserialize, Debug, Clone, PartialEq, JsonSchema, TS)]
#[serde(rename_all = "camelCase")]
#[ts(export_to = "v2/")]
pub struct ChatTreeSetCurrentParams {
    pub thread_id: String,
    pub node_id: String,
    #[ts(optional = nullable)]
    pub expected_revision: Option<u64>,
}

#[derive(Serialize, Deserialize, Debug, Clone, PartialEq, JsonSchema, TS)]
#[serde(rename_all = "camelCase")]
#[ts(export_to = "v2/")]
pub struct ChatTreeSetCurrentResponse {
    pub thread_id: String,
    pub chat_tree: Box<ChatTreeProjection>,
}

#[derive(Serialize, Deserialize, Debug, Clone, PartialEq, JsonSchema, TS)]
#[serde(rename_all = "camelCase")]
#[ts(export_to = "v2/")]
pub struct ChatTreeProjection {
    pub version: u32,
    pub revision: u64,
    pub current_node_id: Option<String>,
    pub visible_node_ids: Vec<String>,
    pub visible_turn_ids: Vec<String>,
    pub nodes: Vec<ChatTreeNode>,
}

#[derive(Serialize, Deserialize, Debug, Clone, PartialEq, JsonSchema, TS)]
#[serde(rename_all = "camelCase")]
#[ts(export_to = "v2/")]
pub struct ChatTreeNode {
    pub node_id: String,
    pub parent_node_id: Option<String>,
    pub turn_id: Option<String>,
    pub order: u64,
    pub status: ChatTreeNodeStatus,
    pub summary: Option<String>,
}

v2_enum_from_core! {
    pub enum ChatTreeNodeStatus from CoreChatTreeNodeStatus {
        Pending,
        Completed,
        Interrupted,
        Replaced,
        ReviewEnded,
    }
}

#[derive(Serialize, Deserialize, Debug, Clone, PartialEq, JsonSchema, TS)]
#[serde(rename_all = "camelCase")]
#[ts(export_to = "v2/")]
pub struct ChatTreeChange {
    pub r#type: ChatTreeChangeKind,
    pub node_id: Option<String>,
}

v2_enum_from_core! {
    pub enum ChatTreeChangeKind from CoreChatTreeChangeKind {
        NodeStarted,
        NodeFinalized,
        NodeSummaryUpdated,
        CurrentNodeChanged,
        TreeRebuilt,
    }
}

#[derive(Serialize, Deserialize, Debug, Clone, PartialEq, JsonSchema, TS)]
#[serde(rename_all = "camelCase")]
#[ts(export_to = "v2/")]
pub struct ChatTreeUpdatedNotification {
    pub thread_id: String,
    pub change: ChatTreeChange,
    pub chat_tree: Box<ChatTreeProjection>,
}
