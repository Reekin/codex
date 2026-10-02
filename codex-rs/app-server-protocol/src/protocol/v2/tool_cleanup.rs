use crate::JsonSchema;
use crate::TS;
use serde::Deserialize;
use serde::Serialize;

/// Tool-result cleanup progress for a loaded thread's current model route.
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq, Eq, JsonSchema, TS)]
#[serde(rename_all = "camelCase")]
#[ts(export_to = "v2/")]
pub struct ThreadToolCleanupStatus {
    /// Whether background tool marking runs on the current model route.
    pub enabled: bool,
    /// Whether a background marking request is in flight.
    pub marking: bool,
    /// Estimated context tokens released by applying every current validated mark.
    #[ts(type = "number")]
    pub pending_savings_tokens: i64,
    /// Release required before automatic cleanup applies the marks.
    #[ts(type = "number")]
    pub required_savings_tokens: i64,
}

#[derive(Serialize, Deserialize, Debug, Clone, PartialEq, JsonSchema, TS)]
#[serde(rename_all = "camelCase")]
#[ts(export_to = "v2/")]
pub struct ThreadToolCleanupReadParams {
    pub thread_id: String,
}

#[derive(Serialize, Deserialize, Debug, Clone, PartialEq, JsonSchema, TS)]
#[serde(rename_all = "camelCase")]
#[ts(export_to = "v2/")]
pub struct ThreadToolCleanupReadResponse {
    pub status: ThreadToolCleanupStatus,
}

#[derive(Serialize, Deserialize, Debug, Clone, PartialEq, JsonSchema, TS)]
#[serde(rename_all = "camelCase")]
#[ts(export_to = "v2/")]
pub struct ThreadToolCleanupApplyParams {
    pub thread_id: String,
}

#[derive(Serialize, Deserialize, Debug, Clone, PartialEq, JsonSchema, TS)]
#[serde(rename_all = "camelCase")]
#[ts(export_to = "v2/")]
pub struct ThreadToolCleanupApplyResponse {
    /// Estimated context tokens released; zero when no mark could shorten the history.
    #[ts(type = "number")]
    pub released_tokens: i64,
    pub status: ThreadToolCleanupStatus,
}
