use std::collections::HashMap;

use codex_history::ResponseItemEnvelope;
use codex_protocol::models::ResponseItem;

pub fn is_user_direction(item: &ResponseItem) -> bool {
    match item {
        ResponseItem::Message {
            role,
            internal_chat_message_metadata_passthrough,
            ..
        } if role == "user" => internal_chat_message_metadata_passthrough
            .as_ref()
            .and_then(|metadata| metadata.content_item_kinds.as_ref())
            .is_none_or(|kinds| {
                kinds.is_empty()
                    || kinds
                        .iter()
                        .any(|kind| kind.0.starts_with("user.") || kind.0 == "unknown")
            }),
        ResponseItem::AgentMessage { .. } => true,
        _ => false,
    }
}

// Type-qualified keys keep unrelated tool families from sharing a call ID.
pub(crate) fn call_key(item: &ResponseItem) -> Option<(&str, &str)> {
    match item {
        ResponseItem::FunctionCall { call_id, .. }
        | ResponseItem::LocalShellCall {
            call_id: Some(call_id),
            ..
        } => Some(("function", call_id)),
        ResponseItem::CustomToolCall { call_id, .. } => Some(("custom", call_id)),
        ResponseItem::ToolSearchCall {
            call_id: Some(call_id),
            ..
        } => Some(("search", call_id)),
        _ => None,
    }
}

pub(crate) fn output_key(item: &ResponseItem) -> Option<(&str, &str)> {
    match item {
        ResponseItem::FunctionCallOutput {
            call_id: Some(call_id),
            ..
        } => Some(("function", call_id)),
        ResponseItem::CustomToolCallOutput { call_id, .. } => Some(("custom", call_id)),
        ResponseItem::ToolSearchOutput {
            call_id: Some(call_id),
            ..
        } => Some(("search", call_id)),
        _ => None,
    }
}

/// Every returned cut preserves complete call/result groups on either side.
pub(crate) fn safe_cuts(items: &[ResponseItemEnvelope]) -> Vec<usize> {
    let mut calls = HashMap::new();
    let mut spans = Vec::new();
    for (index, item) in items.iter().enumerate() {
        if let Some(key) = call_key(&item.item) {
            calls.insert(key, index);
        }
        if let Some(key) = output_key(&item.item)
            && let Some(start) = calls.remove(&key)
        {
            spans.push((start, index));
        }
    }
    spans.extend(calls.into_values().map(|start| (start, items.len())));
    (0..=items.len())
        .filter(|cut| !spans.iter().any(|(start, end)| start < cut && cut <= end))
        .collect()
}

pub(crate) fn protected_start(items: &[ResponseItemEnvelope]) -> usize {
    // Long agent turns may contain hundreds of completed calls. Protect the newest work group,
    // while unresolved calls pull the safe boundary back until their output arrives.
    let recent = items
        .iter()
        .rposition(|item| call_key(&item.item).is_some())
        .max(items.iter().rposition(|item| is_user_direction(&item.item)))
        .unwrap_or(0);
    safe_cuts(items)
        .into_iter()
        .filter(|cut| *cut <= recent)
        .max()
        .unwrap_or(0)
}

pub(crate) fn pinned(item: &ResponseItemEnvelope) -> bool {
    match &item.item {
        ResponseItem::Message { role, .. } if role == "developer" || role == "system" => true,
        ResponseItem::Message {
            role,
            internal_chat_message_metadata_passthrough: Some(metadata),
            ..
        } if role == "user" => metadata.content_item_kinds.as_ref().is_some_and(|kinds| {
            kinds.iter().any(|kind| {
                kind.0.ends_with(".instructions") || kind.0 == "environments.environment_context"
            })
        }),
        _ => false,
    }
}
