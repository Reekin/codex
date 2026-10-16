use std::collections::HashMap;
use std::collections::HashSet;

use codex_history::ResponseItemEnvelope;
use codex_protocol::models::ContentItem;
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

/// Messages Codex renders from typed fragments: instructions, environment and notices. Codex
/// tags each part with its kind; user input carries `user.*` kinds and messages a client
/// injected carry none, which reads as `unknown`.
pub(crate) fn rendered(item: &ResponseItem) -> bool {
    let ResponseItem::Message {
        role,
        internal_chat_message_metadata_passthrough: Some(metadata),
        ..
    } = item
    else {
        return false;
    };
    role != "assistant"
        && metadata.content_item_kinds.as_ref().is_some_and(|kinds| {
            !kinds.is_empty()
                && kinds
                    .iter()
                    .all(|kind| kind.0 != "unknown" && !kind.0.starts_with("user."))
        })
}

/// Developer or system messages a client injected; Codex cannot render them again.
pub(crate) fn client_context(item: &ResponseItem) -> bool {
    matches!(item, ResponseItem::Message { role, .. } if role == "developer" || role == "system")
        && !rendered(item)
}

/// Older copies of client-injected messages that recur later with identical text.
pub(crate) fn repeated_client_context(items: &[ResponseItemEnvelope]) -> Vec<usize> {
    let mut seen = HashSet::new();
    let mut repeated: Vec<_> = items
        .iter()
        .enumerate()
        .rev()
        .filter_map(|(index, item)| {
            let ResponseItem::Message { role, content, .. } = &item.item else {
                return None;
            };
            if !client_context(&item.item) {
                return None;
            }
            let text: String = content
                .iter()
                .filter_map(|part| match part {
                    ContentItem::InputText { text } | ContentItem::OutputText { text } => {
                        Some(text.as_str())
                    }
                    ContentItem::InputImage { .. } | ContentItem::InputAudio { .. } => None,
                })
                .collect();
            (!seen.insert((role.as_str(), text))).then_some(index)
        })
        .collect();
    repeated.reverse();
    repeated
}

/// What removing message parts left of a record.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Removal {
    Unchanged,
    Partial,
    All,
}

/// Removes the message parts whose kind is in `kinds`, for kinds the caller renders again in
/// full. Kinds are matched exactly, so incremental notices of other kinds stay.
pub fn remove_parts(item: &mut ResponseItem, kinds: &HashSet<String>) -> Removal {
    let ResponseItem::Message {
        content,
        internal_chat_message_metadata_passthrough: Some(metadata),
        ..
    } = item
    else {
        return Removal::Unchanged;
    };
    let Some(part_kinds) = metadata.content_item_kinds.as_mut() else {
        return Removal::Unchanged;
    };
    let removed: Vec<bool> = part_kinds
        .iter()
        .map(|kind| kinds.contains(&kind.0))
        .collect();
    if !removed.contains(&true) {
        return Removal::Unchanged;
    }
    // Parts beyond the recorded kinds are unclassified and stay.
    let mut index = 0;
    content.retain(|_| {
        let keep = !removed.get(index).copied().unwrap_or(false);
        index += 1;
        keep
    });
    let mut index = 0;
    part_kinds.retain(|_| {
        let keep = !removed[index];
        index += 1;
        keep
    });
    if content.is_empty() {
        Removal::All
    } else {
        Removal::Partial
    }
}
