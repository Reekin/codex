//! Full compaction keeps dialogue and still-valuable tool records; tool work already judged low
//! value is folded into one line per uninterrupted run, with originals left to recall.

use std::collections::BTreeMap;
use std::collections::HashMap;
use std::collections::HashSet;

use codex_history::LocalCompactionKind;
use codex_history::ResponseItemEnvelope;
use codex_protocol::models::FunctionCallOutputBody;
use codex_protocol::models::ResponseItem;

use crate::Decision;
use crate::StagedDecisions;
use crate::groups::call_key;
use crate::groups::output_key;
use crate::groups::protected_start;

/// Consecutive low-value tool records replaced by one line.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ToolRun {
    /// Removed records, in history order.
    pub indices: Vec<usize>,
    pub first_item_id: String,
    pub last_item_id: String,
    /// Which tools ran, and the conclusions marking already drew from their results.
    pub text: String,
}

/// Runs of tool records from `from` up to the protected newest work group whose results were
/// judged low value: shortened or dropped by a mark, or already cleaned. Calls are removed
/// with their results, together with the reasoning that led into them; kept and unjudged
/// results stay verbatim. `dropped` records are already gone and do not break a run.
pub fn tool_runs(
    items: &[ResponseItemEnvelope],
    staged: &StagedDecisions,
    dropped: &[bool],
    from: usize,
) -> Vec<ToolRun> {
    let end = protected_start(items);
    let decisions: HashMap<&str, &Decision> = staged
        .marks()
        .map(|(_, decision)| (decision.id(), decision))
        .collect();
    let outputs: HashMap<_, _> = items[..end]
        .iter()
        .enumerate()
        .filter_map(|(index, item)| output_key(&item.item).map(|key| (key, index)))
        .collect();
    // Result index -> conclusion worth keeping, for every low-value result.
    let mut low_value = HashMap::new();
    for (index, item) in items.iter().enumerate().take(end) {
        if output_key(&item.item).is_none() {
            continue;
        }
        let id = item.item.id().map(|id| id.as_str()).unwrap_or_default();
        let note = match decisions.get(id) {
            Some(Decision::Shorten { text, .. }) => Some(Some(text.clone())),
            Some(Decision::Drop { .. }) => Some(None),
            Some(Decision::Keep { .. }) | None => cleaned(item).then(|| cleaned_note(&item.item)),
        };
        if let Some(note) = note {
            low_value.insert(index, note);
        }
    }
    let mut removed = vec![false; items.len()];
    for index in from..end {
        if let Some(output) = call_key(&items[index].item)
            .and_then(|key| outputs.get(&key))
            .filter(|output| low_value.contains_key(*output))
        {
            removed[index] = true;
            removed[*output] = true;
        }
    }
    // Reasoning goes with the removed call it led into.
    for index in (from..end).rev() {
        if matches!(items[index].item, ResponseItem::Reasoning { .. }) && !dropped[index] {
            removed[index] = (index + 1..end)
                .find(|next| {
                    !dropped[*next] && !matches!(items[*next].item, ResponseItem::Reasoning { .. })
                })
                .is_some_and(|next| removed[next] && call_key(&items[next].item).is_some());
        }
    }
    let names: HashMap<_, _> = items
        .iter()
        .filter_map(|item| call_key(&item.item).map(|key| (key, tool_name(&item.item))))
        .collect();
    let mut runs = Vec::new();
    let mut current: Vec<usize> = Vec::new();
    for index in from..=end {
        if index < end && dropped[index] {
            continue;
        }
        if index < end && removed[index] {
            current.push(index);
            continue;
        }
        if !current.is_empty() {
            runs.push(run(items, std::mem::take(&mut current), &names, &low_value));
        }
    }
    runs
}

fn run(
    items: &[ResponseItemEnvelope],
    indices: Vec<usize>,
    names: &HashMap<(&str, &str), String>,
    low_value: &HashMap<usize, Option<String>>,
) -> ToolRun {
    let mut calls = HashSet::new();
    let mut notes = Vec::new();
    for index in &indices {
        let item = &items[*index].item;
        if let Some(key) = call_key(item).or_else(|| output_key(item)) {
            calls.insert(key);
        }
        if let Some(Some(note)) = low_value.get(index) {
            let name = output_key(item)
                .and_then(|key| names.get(&key))
                .map_or("tool", String::as_str);
            let id = item.id().map(|id| id.as_str()).unwrap_or_default();
            notes.push(format!("- {name} ({id}): {}", note.trim()));
        }
    }
    let mut counts = BTreeMap::new();
    for key in &calls {
        let name = names.get(key).map_or("tool", String::as_str);
        *counts.entry(name).or_insert(0usize) += 1;
    }
    let mut counts: Vec<_> = counts.into_iter().collect();
    counts.sort_by(|a, b| b.1.cmp(&a.1).then(a.0.cmp(b.0)));
    let tools = counts
        .iter()
        .map(|(name, count)| format!("{name} ×{count}"))
        .collect::<Vec<_>>()
        .join(", ");
    let mut text = format!("Removed tool calls: {tools}.");
    for note in notes {
        text.push('\n');
        text.push_str(&note);
    }
    let id = |index: &usize| {
        items[*index]
            .item
            .id()
            .map(ToString::to_string)
            .unwrap_or_default()
    };
    ToolRun {
        first_item_id: indices.first().map(id).unwrap_or_default(),
        last_item_id: indices.last().map(id).unwrap_or_default(),
        indices,
        text,
    }
}

fn cleaned(item: &ResponseItemEnvelope) -> bool {
    item.metadata
        .as_ref()
        .and_then(|metadata| metadata.local_compaction.as_ref())
        .is_some_and(|source| source.kind == LocalCompactionKind::ToolResult)
}

/// The conclusion an earlier cleanup left in place of a result; dropped results have none.
fn cleaned_note(item: &ResponseItem) -> Option<String> {
    let (ResponseItem::FunctionCallOutput { output, .. }
    | ResponseItem::CustomToolCallOutput { output, .. }) = item
    else {
        return None;
    };
    let FunctionCallOutputBody::Text(text) = &output.body else {
        return None;
    };
    let (note, _) = text.rsplit_once('\n')?;
    (!note.trim().is_empty()).then(|| note.to_string())
}

fn tool_name(item: &ResponseItem) -> String {
    match item {
        ResponseItem::FunctionCall { name, .. } | ResponseItem::CustomToolCall { name, .. } => {
            name.clone()
        }
        ResponseItem::LocalShellCall { .. } => "local_shell".to_string(),
        ResponseItem::ToolSearchCall { .. } => "tool_search".to_string(),
        _ => "tool".to_string(),
    }
}

#[cfg(test)]
#[path = "condense_tests.rs"]
mod tests;
