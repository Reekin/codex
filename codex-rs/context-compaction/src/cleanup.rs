use std::collections::HashMap;
use std::collections::HashSet;

use codex_history::LocalCompactionKind;
use codex_history::LocalCompactionSource;
use codex_history::ResponseItemEnvelope;
use codex_protocol::models::FunctionCallOutputBody;
use codex_protocol::models::FunctionCallOutputContentItem;
use codex_protocol::models::ResponseItem;
use serde::Deserialize;
use serde::Serialize;

use crate::CompactionError;
use crate::MAX_FRAGMENT_BYTES;
use crate::bounded_json;
use crate::groups::call_key;
use crate::groups::output_key;
use crate::groups::protected_start;

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "action", rename_all = "snake_case", deny_unknown_fields)]
pub enum Decision {
    Keep { id: String },
    Shorten { id: String, text: String },
    Drop { id: String },
}

impl Decision {
    fn id(&self) -> &str {
        match self {
            Self::Keep { id } | Self::Shorten { id, .. } | Self::Drop { id } => id,
        }
    }
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Analysis {
    decisions: Vec<Decision>,
}

/// In-memory speculation only. Installation must compare the source under the history lock.
#[derive(Debug, Clone, Default)]
pub struct StagedDecisions {
    pub source: Vec<ResponseItemEnvelope>,
    decisions: Vec<Decision>,
}

pub fn eligible_results(items: &[ResponseItemEnvelope]) -> Vec<String> {
    let protected = protected_start(items);
    completed_results(&items[..protected])
}

fn completed_results(items: &[ResponseItemEnvelope]) -> Vec<String> {
    let calls: HashSet<_> = items
        .iter()
        .filter_map(|item| call_key(&item.item))
        .collect();
    items.iter().filter_map(|envelope| {
        let item = &envelope.item;
        let key = output_key(item)?;
        if !calls.contains(&key)
            || envelope.metadata.as_ref().and_then(|metadata| metadata.local_compaction.as_ref()).is_some()
            || !matches!(item,
                ResponseItem::FunctionCallOutput { output, .. } | ResponseItem::CustomToolCallOutput { output, .. }
                if output.content_items().is_none_or(|parts| parts.iter().all(|part| matches!(part, FunctionCallOutputContentItem::InputText { .. }))))
        {
            return None;
        }
        item.id().map(ToString::to_string)
    }).collect()
}

impl StagedDecisions {
    /// Only unchanged records count as marked; unrelated edits do not invalidate decisions.
    pub fn contains(&self, item: &ResponseItemEnvelope) -> bool {
        self.source.iter().any(|source| source.item == item.item)
    }

    /// Optimistic removable view includes completed results that can age out of protection.
    /// Calls, archive references, known keeps and already installed reductions remain intact.
    pub fn optimistic_replacement(
        &self,
        current: &[ResponseItemEnvelope],
    ) -> Vec<ResponseItemEnvelope> {
        let candidates: HashSet<_> = completed_results(current).into_iter().collect();
        let mut result = current.to_vec();
        for item in &mut result {
            if self.contains(item)
                || !item
                    .item
                    .id()
                    .is_some_and(|id| candidates.contains(id.as_str()))
            {
                continue;
            }
            let id = item.item.id().expect("candidate has ID").to_string();
            if let ResponseItem::FunctionCallOutput { output, .. }
            | ResponseItem::CustomToolCallOutput { output, .. } = &mut item.item
            {
                // The original-ID reference is mandatory; a future concise replacement
                // can be smaller than the drop notice. Omit optional prose for this bound.
                let reference = format!("[Original item: {id}; use recall_read_item.]");
                if reference.len() < output.to_string().len() {
                    output.body = FunctionCallOutputBody::Text(reference);
                }
            }
        }
        result
    }

    pub fn retain_current(&mut self, current: &[ResponseItemEnvelope]) {
        self.source
            .retain(|source| current.iter().any(|item| item.item == source.item));
        let ids: HashSet<_> = self
            .source
            .iter()
            .filter_map(|source| source.item.id())
            .map(ToString::to_string)
            .collect();
        self.decisions
            .retain(|decision| ids.contains(decision.id()));
    }

    /// Validate the complete response before making any decision available to the adapter.
    pub fn parse(source: Vec<ResponseItemEnvelope>, json: &str) -> Result<Self, CompactionError> {
        let eligible = eligible_results(&source);
        Self::parse_candidates(source, &eligible, json)
    }

    pub fn parse_candidates(
        source: Vec<ResponseItemEnvelope>,
        candidates: &[String],
        json: &str,
    ) -> Result<Self, CompactionError> {
        let analysis: Analysis = bounded_json(json)?;
        let eligible: HashSet<_> = eligible_results(&source).into_iter().collect();
        let allowed: HashSet<_> = candidates.iter().cloned().collect();
        if !allowed.is_subset(&eligible) {
            return Err(CompactionError::Invalid("candidate is not eligible"));
        }
        let mut seen = HashSet::new();
        for decision in &analysis.decisions {
            if !allowed.contains(decision.id()) || !seen.insert(decision.id()) {
                return Err(CompactionError::Invalid(
                    "foreign, protected or duplicate result ID",
                ));
            }
            if let Decision::Shorten { text, .. } = decision
                && (text.is_empty() || text.len() + decision.id().len() + 64 > MAX_FRAGMENT_BYTES)
            {
                return Err(CompactionError::Invalid(
                    "replacement text is empty or oversized",
                ));
            }
        }
        if seen.len() != allowed.len() {
            return Err(CompactionError::Invalid(
                "every eligible result needs one decision",
            ));
        }
        Ok(Self {
            source: source
                .into_iter()
                .filter(|item| {
                    item.item
                        .id()
                        .is_some_and(|id| allowed.contains(id.as_str()))
                })
                .collect(),
            decisions: analysis.decisions,
        })
    }

    /// Extend a validated speculative batch while preserving decisions about earlier results.
    pub fn merge(&mut self, newer: Self) {
        let updated: HashSet<_> = newer.decisions.iter().map(Decision::id).collect();
        self.decisions
            .retain(|decision| !updated.contains(decision.id()));
        self.source.retain(|item| {
            !item
                .item
                .id()
                .is_some_and(|id| updated.contains(id.as_str()))
        });
        self.source.extend(newer.source);
        self.decisions.extend(newer.decisions);
    }

    /// Preserve calls and IDs, replacing dropped output with a small archive reference.
    pub fn apply(
        &self,
        current: &[ResponseItemEnvelope],
    ) -> Result<Vec<ResponseItemEnvelope>, CompactionError> {
        let eligible: HashSet<_> = eligible_results(current).into_iter().collect();
        let decisions: HashMap<_, _> = self
            .decisions
            .iter()
            .map(|decision| (decision.id(), decision))
            .collect();
        let mut replacement = current.to_vec();
        for envelope in &mut replacement {
            if !self.contains(envelope) {
                continue;
            }
            let Some(id) = envelope.item.id().map(ToString::to_string) else {
                continue;
            };
            let Some(decision) = decisions.get(id.as_str()) else {
                continue;
            };
            if !eligible.contains(&id) {
                continue;
            }
            let text = match decision {
                Decision::Keep { .. } => continue,
                Decision::Shorten { text, .. } => {
                    format!("{text}\n[Original item: {id}; use recall_read_item.]")
                }
                Decision::Drop { .. } => {
                    format!("[Output omitted; original item: {id}; use recall_read_item.]")
                }
            };
            match &mut envelope.item {
                ResponseItem::FunctionCallOutput { output, .. }
                | ResponseItem::CustomToolCallOutput { output, .. } => {
                    if text.len() >= output.to_string().len() {
                        continue;
                    }
                    output.body = FunctionCallOutputBody::Text(text);
                    envelope.metadata.get_or_insert_default().local_compaction =
                        Some(LocalCompactionSource {
                            first_item_id: id.clone(),
                            last_item_id: id,
                            kind: LocalCompactionKind::ToolResult,
                        });
                }
                _ => {
                    return Err(CompactionError::Invalid(
                        "decision target is no longer a tool result",
                    ));
                }
            }
        }
        Ok(replacement)
    }
}
