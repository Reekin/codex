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
use crate::groups::is_user_direction;
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
    let calls: HashSet<_> = items[..protected]
        .iter()
        .filter_map(|item| call_key(&item.item))
        .collect();
    items[..protected].iter().filter_map(|envelope| {
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
    pub fn kept_ids(&self) -> Vec<String> {
        self.decisions
            .iter()
            .filter_map(|decision| match decision {
                Decision::Keep { id } => Some(id.clone()),
                Decision::Shorten { .. } | Decision::Drop { .. } => None,
            })
            .collect()
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
            source,
            decisions: analysis.decisions,
        })
    }

    pub fn is_current(&self, current: &[ResponseItemEnvelope]) -> bool {
        !self.source.is_empty()
            && current.starts_with(&self.source)
            && !current[self.source.len()..]
                .iter()
                .any(|item| is_user_direction(&item.item))
    }

    /// Extend a validated speculative batch while preserving decisions about earlier results.
    pub fn merge(&mut self, newer: Self) -> Result<(), CompactionError> {
        if !self.is_current(&newer.source) {
            return Err(CompactionError::Stale);
        }
        let updated: HashSet<_> = newer.decisions.iter().map(Decision::id).collect();
        self.decisions
            .retain(|decision| !updated.contains(decision.id()));
        self.decisions.extend(newer.decisions);
        self.source = newer.source;
        Ok(())
    }

    /// Preserve calls and IDs, replacing dropped output with a small archive reference.
    pub fn apply(
        &self,
        current: &[ResponseItemEnvelope],
    ) -> Result<Vec<ResponseItemEnvelope>, CompactionError> {
        if !self.is_current(current) {
            return Err(CompactionError::Stale);
        }
        let decisions: HashMap<_, _> = self
            .decisions
            .iter()
            .map(|decision| (decision.id(), decision))
            .collect();
        let mut replacement = current.to_vec();
        for envelope in &mut replacement {
            let Some(id) = envelope.item.id().map(ToString::to_string) else {
                continue;
            };
            let Some(decision) = decisions.get(id.as_str()) else {
                continue;
            };
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
