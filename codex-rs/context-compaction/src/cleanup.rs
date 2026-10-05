use std::collections::HashMap;
use std::collections::HashSet;

use codex_history::LocalCompactionKind;
use codex_history::LocalCompactionSource;
use codex_history::ResponseItemEnvelope;
use codex_protocol::models::FunctionCallOutputBody;
use codex_protocol::models::FunctionCallOutputContentItem;
use codex_protocol::models::FunctionCallOutputPayload;
use codex_protocol::models::ResponseItem;
use serde::Deserialize;
use serde::Serialize;
use serde_json::json;

use crate::CompactionError;
use crate::MAX_FRAGMENT_BYTES;
use crate::bounded_json;
use crate::groups::call_key;
use crate::groups::is_user_direction;
use crate::groups::output_key;
use crate::groups::protected_start;

/// Calls carrying at least this much argument text are shortened together with their result.
pub const LARGE_CALL_BYTES: usize = 1_000;
/// A call summary replaces the call's arguments; it is a single short line.
pub const MAX_CALL_SUMMARY_BYTES: usize = 400;

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "action", rename_all = "snake_case", deny_unknown_fields)]
pub enum Decision {
    Keep {
        id: String,
        /// Accepted and ignored: a kept result leaves its call unchanged.
        #[serde(default, skip_serializing)]
        call_text: Option<String>,
        /// Latest user input when the result was judged. Relevance changes as the task moves
        /// on, so the keep holds only until the user writes again; set by validation.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        user_turn: Option<String>,
    },
    Shorten {
        id: String,
        text: String,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        call_text: Option<String>,
    },
    Drop {
        id: String,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        call_text: Option<String>,
    },
}

impl Decision {
    pub fn id(&self) -> &str {
        match self {
            Self::Keep { id, .. } | Self::Shorten { id, .. } | Self::Drop { id, .. } => id,
        }
    }

    /// An empty summary means none was given; the call then gets a plain archive reference.
    fn call_text(&self) -> Option<&str> {
        match self {
            Self::Keep { .. } => None,
            Self::Shorten { call_text, .. } | Self::Drop { call_text, .. } => {
                call_text.as_deref().filter(|text| !text.trim().is_empty())
            }
        }
    }
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Analysis {
    /// Entries are validated one by one, so one malformed entry costs only itself.
    decisions: Vec<serde_json::Value>,
}

/// Cuts `text` to at most `max` bytes at a character boundary.
fn truncate_to(text: &mut String, max: usize) {
    if text.len() > max {
        let end = (0..=max)
            .rev()
            .find(|index| text.is_char_boundary(*index))
            .unwrap_or(0);
        text.truncate(end);
    }
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

/// ID of the latest user input, which scopes keep decisions.
pub fn user_turn(items: &[ResponseItemEnvelope]) -> Option<String> {
    items
        .iter()
        .rev()
        .find(|item| is_user_direction(&item.item))
        .and_then(|item| item.item.id())
        .map(ToString::to_string)
}

/// Marks reasoning records to remove: earlier turns' reasoning beyond the newest `keep_tokens`.
/// The current turn's reasoning always stays, since providers require it within a tool-call
/// sequence; it still counts toward the budget.
pub fn dropped_reasoning(
    items: &[ResponseItemEnvelope],
    keep_tokens: usize,
    costs: &[usize],
) -> Vec<bool> {
    let turn_start = items
        .iter()
        .rposition(|item| is_user_direction(&item.item))
        .unwrap_or(0);
    let mut kept = 0usize;
    let mut dropped = vec![false; items.len()];
    for index in (0..items.len()).rev() {
        if matches!(items[index].item, ResponseItem::Reasoning { .. }) {
            kept = kept.saturating_add(costs[index]);
            dropped[index] = index < turn_start && kept > keep_tokens;
        }
    }
    dropped
}

/// Maps each tool result ID to the index of its paired call when that call carries at least
/// `LARGE_CALL_BYTES` of arguments. Such calls are shortened together with their result.
pub fn large_calls(items: &[ResponseItemEnvelope]) -> HashMap<String, usize> {
    let mut calls = HashMap::new();
    let mut large = HashMap::new();
    for (index, envelope) in items.iter().enumerate() {
        if let Some(key) = call_key(&envelope.item)
            && call_arguments(&envelope.item).is_some_and(|text| text.len() >= LARGE_CALL_BYTES)
        {
            calls.insert(key, index);
        }
        if let Some(key) = output_key(&envelope.item)
            && let Some(call) = calls.remove(&key)
            && let Some(id) = envelope.item.id()
        {
            large.insert(id.to_string(), call);
        }
    }
    large
}

fn call_arguments(item: &ResponseItem) -> Option<&str> {
    match item {
        ResponseItem::FunctionCall { arguments, .. } => Some(arguments),
        ResponseItem::CustomToolCall { input, .. } => Some(input),
        _ => None,
    }
}

/// Replaces call arguments in place, keeping the call's type, name and pairing ID.
fn shorten_call(item: &mut ResponseItem, summary: Option<&str>) -> bool {
    let (id, arguments, function) = match item {
        ResponseItem::FunctionCall {
            id: Some(id),
            arguments,
            ..
        } => (id.to_string(), arguments, true),
        ResponseItem::CustomToolCall {
            id: Some(id),
            input,
            ..
        } => (id.to_string(), input, false),
        _ => return false,
    };
    let text = match summary {
        Some(summary) => format!("{summary}\n[Original item: {id}; use recall_read_item.]"),
        None => format!("[Arguments omitted; original item: {id}; use recall_read_item.]"),
    };
    // Providers parse function arguments as a JSON object; free-form input stays plain text.
    let text = if function {
        json!({ "summary": text }).to_string()
    } else {
        text
    };
    if text.len() >= arguments.len() {
        return false;
    }
    *arguments = text;
    true
}

fn image_count(output: &FunctionCallOutputPayload) -> usize {
    output.content_items().map_or(0, |parts| {
        parts
            .iter()
            .filter(|part| matches!(part, FunctionCallOutputContentItem::InputImage { .. }))
            .count()
    })
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
                if output.content_items().is_none_or(|parts| parts.iter().all(|part| matches!(part, FunctionCallOutputContentItem::InputText { .. } | FunctionCallOutputContentItem::InputImage { .. }))))
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

    /// Whether the result still has a valid decision: a shorten or drop, or a keep made within
    /// the current user turn.
    pub fn decided(&self, item: &ResponseItemEnvelope, current_turn: Option<&str>) -> bool {
        self.contains(item)
            && self.decisions.iter().any(|decision| {
                item.item
                    .id()
                    .is_some_and(|id| id.as_str() == decision.id())
                    && match decision {
                        Decision::Keep { user_turn, .. } => {
                            user_turn.is_some() && user_turn.as_deref() == current_turn
                        }
                        Decision::Shorten { .. } | Decision::Drop { .. } => true,
                    }
            })
    }

    /// Validated decisions paired with the exact records they were made for.
    pub fn marks(&self) -> impl Iterator<Item = (&ResponseItemEnvelope, &Decision)> {
        self.decisions.iter().filter_map(|decision| {
            self.source
                .iter()
                .find(|item| {
                    item.item
                        .id()
                        .is_some_and(|id| id.as_str() == decision.id())
                })
                .map(|item| (item, decision))
        })
    }

    /// Rebuilds durable decisions for records the caller verified as unchanged.
    pub fn from_marks(marks: Vec<(ResponseItemEnvelope, Decision)>) -> Self {
        let (source, decisions) = marks.into_iter().unzip();
        Self { source, decisions }
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

    /// Keeps every usable decision of a response. Unparseable JSON rejects the batch; a decision
    /// for a foreign or repeated ID is ignored, an oversized text is cut to its limit, an empty
    /// shortening becomes a drop, and a result without a decision stays unmarked.
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
        let mut decisions: Vec<Decision> = Vec::new();
        let turn = user_turn(&source);
        for entry in analysis.decisions {
            let Ok(decision) = serde_json::from_value::<Decision>(entry) else {
                continue;
            };
            if !allowed.contains(decision.id())
                || decisions.iter().any(|kept| kept.id() == decision.id())
            {
                continue;
            }
            let mut decision = match decision {
                Decision::Shorten {
                    id,
                    call_text,
                    text,
                } if text.trim().is_empty() => Decision::Drop { id, call_text },
                Decision::Shorten {
                    id,
                    mut text,
                    call_text,
                } => {
                    truncate_to(&mut text, MAX_FRAGMENT_BYTES.saturating_sub(id.len() + 64));
                    Decision::Shorten {
                        id,
                        text,
                        call_text,
                    }
                }
                Decision::Keep { id, call_text, .. } => Decision::Keep {
                    id,
                    call_text,
                    user_turn: turn.clone(),
                },
                drop @ Decision::Drop { .. } => drop,
            };
            if let Decision::Shorten { call_text, .. } | Decision::Drop { call_text, .. } =
                &mut decision
                && let Some(text) = call_text
            {
                truncate_to(text, MAX_CALL_SUMMARY_BYTES);
            }
            decisions.push(decision);
        }
        if decisions.is_empty() {
            return Err(CompactionError::Invalid("no usable decision"));
        }
        let decided: HashSet<_> = decisions.iter().map(Decision::id).collect();
        Ok(Self {
            source: source
                .into_iter()
                .filter(|item| {
                    item.item
                        .id()
                        .is_some_and(|id| decided.contains(id.as_str()))
                })
                .collect(),
            decisions,
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

    /// Preserve call/result pairing and IDs. Shortened and dropped results lose their images and
    /// bulky text; a large paired call keeps its record but its arguments become a summary.
    pub fn apply(
        &self,
        current: &[ResponseItemEnvelope],
    ) -> Result<Vec<ResponseItemEnvelope>, CompactionError> {
        let eligible: HashSet<_> = eligible_results(current).into_iter().collect();
        let calls = large_calls(current);
        let decisions: HashMap<_, _> = self
            .decisions
            .iter()
            .map(|decision| (decision.id(), decision))
            .collect();
        let mut replacement = current.to_vec();
        for index in 0..replacement.len() {
            if !self.contains(&replacement[index]) {
                continue;
            }
            let Some(id) = replacement[index].item.id().map(ToString::to_string) else {
                continue;
            };
            let Some(decision) = decisions.get(id.as_str()) else {
                continue;
            };
            if !eligible.contains(&id) {
                continue;
            }
            let summary = match decision {
                Decision::Keep { .. } => continue,
                Decision::Shorten { text, .. } => Some(text),
                Decision::Drop { .. } => None,
            };
            let mut changed = match &mut replacement[index].item {
                ResponseItem::FunctionCallOutput { output, .. }
                | ResponseItem::CustomToolCallOutput { output, .. } => {
                    let images = image_count(output);
                    let text = match (summary, images) {
                        (Some(text), 0) => {
                            format!("{text}\n[Original item: {id}; use recall_read_item.]")
                        }
                        (Some(text), images) => format!(
                            "{text}\n[{images} images omitted. Original item: {id}; use recall_read_item.]"
                        ),
                        (None, 0) => {
                            format!("[Output omitted; original item: {id}; use recall_read_item.]")
                        }
                        (None, images) => format!(
                            "[Output omitted with {images} images; original item: {id}; use recall_read_item.]"
                        ),
                    };
                    // Images always go; otherwise replace only when the text is shorter.
                    let replace = images > 0 || text.len() < output.to_string().len();
                    if replace {
                        output.body = FunctionCallOutputBody::Text(text);
                    }
                    replace
                }
                _ => {
                    return Err(CompactionError::Invalid(
                        "decision target is no longer a tool result",
                    ));
                }
            };
            if let Some(&call) = calls.get(&id) {
                changed |= shorten_call(&mut replacement[call].item, decision.call_text());
            }
            if changed {
                replacement[index]
                    .metadata
                    .get_or_insert_default()
                    .local_compaction = Some(LocalCompactionSource {
                    first_item_id: id.clone(),
                    last_item_id: id,
                    kind: LocalCompactionKind::ToolResult,
                });
            }
        }
        Ok(replacement)
    }
}
