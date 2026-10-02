use codex_history::LocalCompactionKind;
use codex_history::LocalCompactionSource;
use codex_history::ResponseItemEnvelope;
use serde::Deserialize;
use serde::Serialize;

use crate::CompactionError;
use crate::MAX_FRAGMENT_BYTES;
use crate::bounded_json;
use crate::groups::is_user_direction;
use crate::groups::pinned;
use crate::groups::safe_cuts;

#[derive(Debug, Clone, Serialize)]
pub struct SourceRange {
    pub first_item_id: String,
    pub last_item_id: String,
}

/// One bounded semantic response. The adapter supplies ranges; the model cannot invent them.
#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SummaryOutput {
    pub l2: String,
    pub l3: String,
    pub ledger: String,
}

#[derive(Debug)]
pub struct SummaryFragment {
    pub text: String,
    pub source: LocalCompactionSource,
}

/// Chronological L3 overview, L2 condensed dialogue, L1 cleaned verbatim dialogue, recent originals.
/// Existing summaries and the ledger are consumed on promotion, never accumulated per turn.
#[derive(Debug, Clone, Serialize)]
pub struct TierPlan {
    pub l2: Option<SourceRange>,
    pub l3: Option<SourceRange>,
    pub max_fragment_bytes: usize,
    /// History-only budget; the adapter reserves fixed context and continuation headroom.
    pub result_budget_tokens: usize,
    pub hard_cap_tokens: usize,
    /// Conservative bound using one token per summary byte, including provenance wrappers.
    pub projected_tokens: usize,
    pub retained_tokens: usize,
    #[serde(skip)]
    pub retained_prefix: Vec<ResponseItemEnvelope>,
    #[serde(skip)]
    pub retained_tail: Vec<ResponseItemEnvelope>,
    #[serde(skip)]
    ledger_range: Option<SourceRange>,
}

impl TierPlan {
    pub fn new(
        items: &[ResponseItemEnvelope],
        preferred_tokens: usize,
        hard_cap_tokens: usize,
        item_tokens: &[usize],
    ) -> Result<Self, CompactionError> {
        if item_tokens.len() != items.len() {
            return Err(CompactionError::Invalid(
                "history token costs do not match records",
            ));
        }
        let preferred_tokens = preferred_tokens.min(hard_cap_tokens);
        let retained_tokens = item_tokens
            .iter()
            .fold(0usize, |total, cost| total.saturating_add(*cost));
        let unchanged = || Self {
            l2: None,
            l3: None,
            max_fragment_bytes: 0,
            result_budget_tokens: preferred_tokens.max(retained_tokens),
            hard_cap_tokens,
            projected_tokens: retained_tokens,
            retained_tokens,
            retained_prefix: Vec::new(),
            retained_tail: items.to_vec(),
            ledger_range: None,
        };
        if retained_tokens <= preferred_tokens {
            return Ok(unchanged());
        }
        let cuts = safe_cuts(items);
        let l1_budget = (preferred_tokens / 3).min(6_000);
        let old_summary_end = items
            .iter()
            .rposition(is_summary)
            .map_or(0, |index| index + 1);
        let active_user = items
            .iter()
            .rposition(|item| !is_summary(item) && !pinned(item) && is_user_direction(&item.item));
        let keep: Vec<_> = items
            .iter()
            .enumerate()
            .map(|(index, item)| pinned(item) || active_user == Some(index))
            .collect();
        // The last safe cut is before unresolved calls. Completed groups, including the newest
        // one in a long user turn, are eligible for promotion as a whole.
        let pending_start = cuts.last().copied().unwrap_or(0);
        let mut suffix = vec![0usize; items.len() + 1];
        let mut fixed = 0usize;
        for index in (0..items.len()).rev() {
            suffix[index] =
                suffix[index + 1].saturating_add(if keep[index] { 0 } else { item_tokens[index] });
            if keep[index] {
                fixed = fixed.saturating_add(item_tokens[index]);
            }
        }
        // Exhaust promotions at the preferred occupancy before accepting a higher valid result.
        // A byte is a conservative token bound, unlike an assumed prose compression ratio.
        const MIN_SUMMARY_BYTES: usize = 128;
        for (budget, minimum_bytes) in [(preferred_tokens, MIN_SUMMARY_BYTES), (hard_cap_tokens, 1)]
        {
            for cut in cuts.iter().copied().filter(|cut| *cut >= old_summary_end) {
                let recent_tokens = suffix[cut].saturating_sub(suffix[pending_start]);
                if recent_tokens > l1_budget {
                    continue;
                }
                let retained_tokens = fixed.saturating_add(suffix[cut]);
                if retained_tokens >= budget {
                    continue;
                }
                let Some(mut plan) = Self::at_cut(items, &keep, &cuts, cut, old_summary_end) else {
                    continue;
                };
                let ranges: Vec<_> = plan
                    .l2
                    .iter()
                    .chain(plan.l3.iter())
                    .chain(plan.ledger_range.iter())
                    .collect();
                // LocalCompactionFragment adds markers, a kind name, provenance, recall guidance,
                // and message framing. Reserve their bytes plus the actual endpoint lengths.
                let overhead = ranges.iter().fold(0usize, |total, range| {
                    total
                        .saturating_add(256)
                        .saturating_add(range.first_item_id.len())
                        .saturating_add(range.last_item_id.len())
                });
                let available = budget
                    .saturating_sub(retained_tokens)
                    .saturating_sub(overhead);
                let fragment_bytes = (available / ranges.len()).min(MAX_FRAGMENT_BYTES);
                if fragment_bytes < minimum_bytes {
                    continue;
                }
                plan.max_fragment_bytes = fragment_bytes;
                plan.retained_tokens = retained_tokens;
                plan.projected_tokens = retained_tokens + overhead + fragment_bytes * ranges.len();
                plan.result_budget_tokens = preferred_tokens.max(plan.projected_tokens);
                plan.hard_cap_tokens = hard_cap_tokens;
                return Ok(plan);
            }
        }
        if retained_tokens <= hard_cap_tokens {
            return Ok(unchanged());
        }
        Err(CompactionError::Invalid(
            "irreducible active input exceeds the safe history budget",
        ))
    }

    fn at_cut(
        items: &[ResponseItemEnvelope],
        keep: &[bool],
        cuts: &[usize],
        l1: usize,
        old_summary_end: usize,
    ) -> Option<Self> {
        let midpoint = l1 / 2;
        let l3_end = cuts
            .iter()
            .copied()
            .filter(|cut| {
                *cut <= l1 && *cut >= old_summary_end && *cut <= midpoint.max(old_summary_end)
            })
            .max()
            .unwrap_or(l1);
        let summarized: Vec<_> = items[..l1]
            .iter()
            .enumerate()
            .filter(|(index, _)| !keep[*index])
            .map(|(_, item)| item.clone())
            .collect();
        let l3_items: Vec<_> = items[..l3_end]
            .iter()
            .enumerate()
            .filter(|(index, _)| !keep[*index])
            .map(|(_, item)| item.clone())
            .collect();
        let l2_items: Vec<_> = items[l3_end..l1]
            .iter()
            .enumerate()
            .filter(|(index, _)| !keep[*index + l3_end])
            .map(|(_, item)| item.clone())
            .collect();
        let l3 = source_range(&l3_items);
        let l2 = source_range(&l2_items);
        let ledger_range = source_range(&summarized)?;
        Some(Self {
            l2,
            l3,
            max_fragment_bytes: 0,
            result_budget_tokens: 0,
            hard_cap_tokens: 0,
            projected_tokens: 0,
            retained_tokens: 0,
            retained_prefix: items[..l1]
                .iter()
                .enumerate()
                .filter(|(index, _)| keep[*index])
                .map(|(_, item)| item.clone())
                .collect(),
            retained_tail: items[l1..].to_vec(),
            ledger_range: Some(ledger_range),
        })
    }

    pub fn parse(&self, json: &str) -> Result<Vec<SummaryFragment>, CompactionError> {
        let output: SummaryOutput = bounded_json(json)?;
        let mut fragments = Vec::new();
        for (text, range, kind) in [
            (
                output.l3,
                self.l3.as_ref(),
                LocalCompactionKind::OldestOverview,
            ),
            (
                output.l2,
                self.l2.as_ref(),
                LocalCompactionKind::CondensedDialogue,
            ),
            (
                output.ledger,
                self.ledger_range.as_ref(),
                LocalCompactionKind::ConstraintsLedger,
            ),
        ] {
            if text.len() > self.max_fragment_bytes
                || (range.is_some()
                    && kind != LocalCompactionKind::ConstraintsLedger
                    && text.trim().is_empty())
                || (range.is_none() && !text.is_empty())
            {
                return Err(CompactionError::Invalid(
                    "summary violates required range or size bound",
                ));
            }
            if let Some(range) = range {
                fragments.push(SummaryFragment {
                    text,
                    source: LocalCompactionSource {
                        first_item_id: range.first_item_id.clone(),
                        last_item_id: range.last_item_id.clone(),
                        kind,
                    },
                });
            }
        }
        Ok(fragments)
    }
}

#[cfg(test)]
#[path = "tiers_tests.rs"]
mod tests;

fn is_summary(item: &ResponseItemEnvelope) -> bool {
    item.metadata
        .as_ref()
        .and_then(|metadata| metadata.local_compaction.as_ref())
        .is_some_and(|source| source.kind != LocalCompactionKind::ToolResult)
}

fn source_range(items: &[ResponseItemEnvelope]) -> Option<SourceRange> {
    let mut sources = items
        .iter()
        .filter(|item| !pinned(item))
        // Range endpoints must name records whose source labels can be shown to the model.
        .filter(|item| {
            matches!(
                item.item,
                codex_protocol::models::ResponseItem::Message { .. }
                    | codex_protocol::models::ResponseItem::AgentMessage { .. }
                    | codex_protocol::models::ResponseItem::FunctionCall { .. }
                    | codex_protocol::models::ResponseItem::FunctionCallOutput { .. }
                    | codex_protocol::models::ResponseItem::CustomToolCall { .. }
                    | codex_protocol::models::ResponseItem::CustomToolCallOutput { .. }
                    | codex_protocol::models::ResponseItem::LocalShellCall {
                        call_id: Some(_),
                        ..
                    }
            )
        })
        .filter_map(|item| {
            if let Some(source) = item
                .metadata
                .as_ref()
                .and_then(|metadata| metadata.local_compaction.as_ref())
            {
                if source.kind == LocalCompactionKind::ConstraintsLedger {
                    return None;
                }
                Some((source.first_item_id.clone(), source.last_item_id.clone()))
            } else {
                item.item.id().map(|id| (id.to_string(), id.to_string()))
            }
        });
    let (first_item_id, mut last_item_id) = sources.next()?;
    for (_, last) in sources {
        last_item_id = last;
    }
    Some(SourceRange {
        first_item_id,
        last_item_id,
    })
}
