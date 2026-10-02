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
use crate::groups::protected_start;
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
    #[serde(skip)]
    pub retained_prefix: Vec<ResponseItemEnvelope>,
    #[serde(skip)]
    pub retained_tail: Vec<ResponseItemEnvelope>,
    #[serde(skip)]
    ledger_range: SourceRange,
}

impl TierPlan {
    pub fn new(
        items: &[ResponseItemEnvelope],
        target_tokens: usize,
        item_tokens: &[usize],
    ) -> Option<Self> {
        if item_tokens.len() != items.len() {
            return None;
        }
        let recent = protected_start(items);
        if recent == 0 {
            return None;
        }
        let cuts = safe_cuts(items);
        // L1 stays verbatim, within a fixed share of the target. Never keep an old summary here.
        let l1_budget = (target_tokens / 3).min(6_000);
        let mut l1 = recent;
        for cut in cuts.iter().rev().copied().filter(|cut| *cut < recent) {
            if item_tokens[cut..recent].iter().sum::<usize>() > l1_budget
                || items[cut..recent].iter().any(is_summary)
            {
                break;
            }
            l1 = cut;
        }
        if l1 == 0 {
            return None;
        }
        let old_summary_end = items[..l1]
            .iter()
            .rposition(is_summary)
            .map_or(0, |index| index + 1);
        let midpoint = l1 / 2;
        let l3_end = cuts
            .iter()
            .copied()
            .filter(|cut| {
                *cut <= l1 && *cut >= old_summary_end && *cut <= midpoint.max(old_summary_end)
            })
            .max()
            .unwrap_or(l1);
        let active_user = items.iter().rposition(|item| is_user_direction(&item.item));
        let is_retained =
            |index: usize, item: &ResponseItemEnvelope| pinned(item) || active_user == Some(index);
        let summarized: Vec<_> = items[..l1]
            .iter()
            .enumerate()
            .filter(|(index, item)| !is_retained(*index, item))
            .map(|(_, item)| item.clone())
            .collect();
        let l3_items: Vec<_> = items[..l3_end]
            .iter()
            .enumerate()
            .filter(|(index, item)| !is_retained(*index, item))
            .map(|(_, item)| item.clone())
            .collect();
        let l2_items: Vec<_> = items[l3_end..l1]
            .iter()
            .enumerate()
            .filter(|(index, item)| !is_retained(*index + l3_end, item))
            .map(|(_, item)| item.clone())
            .collect();
        let l3 = source_range(&l3_items);
        let l2 = source_range(&l2_items);
        let ledger_range = source_range(&summarized)?;
        let retained_tokens: usize = item_tokens
            .iter()
            .enumerate()
            .filter(|(index, _)| *index >= l1 || is_retained(*index, &items[*index]))
            .map(|(_, tokens)| *tokens)
            .sum();
        let fragment_count = usize::from(l2.is_some()) + usize::from(l3.is_some()) + 1;
        // Reserve space for provenance, fragment wrappers, and JSON overhead before text.
        let max_fragment_bytes = target_tokens
            .saturating_sub(retained_tokens)
            .saturating_sub(fragment_count * 180)
            .saturating_mul(4)
            / fragment_count;
        Some(Self {
            l2,
            l3,
            max_fragment_bytes: max_fragment_bytes.min(MAX_FRAGMENT_BYTES),
            retained_prefix: items[..l1]
                .iter()
                .enumerate()
                .filter(|(index, item)| is_retained(*index, item))
                .map(|(_, item)| item.clone())
                .collect(),
            retained_tail: items[l1..].to_vec(),
            ledger_range,
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
                Some(&self.ledger_range),
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
