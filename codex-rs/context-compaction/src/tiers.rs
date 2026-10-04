use codex_history::LocalCompactionKind;
use codex_history::ResponseItemEnvelope;
use codex_protocol::models::ResponseItem;

use crate::CompactionError;
use crate::groups::is_user_direction;
use crate::groups::pinned;
use crate::groups::safe_cuts;

/// First and last original records a summary covers.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SourceRange {
    pub first_item_id: String,
    pub last_item_id: String,
}

/// Full compaction keeps the current window (L1), after tool cleanup, verbatim; it becomes the
/// previous window (L2) of the next full compaction. Earlier summaries and the previous window
/// are summarized into one handoff summary (L3). When the cleaned current window alone exceeds
/// the budget, its oldest part is summarized as well, so repeated compaction always converges.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct WindowPlan {
    /// Records before this index are summarized, except the kept ones below.
    pub cut: usize,
    /// Canonical instructions before `cut`, kept verbatim ahead of the summary.
    pub kept_instructions: Vec<usize>,
    /// The active user input when it lies before `cut`, kept verbatim after the summary.
    pub kept_input: Option<usize>,
    /// Original range of the summarized records; `None` when nothing needs a summary.
    pub summarized: Option<SourceRange>,
    /// History tokens after compaction, reserving `summary_tokens` for a summary.
    pub projected_tokens: usize,
}

impl WindowPlan {
    /// `costs` are per-record tokens after tool cleanup, zero for records cleanup removes.
    /// Picks the earliest pair-safe cut at or after the current window start that fits
    /// `budget_tokens`, or the latest one when none fits.
    pub fn new(
        items: &[ResponseItemEnvelope],
        costs: &[usize],
        budget_tokens: usize,
        summary_tokens: usize,
    ) -> Result<Self, CompactionError> {
        if costs.len() != items.len() {
            return Err(CompactionError::Invalid(
                "history token costs do not match records",
            ));
        }
        let cuts = safe_cuts(items);
        let boundary = items
            .iter()
            .rposition(|item| is_summary(item) || previous_window(item))
            .map_or(0, |index| index + 1);
        let window_start = cuts
            .iter()
            .copied()
            .filter(|cut| *cut <= boundary)
            .max()
            .unwrap_or(0);
        let last_cut = cuts.last().copied().unwrap_or(0);
        let active_input = items
            .iter()
            .rposition(|item| !pinned(item) && !is_summary(item) && is_user_direction(&item.item));
        let mut suffix = vec![0usize; items.len() + 1];
        for index in (0..items.len()).rev() {
            suffix[index] = suffix[index + 1].saturating_add(costs[index]);
        }
        let mut latest = None;
        for cut in cuts
            .into_iter()
            .filter(|cut| (window_start..=last_cut).contains(cut))
        {
            let kept_instructions: Vec<_> =
                (0..cut).filter(|index| pinned(&items[*index])).collect();
            let kept_input = active_input.filter(|index| *index < cut);
            let summarized = source_range(
                items[..cut]
                    .iter()
                    .enumerate()
                    .filter(|(index, item)| !pinned(item) && kept_input != Some(*index))
                    .map(|(_, item)| item),
            );
            let projected_tokens = kept_instructions
                .iter()
                .chain(kept_input.iter())
                .map(|index| costs[*index])
                .fold(suffix[cut], usize::saturating_add)
                .saturating_add(if summarized.is_some() {
                    summary_tokens
                } else {
                    0
                });
            let plan = Self {
                cut,
                kept_instructions,
                kept_input,
                summarized,
                projected_tokens,
            };
            if plan.projected_tokens <= budget_tokens {
                return Ok(plan);
            }
            latest = Some(plan);
        }
        latest.ok_or(CompactionError::Invalid(
            "history has no safe compaction boundary",
        ))
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

fn previous_window(item: &ResponseItemEnvelope) -> bool {
    item.metadata
        .as_ref()
        .is_some_and(|metadata| metadata.previous_window)
}

/// Earlier summaries contribute the range they already cover; other records their own ID.
fn source_range<'a>(items: impl Iterator<Item = &'a ResponseItemEnvelope>) -> Option<SourceRange> {
    let mut sources = items.filter_map(|item| {
        if let Some(source) = item
            .metadata
            .as_ref()
            .and_then(|metadata| metadata.local_compaction.as_ref())
            .filter(|source| source.kind != LocalCompactionKind::ToolResult)
        {
            return Some((source.first_item_id.clone(), source.last_item_id.clone()));
        }
        if matches!(item.item, ResponseItem::Reasoning { .. }) {
            return None;
        }
        item.item.id().map(|id| (id.to_string(), id.to_string()))
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
