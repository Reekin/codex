use std::collections::HashMap;
use std::ops::Range;

use codex_history::LocalCompactionKind;
use codex_history::ResponseItemEnvelope;
use codex_protocol::models::ResponseItem;

use crate::CompactionError;
use crate::groups::call_key;
use crate::groups::client_context;
use crate::groups::output_key;
use crate::groups::protected_start;
use crate::groups::rendered;
use crate::groups::repeated_client_context;
use crate::groups::safe_cuts;
use crate::groups::user_input;

/// First and last original records a summary covers.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SourceRange {
    pub first_item_id: String,
    pub last_item_id: String,
}

/// Full compaction keeps the current window (L1) and summarizes earlier summaries and the
/// previous window (L2) into one handoff summary (L3). The kept window becomes the previous
/// window of the next full compaction, so repeated compaction converges.
///
/// The plan works on the cleaned history: records cleanup dropped are absent and low-value
/// tool work is already one record per run. Codex renders its instructions afresh ahead of
/// the plan, so rendered records ahead of the window are neither kept nor summarized; messages
/// a client injected cannot be rendered again and are kept once.
///
/// When the window does not fit, earlier user turns give way in value order, each layer
/// oldest first and stopping as soon as the rest fits: their remaining tool records, then
/// their intermediate assistant messages, then whole turns, the user's message and final reply
/// included. The current user turn gives way last, keeping its input and newest work group.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct WindowPlan {
    /// Records left out of the new history.
    pub removed: Vec<bool>,
    /// The current window's first record. Kept records from here on stay in place.
    pub window_start: usize,
    /// Client-injected context ahead of the window, kept ahead of the summary.
    pub kept_context: Vec<usize>,
    /// The active user input when it lies ahead of the window, kept after the summary.
    pub kept_input: Option<usize>,
    /// Original range of the summarized records; `None` when nothing needs a summary.
    pub summarized: Option<SourceRange>,
    /// History tokens after compaction, reserving `summary_tokens` for a summary.
    pub projected_tokens: usize,
}

impl WindowPlan {
    /// `costs` are per-record tokens of the cleaned history.
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
        let boundary = items
            .iter()
            .rposition(|item| is_summary(item) || previous_window(item))
            .map_or(0, |index| index + 1);
        let window_start = safe_cuts(items)
            .into_iter()
            .filter(|cut| *cut <= boundary)
            .max()
            .unwrap_or(0);
        let active_input = items.iter().rposition(|item| user_input(&item.item));
        let kept_input = active_input.filter(|index| *index < window_start);
        let mut plan = Planner {
            items,
            costs,
            removed: vec![false; items.len()],
            carried: vec![false; items.len()],
            kept_tokens: costs.iter().sum(),
            summary: false,
            budget_tokens,
            summary_tokens,
            outputs: items
                .iter()
                .enumerate()
                .filter_map(|(index, item)| output_key(&item.item).map(|key| (key, index)))
                .collect(),
        };
        for index in repeated_client_context(items) {
            plan.drop_copy(index);
        }
        let mut kept_context = Vec::new();
        for index in 0..window_start {
            if plan.removed[index] {
                continue;
            }
            if client_context(&items[index].item) {
                kept_context.push(index);
            } else if kept_input != Some(index) {
                plan.remove(index);
            }
        }
        let current = active_input
            .filter(|index| *index >= window_start)
            .unwrap_or(window_start);
        let mut starts = vec![window_start];
        starts.extend((window_start + 1..current).filter(|index| user_input(&items[*index].item)));
        let earlier: Vec<Range<usize>> = starts
            .iter()
            .zip(starts.iter().skip(1).chain([&current]))
            .map(|(start, end)| *start..*end)
            .filter(|turn| !turn.is_empty())
            .collect();
        let newest = protected_start(items).max(current);
        let body = if items
            .get(current)
            .is_some_and(|item| user_input(&item.item))
        {
            current + 1
        } else {
            current
        };
        plan.give_way(&earlier, current, body..newest);
        let summarized = if plan.summary {
            source_range(
                items
                    .iter()
                    .zip(&plan.carried)
                    .filter(|(_, carried)| **carried)
                    .map(|(item, _)| item),
            )
        } else {
            None
        };
        Ok(Self {
            projected_tokens: plan.total(),
            removed: plan.removed,
            window_start,
            kept_context,
            kept_input,
            summarized,
        })
    }
}

struct Planner<'a> {
    items: &'a [ResponseItemEnvelope],
    costs: &'a [usize],
    removed: Vec<bool>,
    /// Removed records whose content the summary carries.
    carried: Vec<bool>,
    kept_tokens: usize,
    /// Whether any removed record needs the summary.
    summary: bool,
    budget_tokens: usize,
    summary_tokens: usize,
    outputs: HashMap<(&'a str, &'a str), usize>,
}

impl Planner<'_> {
    fn total(&self) -> usize {
        self.kept_tokens
            .saturating_add(if self.summary { self.summary_tokens } else { 0 })
    }

    fn fits(&self) -> bool {
        self.total() <= self.budget_tokens
    }

    /// An older copy of client context that recurs later; nothing to summarize.
    fn drop_copy(&mut self, index: usize) {
        if !self.removed[index] {
            self.removed[index] = true;
            self.kept_tokens = self.kept_tokens.saturating_sub(self.costs[index]);
        }
    }

    fn remove(&mut self, index: usize) {
        if !self.removed[index] {
            self.drop_copy(index);
            self.carried[index] = summarized(&self.items[index]);
            self.summary |= self.carried[index];
        }
    }

    /// Earlier turns are the ones before `current`; `body` is the current turn after its
    /// input, up to its newest work group.
    fn give_way(&mut self, earlier: &[Range<usize>], current: usize, body: Range<usize>) {
        if self.fits() {
            return;
        }
        for turn in earlier {
            if self.tool_records(turn.clone(), current) {
                return;
            }
        }
        for turn in earlier {
            if self.interim_replies(turn.clone(), self.final_reply(turn)) {
                return;
            }
        }
        for turn in earlier {
            if self.whole(turn.clone(), current) {
                return;
            }
        }
        let _ = self.tool_records(body.clone(), body.end)
            || self.interim_replies(body.clone(), None)
            || self.whole(body.clone(), body.end);
    }

    /// Tool calls with their results and the reasoning that led into them, oldest first.
    fn tool_records(&mut self, turn: Range<usize>, limit: usize) -> bool {
        for index in turn {
            let item = &self.items[index].item;
            if !self.removed[index] && (call_key(item).is_some() || output_key(item).is_some()) {
                self.remove_with_result(index, limit);
                if self.fits() {
                    return true;
                }
            }
        }
        false
    }

    /// Assistant messages of a turn, oldest first, except its final reply.
    fn interim_replies(&mut self, turn: Range<usize>, last: Option<usize>) -> bool {
        for index in turn {
            if !self.removed[index] && assistant(&self.items[index].item) && Some(index) != last {
                self.remove(index);
                if self.fits() {
                    return true;
                }
            }
        }
        false
    }

    /// Everything left of a turn except client context, which Codex cannot render again.
    fn whole(&mut self, turn: Range<usize>, limit: usize) -> bool {
        let mut changed = false;
        for index in turn {
            if !self.removed[index] && !client_context(&self.items[index].item) {
                self.remove_with_result(index, limit);
                changed = true;
            }
        }
        changed && self.fits()
    }

    /// Removes a record; a call goes with its result unless that lies at or after `limit`, and
    /// reasoning that led into it goes too.
    fn remove_with_result(&mut self, index: usize, limit: usize) {
        let result =
            call_key(&self.items[index].item).and_then(|key| self.outputs.get(&key).copied());
        if result.is_some_and(|result| result >= limit) {
            return;
        }
        self.remove(index);
        if let Some(result) = result {
            self.remove(result);
        }
        let mut reasoning = index;
        while reasoning > 0
            && matches!(
                self.items[reasoning - 1].item,
                ResponseItem::Reasoning { .. }
            )
        {
            reasoning -= 1;
            self.remove(reasoning);
        }
    }

    /// The last assistant message before the next user input answers the turn.
    fn final_reply(&self, turn: &Range<usize>) -> Option<usize> {
        turn.clone()
            .rev()
            .find(|index| assistant(&self.items[*index].item))
    }
}

#[cfg(test)]
#[path = "tiers_tests.rs"]
mod tests;

fn assistant(item: &ResponseItem) -> bool {
    matches!(item, ResponseItem::Message { role, .. } if role == "assistant")
}

/// Content the new summary carries: anything but Codex's instructions, which are rendered
/// afresh. Earlier summaries and folded tool lines are carried too.
fn summarized(item: &ResponseItemEnvelope) -> bool {
    local_compaction(item).is_some() || !rendered(&item.item)
}

fn local_compaction(item: &ResponseItemEnvelope) -> Option<LocalCompactionKind> {
    item.metadata
        .as_ref()
        .and_then(|metadata| metadata.local_compaction.as_ref())
        .map(|source| source.kind)
        .filter(|kind| *kind != LocalCompactionKind::ToolResult)
}

fn is_summary(item: &ResponseItemEnvelope) -> bool {
    local_compaction(item).is_some_and(|kind| kind != LocalCompactionKind::ToolRun)
}

fn previous_window(item: &ResponseItemEnvelope) -> bool {
    item.metadata
        .as_ref()
        .is_some_and(|metadata| metadata.previous_window)
}

/// Earlier summaries and folded lines contribute the range they cover; other records their
/// own ID.
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
