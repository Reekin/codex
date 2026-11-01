//! The private marking and summary questions, and the packing of marking batches.

use std::collections::HashSet;

use codex_history::ResponseItemEnvelope;
use codex_protocol::models::ResponseItem;
use serde::Serialize;
use serde_json::json;

use crate::CompactionError;
use crate::LocalCompactionRequest;
use crate::MAX_ANALYSIS_BYTES;
use crate::MAX_CALL_SUMMARY_BYTES;
use crate::MAX_FRAGMENT_BYTES;
use crate::MAX_SUMMARY_BYTES;
use crate::StagedDecisions;
use crate::eligible_results;
use crate::large_calls;
use crate::user_turn;

pub const CLASSIFY_MARKER: &str = "LOCAL_COMPACTION_CLASSIFY";
pub const SUMMARIZE_MARKER: &str = "LOCAL_COMPACTION_SUMMARIZE";

/// A completed tool result the model locates by the call ID it sees in the conversation.
#[derive(Debug, Clone, Serialize)]
struct Candidate {
    id: String,
    call_id: String,
    /// The paired call carries large arguments that shorten/drop replace with `call_text`.
    #[serde(skip_serializing_if = "std::ops::Not::not")]
    summarize_call: bool,
}

impl Candidate {
    fn new(item: &ResponseItem) -> Option<Self> {
        let (id, call_id) = match item {
            ResponseItem::FunctionCallOutput {
                id: Some(id),
                call_id: Some(call_id),
                ..
            }
            | ResponseItem::CustomToolCallOutput {
                id: Some(id),
                call_id,
                ..
            } => (id, call_id),
            _ => return None,
        };
        Some(Self {
            id: id.to_string(),
            call_id: call_id.clone(),
            summarize_call: false,
        })
    }
}

/// Drops the trailing candidates until the request fits its byte cap.
fn classifier(
    candidates: &mut Vec<Candidate>,
    guidance: &str,
) -> Result<LocalCompactionRequest, CompactionError> {
    loop {
        let request = LocalCompactionRequest::new(
            CLASSIFY_MARKER,
            json!({
                "candidates": candidates,
                "max_replacement_bytes": 2800,
                "max_call_text_bytes": MAX_CALL_SUMMARY_BYTES,
                "instructions": "Do not call tools and do not continue the task. Privately classify each candidate completed tool result above exactly once as keep, shorten, or drop. Find each result by its call_id in the conversation and answer with its id. Read current dialogue for relevance. Keep evidence needed for active work, unresolved questions, failures and verification. Shorten must preserve useful exact facts. Drop only dispensable output. Shorten and drop remove any images in the result; keep a result only if its images still need to be looked at, and state in shorten text what they showed. Originals stay retrievable. For candidates with summarize_call, shorten and drop also replace the call's arguments: give call_text, one line saying what the call did (for a patch, which files changed and how). Do not classify any other result. Return JSON only, with no prose or fences.",
                "required_output": {"decisions": [{"id": "candidate id", "action": "keep|shorten|drop", "text": "only for shorten", "call_text": "only with summarize_call, for shorten or drop"}]},
                "supplemental_guidance": guidance,
            }),
            LocalCompactionRequest::MAX_CLASSIFY_BYTES,
        );
        match request {
            Err(_) if candidates.len() > 1 => {
                candidates.pop();
            }
            result => return result,
        }
    }
}

/// One handoff summary of everything sent above it: earlier summaries and older records.
pub fn summary_request(guidance: &str) -> Result<LocalCompactionRequest, CompactionError> {
    LocalCompactionRequest::new(
        SUMMARIZE_MARKER,
        json!({
            "instructions": "Do not call tools and do not continue the task. You are performing a context checkpoint compaction: create a handoff summary of the whole conversation above, including any earlier summaries in local_compaction blocks, for another LLM that will resume the task. Include the user's goals and current progress, key decisions, important context, constraints and user preferences (including corrections), what remains to be done as clear next steps, and critical data, examples, file paths and references needed to continue. Say what was verified and what was not. Be concise and structured. Answer with the summary text only.",
            "max_bytes": MAX_SUMMARY_BYTES,
            "supplemental_guidance": guidance,
        }),
        LocalCompactionRequest::MAX_BYTES,
    )
}

/// What the next marking step should do.
#[derive(Debug)]
pub enum Batch {
    /// Nothing is sent: too little unmarked output to start, or a remainder too small to pay
    /// for a request on its own, which waits for the next start.
    Idle,
    /// Send `request` for the results in `ids`; `backlog` holds the rest of the current start.
    Send {
        ids: Vec<String>,
        request: LocalCompactionRequest,
        backlog: Option<HashSet<String>>,
    },
    /// The question cannot be built; the backlog stays for a later retry.
    Invalid {
        backlog: HashSet<String>,
        error: CompactionError,
    },
}

/// Settings that decide when marking starts and how a batch is packed.
#[derive(Debug, Clone, Copy)]
pub struct BatchLimits {
    /// Usable window in the same unit as the record costs.
    pub window_tokens: usize,
    /// Unmarked tool volume, as a window percentage, that starts working through the backlog.
    pub start_percent: usize,
}

impl BatchLimits {
    fn share(self, percent: usize) -> usize {
        self.window_tokens
            .saturating_mul(percent)
            .div_ceil(100)
            .max(1)
    }
}

/// Every batch resends the whole context, so marking starts once enough tool output is
/// unmarked, then works through the backlog present at that point, oldest first and one batch
/// at a time; output produced meanwhile waits for the next start. A batch adds results while
/// their worst-case replacements fit half the response limit and the candidate list fits its
/// request cap.
pub fn next_batch(
    source: &[ResponseItemEnvelope],
    staged: &StagedDecisions,
    backlog: Option<HashSet<String>>,
    limits: BatchLimits,
    guidance: &str,
    cost: impl Fn(&ResponseItem) -> usize,
) -> Batch {
    let eligible = eligible_results(source);
    let turn = user_turn(source);
    // A result's weight includes the large call it is shortened together with.
    let calls = large_calls(source);
    let weight = |item: &ResponseItemEnvelope| {
        cost(&item.item).saturating_add(
            item.item
                .id()
                .and_then(|id| calls.get(id.as_str()))
                .map_or(0, |&call| cost(&source[call].item)),
        )
    };
    // History order is oldest first.
    let unmarked: Vec<_> = source
        .iter()
        .filter(|item| {
            !staged.decided(item, turn.as_deref())
                && item
                    .item
                    .id()
                    .is_some_and(|id| eligible.iter().any(|eligible| eligible == id.as_str()))
        })
        .collect();
    let start = limits.share(limits.start_percent);
    let mut backlog: HashSet<String> = match backlog {
        Some(backlog) => backlog,
        None if unmarked.iter().map(|item| weight(item)).sum::<usize>() >= start => unmarked
            .iter()
            .filter_map(|item| item.item.id().map(ToString::to_string))
            .collect(),
        None => return Batch::Idle,
    };
    let mut response_bytes = 0_usize;
    let mut batch_tokens = 0_usize;
    let mut full = false;
    let mut candidates = Vec::new();
    for item in unmarked.iter().filter(|item| {
        item.item
            .id()
            .is_some_and(|id| backlog.contains(id.as_str()))
    }) {
        let Some(mut candidate) = Candidate::new(&item.item) else {
            continue;
        };
        candidate.summarize_call = calls.contains_key(&candidate.id);
        // A replacement is never larger than its original, so small outputs cost little
        // response space and many fit in one batch.
        let worst = cost(&item.item)
            .saturating_mul(4)
            .min(MAX_FRAGMENT_BYTES)
            .saturating_add(candidate.id.len() + 64)
            .saturating_add(if candidate.summarize_call {
                MAX_CALL_SUMMARY_BYTES
            } else {
                0
            });
        if response_bytes.saturating_add(worst) > MAX_ANALYSIS_BYTES / 2 {
            full = true;
            break;
        }
        response_bytes += worst;
        batch_tokens += weight(item);
        candidates.push(candidate);
    }
    let packed = candidates.len();
    let request = match classifier(&mut candidates, guidance) {
        Ok(request) => request,
        Err(error) => return Batch::Invalid { backlog, error },
    };
    full |= candidates.len() < packed;
    // A short remainder waits for the next backlog rather than paying for a request alone.
    if candidates.is_empty() || (!full && batch_tokens < start.min(limits.share(1))) {
        return Batch::Idle;
    }
    let ids: Vec<_> = candidates
        .into_iter()
        .map(|candidate| candidate.id)
        .collect();
    // Each result is sent once per backlog, even if its keep expires meanwhile.
    for id in &ids {
        backlog.remove(id);
    }
    Batch::Send {
        ids,
        request,
        backlog: (!backlog.is_empty()).then_some(backlog),
    }
}

#[cfg(test)]
#[path = "request_tests.rs"]
mod tests;
