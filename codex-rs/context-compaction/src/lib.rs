//! Local compaction policy and validation, independent of model request orchestration.

mod cleanup;
mod groups;
mod tiers;

pub use cleanup::Decision;
pub use cleanup::MAX_CALL_SUMMARY_BYTES;
pub use cleanup::StagedDecisions;
pub use cleanup::dropped_reasoning;
pub use cleanup::eligible_results;
pub use cleanup::large_calls;
pub use groups::is_user_direction;
pub use tiers::SourceRange;
pub use tiers::WindowPlan;

/// A shortened tool result is at most 3,000 UTF-8 bytes (also an upper bound on token count).
pub const MAX_FRAGMENT_BYTES: usize = 3_000;
/// The handoff summary of earlier windows; also an upper bound on its token count.
pub const MAX_SUMMARY_BYTES: usize = 16_000;
pub const MAX_ANALYSIS_BYTES: usize = 256_000;

#[derive(Debug, thiserror::Error)]
pub enum CompactionError {
    #[error("invalid local compaction JSON: {0}")]
    Json(#[from] serde_json::Error),
    #[error("invalid local compaction output: {0}")]
    Invalid(&'static str),
    #[error("local compaction source changed")]
    Stale,
}

/// Occupancy includes the fixed prompt budget, which is never reclaimable.
#[derive(Clone, Copy, Debug)]
pub struct Budget {
    pub window_tokens: usize,
    pub fixed_tokens: usize,
    pub reclaim_percent: usize,
    pub compact_target_percent: usize,
    pub keep_reasoning_percent: usize,
}

impl Budget {
    pub fn required_savings(self) -> usize {
        self.window_tokens
            .saturating_mul(self.reclaim_percent)
            .div_ceil(100)
            .max(1)
    }

    /// History allowed after full compaction before the current window's oldest part is
    /// summarized too.
    pub fn history_target(self) -> usize {
        (self
            .window_tokens
            .saturating_mul(self.compact_target_percent)
            / 100)
            .saturating_sub(self.fixed_tokens)
    }

    /// Newest reasoning kept by cleanup, for models that are sent earlier turns' reasoning.
    pub fn reasoning_tokens(self) -> usize {
        self.window_tokens
            .saturating_mul(self.keep_reasoning_percent)
            / 100
    }

    pub fn useful(self, before_tokens: usize, after_tokens: usize) -> bool {
        let saved = before_tokens.saturating_sub(after_tokens);
        saved >= self.required_savings()
    }

    /// An optimistic bound: all remaining growth could be removable tool output.
    /// Callers include protected-but-unmarked outputs that can become eligible later.
    pub fn can_reach(
        self,
        pending_savings: usize,
        unmarked_upper_bound: usize,
        current_total: usize,
        hard_limit: usize,
    ) -> bool {
        pending_savings
            .saturating_add(unmarked_upper_bound)
            .saturating_add(hard_limit.saturating_sub(current_total))
            >= self.required_savings()
    }
}

pub(crate) fn bounded_json<T: serde::de::DeserializeOwned>(
    text: &str,
) -> Result<T, CompactionError> {
    if text.len() > MAX_ANALYSIS_BYTES {
        return Err(CompactionError::Invalid("analysis exceeds the byte limit"));
    }
    Ok(serde_json::from_str(text)?)
}

#[cfg(test)]
#[path = "compaction_tests.rs"]
mod tests;
