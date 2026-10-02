//! Local compaction policy and validation, independent of model request orchestration.

mod cleanup;
mod groups;
mod tiers;

pub use cleanup::Decision;
pub use cleanup::StagedDecisions;
pub use cleanup::eligible_results;
pub use groups::is_user_direction;
pub use tiers::SummaryOutput;
pub use tiers::TierPlan;

/// A fragment is at most 3,000 UTF-8 bytes (also an upper bound on token count).
pub const MAX_FRAGMENT_BYTES: usize = 3_000;
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
    pub trigger_percent: usize,
    pub target_percent: usize,
    pub minimum_savings_percent: usize,
}

impl Budget {
    pub fn should_analyze(self, history_tokens: usize) -> bool {
        self.fixed_tokens.saturating_add(history_tokens)
            >= self.window_tokens.saturating_mul(self.trigger_percent) / 100
    }

    pub fn history_target(self) -> usize {
        (self.window_tokens.saturating_mul(self.target_percent) / 100)
            .saturating_sub(self.fixed_tokens)
    }

    pub fn useful(self, before_tokens: usize, after_tokens: usize) -> bool {
        let saved = before_tokens.saturating_sub(after_tokens);
        saved > 0
            && saved
                >= self
                    .window_tokens
                    .saturating_mul(self.minimum_savings_percent)
                    / 100
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
