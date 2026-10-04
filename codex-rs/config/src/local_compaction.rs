use schemars::JsonSchema;
use serde::Deserialize;
use serde::Serialize;

/// Background marking, reclaim thresholds, and full-compaction budgets.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(default, deny_unknown_fields)]
pub struct LocalCompactionConfig {
    /// Use local cleanup even when the provider supports remote compaction.
    pub force_local: bool,
    /// Window percentage points that completed tool marks must reclaim before cleanup.
    pub reclaim_percent: u8,
    /// Unmarked tool tokens as a percentage of the window that starts a background batch.
    pub mark_after_tokens_percent: u8,
    /// Occupancy full compaction keeps the cleaned current window within; beyond it the window's oldest part is summarized too.
    pub compact_target_percent: u8,
    /// Window percentage of newest earlier-turn reasoning kept when tool marks are applied.
    pub keep_reasoning_percent: u8,
}

impl Default for LocalCompactionConfig {
    fn default() -> Self {
        Self {
            force_local: false,
            reclaim_percent: 30,
            mark_after_tokens_percent: 5,
            compact_target_percent: 50,
            keep_reasoning_percent: 5,
        }
    }
}

impl LocalCompactionConfig {
    pub fn validate(&self) -> std::io::Result<()> {
        for (name, percent) in [
            ("reclaim_percent", self.reclaim_percent),
            ("mark_after_tokens_percent", self.mark_after_tokens_percent),
            ("compact_target_percent", self.compact_target_percent),
            ("keep_reasoning_percent", self.keep_reasoning_percent),
        ] {
            if !(1..100).contains(&percent) {
                return Err(std::io::Error::new(
                    std::io::ErrorKind::InvalidInput,
                    format!("local_compaction.{name} must be between 1 and 99"),
                ));
            }
        }
        Ok(())
    }
}
