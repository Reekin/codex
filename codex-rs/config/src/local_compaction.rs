use schemars::JsonSchema;
use serde::Deserialize;
use serde::Serialize;

/// Occupancy and batching policy for local conversation cleanup.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(default, deny_unknown_fields)]
pub struct LocalCompactionConfig {
    /// Use local cleanup even when the provider supports remote compaction.
    pub force_local: bool,
    /// Percentage of the usable context window that starts automatic cleanup.
    pub trigger_percent: u8,
    /// Desired percentage of the usable context window after cleanup.
    pub target_percent: u8,
    /// Minimum savings as a percentage of the usable context window per cleanup.
    pub minimum_savings_percent: u8,
}

impl Default for LocalCompactionConfig {
    fn default() -> Self {
        Self {
            force_local: false,
            trigger_percent: 50,
            target_percent: 30,
            minimum_savings_percent: 5,
        }
    }
}

impl LocalCompactionConfig {
    pub fn validate(&self) -> std::io::Result<()> {
        if self.target_percent == 0
            || self.target_percent >= self.trigger_percent
            || self.trigger_percent > 100
        {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidInput,
                "local_compaction requires 0 < target_percent < trigger_percent <= 100",
            ));
        }
        if self.minimum_savings_percent == 0
            || self.minimum_savings_percent > self.trigger_percent - self.target_percent
        {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidInput,
                "local_compaction.minimum_savings_percent must be positive and no greater than trigger_percent - target_percent",
            ));
        }
        Ok(())
    }
}
