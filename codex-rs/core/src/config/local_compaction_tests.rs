use super::Config;
use super::ConfigOverrides;
use super::LocalCompactionConfig;
use core_test_support::TempDirExt;
use pretty_assertions::assert_eq;
use tempfile::tempdir;

#[tokio::test]
async fn loads_local_compaction_overrides_and_partial_defaults() -> anyhow::Result<()> {
    for (text, expected) in [(
        "[local_compaction]\nreclaim_percent = 20\nmark_after_tokens_percent = 4\ncompact_target_percent = 25\nkeep_reasoning_percent = 8",
        LocalCompactionConfig {
            reclaim_percent: 20,
            mark_after_tokens_percent: 4,
            compact_target_percent: 25,
            keep_reasoning_percent: 8,
        },
    )] {
        let home = tempdir()?;
        let config = Config::load_from_base_config_with_overrides(
            toml::from_str(text)?,
            ConfigOverrides::default(),
            home.abs(),
        )
        .await?;
        assert_eq!(config.local_compaction, expected);
    }
    Ok(())
}

#[tokio::test]
async fn rejects_invalid_local_compaction_budgets() -> anyhow::Result<()> {
    for text in [
        "reclaim_percent = 0",
        "reclaim_percent = 100",
        "compact_target_percent = 0",
        "compact_target_percent = 100",
        "mark_after_tokens_percent = 0",
        "mark_after_tokens_percent = 100",
        "keep_reasoning_percent = 0",
        "keep_reasoning_percent = 100",
    ] {
        let home = tempdir()?;
        let result = Config::load_from_base_config_with_overrides(
            toml::from_str(&format!("[local_compaction]\n{text}"))?,
            ConfigOverrides::default(),
            home.abs(),
        )
        .await;
        let error = match result {
            Ok(_) => panic!("accepted invalid local_compaction: {text}"),
            Err(error) => error,
        };
        assert_eq!(error.kind(), std::io::ErrorKind::InvalidInput);
        assert!(error.to_string().contains("local_compaction"));
    }
    Ok(())
}
