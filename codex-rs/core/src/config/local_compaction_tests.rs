use super::Config;
use super::ConfigOverrides;
use super::LocalCompactionConfig;
use codex_features::Feature;
use core_test_support::TempDirExt;
use pretty_assertions::assert_eq;
use tempfile::tempdir;

#[tokio::test]
async fn loads_local_compaction_overrides_and_partial_defaults() -> anyhow::Result<()> {
    for (text, expected) in [
        (
            "[local_compaction]\nforce_local = true",
            LocalCompactionConfig {
                force_local: true,
                ..LocalCompactionConfig::default()
            },
        ),
        (
            "[local_compaction]\nreclaim_percent = 20\nmark_after_tokens_percent = 4\nmark_after_records = 16\ncompact_target_percent = 25",
            LocalCompactionConfig {
                force_local: false,
                reclaim_percent: 20,
                mark_after_tokens_percent: 4,
                mark_after_records: 16,
                compact_target_percent: 25,
            },
        ),
    ] {
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
        "mark_after_records = 0",
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

#[tokio::test]
async fn forced_local_startup_bypasses_configured_history_notes() -> anyhow::Result<()> {
    let home = tempdir()?;
    let mut config = Config::load_from_base_config_with_overrides(
        toml::from_str(
            "[local_compaction]\nforce_local = true\n[features.token_budget]\nenabled = true\nuse_history_notes_extension = true",
        )?,
        ConfigOverrides::default(),
        home.abs(),
    )
    .await?;
    config.prepare_token_budget_for_startup()?;
    assert_eq!(
        (
            config.features.enabled(Feature::TokenBudget),
            config.token_budget,
            config.token_budget_startup_config,
        ),
        (false, None, None),
    );
    Ok(())
}
