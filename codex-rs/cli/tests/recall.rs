use codex_protocol::protocol::SessionMeta;
use codex_protocol::protocol::SessionMetaLine;
use codex_rollout::RolloutItem;
use codex_rollout::RolloutLine;
use pretty_assertions::assert_eq;
use serde_json::Value;
use serde_json::json;

#[test]
fn recall_cli_reads_original_item_in_bounded_pages() -> anyhow::Result<()> {
    let dir = tempfile::tempdir()?;
    let path = dir.path().join("original.jsonl");
    let original = json!({"type":"message","id":"msg_original","role":"assistant","content":[{"type":"output_text","text":"original evidence"}]});
    let items = [
        RolloutItem::SessionMeta(SessionMetaLine {
            meta: SessionMeta::default(),
            git: None,
        }),
        RolloutItem::ResponseItem(
            serde_json::from_value::<codex_protocol::models::ResponseItem>(original.clone())?
                .into(),
        ),
    ];
    let records = items
        .into_iter()
        .enumerate()
        .map(|(index, item)| {
            serde_json::to_string(&RolloutLine {
                timestamp: "2026-10-02T00:00:00Z".into(),
                ordinal: Some(index as u64),
                item,
            })
        })
        .collect::<Result<Vec<_>, _>>()?
        .join("\n")
        + "\n";
    std::fs::write(&path, records)?;
    let mut assembled = String::new();
    let mut start = 0;
    loop {
        let output = assert_cmd::Command::new(codex_utils_cargo_bin::cargo_bin("codex")?)
            .arg("recall").arg("--rollout").arg(&path)
            .arg("--query").arg(json!({"action":"read_item","item_id":"msg_original","start_char":start,"max_chars":17}).to_string())
            .assert().success().get_output().stdout.clone();
        let page: Value = serde_json::from_slice(&output)?;
        assembled.push_str(page["text"].as_str().expect("text"));
        let Some(next) = page["next_char"].as_u64() else {
            break;
        };
        start = next;
    }
    let recalled: Value = serde_json::from_str(&assembled)?;
    assert_eq!(recalled, original);
    Ok(())
}
