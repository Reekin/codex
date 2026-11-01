use codex_history::ResponseItemEnvelope;
use codex_protocol::models::FunctionCallOutputBody;
use codex_protocol::models::ResponseItem;
use pretty_assertions::assert_eq;
use serde_json::json;

use super::append_marks;
use super::marks_path;
use super::restore_marks;
use crate::StagedDecisions;

fn history() -> Vec<ResponseItemEnvelope> {
    let item = |value| ResponseItemEnvelope::new(serde_json::from_value(value).unwrap());
    let mut items = vec![item(
        json!({"type":"message","id":"user_1","role":"user","content":[{"type":"input_text","text":"Investigate."}]}),
    )];
    for index in 0..2 {
        items.push(item(json!({"type":"function_call","id":format!("call_{index}"),"call_id":format!("c{index}"),"name":"exec","arguments":"{}"})));
        items.push(item(json!({"type":"function_call_output","id":format!("result_{index}"),"call_id":format!("c{index}"),"output":"evidence"})));
    }
    items.push(item(json!({"type":"function_call","id":"call_new","call_id":"c_new","name":"exec","arguments":"{}"})));
    items
}

fn decided(staged: &StagedDecisions) -> Vec<String> {
    staged
        .marks()
        .map(|(_, decision)| decision.id().to_string())
        .collect()
}

#[tokio::test]
async fn restore_keeps_unchanged_records_and_forks_copy_inherited_marks() {
    let home = tempfile::tempdir().unwrap();
    let parent = marks_path(home.path(), "parent");
    let child = marks_path(home.path(), "child");
    let source = history();
    let staged = StagedDecisions::parse(
        source.clone(),
        &json!({"decisions":[
            {"id":"result_0","action":"drop"},
            {"id":"result_1","action":"shorten","text":"Concise evidence."}
        ]})
        .to_string(),
    )
    .unwrap();
    append_marks(&parent, &staged).await;

    // A changed record loses its mark; the unchanged one is restored.
    let mut changed = source.clone();
    if let ResponseItem::FunctionCallOutput { output, .. } = &mut changed[4].item {
        output.body = FunctionCallOutputBody::Text("rewritten".to_string());
    }
    let restored = restore_marks(&parent, None, &changed).await;
    assert_eq!(decided(&restored), vec!["result_0"]);

    // A fork reads its parent's marks and keeps its own copy, so it no longer needs the parent.
    let inherited = restore_marks(&child, Some(&parent), &source).await;
    assert_eq!(decided(&inherited), vec!["result_0", "result_1"]);
    std::fs::remove_file(&parent).unwrap();
    let own = restore_marks(&child, None, &source).await;
    assert_eq!(decided(&own), vec!["result_0", "result_1"]);
}
