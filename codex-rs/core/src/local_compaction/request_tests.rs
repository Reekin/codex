use super::label_source_items;
use codex_protocol::models::ContentItem;
use codex_protocol::models::ResponseItem;
use pretty_assertions::assert_eq;
use serde_json::json;

#[test]
fn private_source_labels_bind_visible_evidence_to_result_and_call_ids() {
    let original: Vec<ResponseItem> = serde_json::from_value(json!([
        {"type":"message","id":"user_1","role":"user","content":[{"type":"input_text","text":"Keep exact evidence."}]},
        {"type":"function_call","id":"call_item_1","call_id":"call_1","name":"exec","arguments":"{}"},
        {"type":"function_call_output","id":"result_1","call_id":"call_1","output":"Permission denied; no change occurred."},
        {"type":"message","id":"reply_1","role":"assistant","content":[{"type":"output_text","text":"Verification remains pending."}]}
    ])).unwrap();
    let mut private = original.clone();
    label_source_items(&mut private).unwrap();
    assert_eq!(private[1], original[1]);
    let ResponseItem::FunctionCallOutput { output, .. } = &private[2] else {
        panic!("tool result")
    };
    let text = output.to_string();
    let (marker, body) = text.split_once('\n').unwrap();
    let (header, evidence) = body.split_once('\n').unwrap();
    assert_eq!(marker, "LOCAL_COMPACTION_SOURCE");
    assert_eq!(
        serde_json::from_str::<serde_json::Value>(header).unwrap(),
        json!({"item_id":"result_1","call_item_id":"call_item_1"})
    );
    assert_eq!(evidence, "Permission denied; no change occurred.");
    let ResponseItem::Message { content, .. } = &private[0] else {
        panic!("user message")
    };
    let ContentItem::InputText { text } = &content[0] else {
        panic!("text")
    };
    assert!(text.contains("user_1"));
    assert!(text.ends_with("Keep exact evidence."));
    let ResponseItem::FunctionCallOutput { output, .. } = &original[2] else {
        panic!("original result")
    };
    assert_eq!(output.to_string(), "Permission denied; no change occurred.");
}
