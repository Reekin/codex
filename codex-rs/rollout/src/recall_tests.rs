use super::*;
use crate::RolloutLine;
use codex_protocol::ThreadId;
use codex_protocol::protocol::HistoryPosition;
use codex_protocol::protocol::SessionMeta;
use codex_protocol::protocol::SessionMetaLine;
use pretty_assertions::assert_eq;

fn response(id: &str, text: &str) -> RolloutItem {
    RolloutItem::ResponseItem(
        serde_json::from_value::<codex_protocol::models::ResponseItem>(json!({
            "type": "message", "id": id, "role": "assistant",
            "content": [{"type": "output_text", "text": text}]
        }))
        .expect("response")
        .into(),
    )
}

fn read(item_id: &str, start_char: usize, max_chars: usize) -> RecallQuery {
    RecallQuery::ReadItem {
        item_id: item_id.into(),
        start_char,
        max_chars,
    }
}

#[test]
fn first_original_wins_and_unicode_pages_reassemble() {
    let mut archive = RecallArchive::default();
    archive
        .record(response("msg_original", "原始🙂evidence"), "1")
        .unwrap();
    archive
        .record(response("msg_original", "replacement"), "2")
        .unwrap();
    let original = archive.query(read("msg_original", 0, 8000)).unwrap();
    let mut offset = 0;
    let mut assembled = String::new();
    loop {
        let page = archive.query(read("msg_original", offset, 3)).unwrap();
        assembled.push_str(page["text"].as_str().unwrap());
        let Some(next) = page["next_char"].as_u64() else {
            break;
        };
        offset = next as usize;
    }
    assert_eq!(assembled, original["text"].as_str().unwrap());
    let content: Value = serde_json::from_str(&assembled).unwrap();
    assert_eq!(
        content["content"],
        json!([{"type":"output_text", "text":"原始🙂evidence"}])
    );
    assert_eq!(
        archive.query(read("msg_original", usize::MAX, 1)).unwrap()["next_char"],
        Value::Null
    );
    assert!(archive.query(read("missing", 0, 10)).is_err());
}

#[test]
fn event_turn_fallback_and_inter_agent_dialogue_are_queryable() {
    let mut archive = RecallArchive::default();
    let event = serde_json::from_value(json!({"type":"event_msg","payload":{
        "type":"task_started", "turn_id":"turn_event", "model_context_window":null
    }}))
    .unwrap();
    archive.record(event, "event").unwrap();
    let message = serde_json::from_value::<codex_protocol::models::ResponseItem>(json!({
        "type":"agent_message", "id":"amsg_1", "author":"worker", "recipient":"parent",
        "content":[{"type":"input_text","text":"independent result"}]
    }))
    .unwrap();
    archive
        .record(RolloutItem::ResponseItem(message.into()), "message")
        .unwrap();
    let result = archive
        .query(RecallQuery::ReadTurn {
            turn_id: "turn_event".into(),
            detail: RecallDetail::Dialogue,
            offset: 0,
            limit: 10,
        })
        .unwrap();
    assert_eq!(
        (
            result["total"].clone(),
            result["data"][0]["item_id"].clone()
        ),
        (json!(1), json!("amsg_1"))
    );
}

#[test]
fn serialized_pages_bound_escaping_without_losing_offsets() {
    let mut archive = RecallArchive::default();
    for index in 0..12 {
        archive
            .record(
                response(&format!("msg_{index}"), &"🙂\u{0000}\\\"".repeat(3000)),
                &index.to_string(),
            )
            .unwrap();
    }
    let page = archive.query(read("msg_0", 0, 8000)).unwrap();
    assert!(page.to_string().len() <= MAX_PAGE_BYTES);
    let next = page["next_char"].as_u64().unwrap() as usize;
    assert_eq!(next, page["text"].as_str().unwrap().chars().count());
    let continuation = archive.query(read("msg_0", next, 8000)).unwrap();
    assert_eq!(
        continuation["text"].as_str().unwrap(),
        slice_chars(
            &archive.items[0].text,
            next,
            continuation["text"].as_str().unwrap().chars().count()
        )
    );
    let mut offset = 0;
    let mut ids = Vec::new();
    loop {
        let page = archive
            .query(RecallQuery::ReadTurn {
                turn_id: "unattributed".into(),
                detail: RecallDetail::Full,
                offset,
                limit: 10,
            })
            .unwrap();
        assert!(page.to_string().len() <= MAX_PAGE_BYTES);
        ids.extend(
            page["data"]
                .as_array()
                .unwrap()
                .iter()
                .map(|item| item["item_id"].as_str().unwrap().to_owned()),
        );
        let Some(next) = page["next_offset"].as_u64() else {
            break;
        };
        offset = next as usize;
    }
    assert_eq!(
        ids,
        (0..12)
            .map(|index| format!("msg_{index}"))
            .collect::<Vec<_>>()
    );
}

#[test]
fn pages_filter_turns_and_preserve_call_result_identity() {
    let mut archive = RecallArchive {
        turn_id: "turn_a".into(),
        ..Default::default()
    };
    for index in 0..12 {
        archive
            .record(
                response(&format!("msg_{index}"), "Needle"),
                &index.to_string(),
            )
            .unwrap();
    }
    for (kind, extra) in [
        ("function_call", json!({"name":"shell","arguments":"{}"})),
        ("function_call_output", json!({"output":"Needle result"})),
    ] {
        let mut value = extra;
        value["type"] = kind.into();
        value["call_id"] = "call_shared".into();
        let item = serde_json::from_value::<codex_protocol::models::ResponseItem>(value).unwrap();
        archive
            .record(RolloutItem::ResponseItem(item.into()), kind)
            .unwrap();
    }
    archive.turn_id = "turn_b".into();
    archive
        .record(response("msg_other", "elsewhere"), "other")
        .unwrap();
    let results = archive
        .query(RecallQuery::Search {
            query: "needle".into(),
            turn_id: None,
            offset: 0,
            limit: 100,
        })
        .unwrap();
    assert_eq!(
        (
            results["data"].as_array().unwrap().len(),
            results["total"].clone(),
            results["next_offset"].clone()
        ),
        (10, json!(13), json!(10))
    );
    let tools = archive
        .query(RecallQuery::ReadTurn {
            turn_id: "turn_a".into(),
            detail: RecallDetail::Tools,
            offset: 0,
            limit: 10,
        })
        .unwrap();
    assert_eq!(
        tools["data"]
            .as_array()
            .unwrap()
            .iter()
            .map(|item| item["item_id"].as_str().unwrap())
            .collect::<Vec<_>>(),
        vec![
            "function_call:call_shared",
            "function_call_output:call_shared"
        ]
    );
    let turns = archive
        .query(RecallQuery::ListTurns {
            offset: 1,
            limit: 1,
        })
        .unwrap();
    assert_eq!(
        (
            turns["data"][0]["turn_id"].clone(),
            turns["next_offset"].clone()
        ),
        (json!("turn_b"), Value::Null)
    );
}

fn write_records(path: &Path, items: Vec<RolloutItem>) -> u64 {
    let jsonl = items
        .into_iter()
        .enumerate()
        .map(|(ordinal, item)| {
            serde_json::to_string(&RolloutLine {
                timestamp: "2026-10-02T00:00:00Z".into(),
                ordinal: Some(ordinal as u64),
                item,
            })
            .unwrap()
                + "\n"
        })
        .collect::<String>();
    std::fs::write(path, &jsonl).unwrap();
    jsonl.len() as u64
}

fn meta(id: ThreadId, base: Option<HistoryPosition>) -> RolloutItem {
    RolloutItem::SessionMeta(SessionMetaLine {
        meta: SessionMeta {
            id,
            session_id: id.into(),
            history_base: base,
            ..Default::default()
        },
        git: None,
    })
}

#[tokio::test]
async fn fork_reads_only_frozen_parent_prefix_and_ignores_compacted_view() {
    let dir = tempfile::tempdir().unwrap();
    let sessions = dir.path().join("sessions/2026/10/02");
    std::fs::create_dir_all(&sessions).unwrap();
    let parent = ThreadId::new();
    let parent_path = sessions.join(format!("rollout-2026-10-02T00-00-00-{parent}.jsonl"));
    let prefix_items = vec![
        meta(parent, None),
        response("msg_parent", "parent original"),
    ];
    let bound = write_records(&parent_path, prefix_items.clone());
    let mut parent_items = prefix_items;
    parent_items.push(response("msg_future", "after fork"));
    write_records(&parent_path, parent_items);
    let child_path = dir.path().join("child.jsonl");
    let compacted = serde_json::from_value(json!({"type":"compacted","payload":{
        "message":"summary", "replacement_history":[]
    }}))
    .unwrap();
    write_records(
        &child_path,
        vec![
            meta(
                ThreadId::new(),
                Some(HistoryPosition {
                    thread_id: parent,
                    end_ordinal_exclusive: 2,
                    end_byte_offset: bound,
                }),
            ),
            compacted,
            response("msg_child", "child original"),
        ],
    );
    let archive = RecallArchive::load(&child_path, Some(dir.path()))
        .await
        .unwrap();
    assert_eq!(
        archive
            .items
            .iter()
            .map(|item| item.id.as_str())
            .collect::<Vec<_>>(),
        vec!["msg_parent", "msg_child"]
    );
    assert!(RecallArchive::load(&child_path, None).await.is_err());
}
