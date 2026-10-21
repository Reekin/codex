use codex_history::LocalCompactionKind;
use codex_history::LocalCompactionSource;
use codex_history::ResponseItemEnvelope;
use pretty_assertions::assert_eq;
use serde_json::json;

use super::ToolRun;
use super::tool_runs;
use crate::Decision;
use crate::StagedDecisions;

fn item(value: serde_json::Value) -> ResponseItemEnvelope {
    ResponseItemEnvelope::new(serde_json::from_value(value).unwrap())
}

fn message(id: &str, role: &str, text: &str) -> ResponseItemEnvelope {
    item(json!({"type":"message", "id":id, "role":role,
        "content":[{"type":"input_text", "text":text}]}))
}

fn call(id: &str, name: &str) -> ResponseItemEnvelope {
    item(
        json!({"type":"function_call", "id":format!("call_{id}"), "call_id":id,
        "name":name, "arguments":"{}"}),
    )
}

fn output(id: &str, text: &str) -> ResponseItemEnvelope {
    item(
        json!({"type":"function_call_output", "id":format!("out_{id}"), "call_id":id,
        "output":text}),
    )
}

fn cleaned(id: &str, text: &str) -> ResponseItemEnvelope {
    let mut output = output(id, text);
    output.metadata.get_or_insert_default().local_compaction = Some(LocalCompactionSource {
        first_item_id: format!("out_{id}"),
        last_item_id: format!("out_{id}"),
        kind: LocalCompactionKind::ToolResult,
    });
    output
}

fn history() -> Vec<ResponseItemEnvelope> {
    vec![
        message("user_1", "user", "Investigate."),
        item(json!({"type":"reasoning", "id":"rs_1", "summary":[]})),
        call("a", "exec"),
        output("a", "long listing"),
        call("b", "exec"),
        cleaned(
            "b",
            "Ran pytest: 3 passed\n[Original item: out_b; use recall_read_item.]",
        ),
        message("reply_1", "assistant", "Found it."),
        call("c", "exec"),
        output("c", "unjudged evidence"),
        call("d", "apply_patch"),
        output("d", "patch applied to 4 files"),
        message("user_2", "user", "Next."),
        call("e", "exec"),
        output("e", "newest evidence"),
    ]
}

fn staged(items: &[ResponseItemEnvelope]) -> StagedDecisions {
    let drop = |id: &str| Decision::Drop {
        id: id.to_string(),
        call_text: None,
    };
    StagedDecisions::from_marks(vec![
        (items[3].clone(), drop("out_a")),
        (
            items[10].clone(),
            Decision::Shorten {
                id: "out_d".to_string(),
                text: "Config is valid.".to_string(),
                call_text: None,
            },
        ),
        (items[13].clone(), drop("out_e")),
    ])
}

#[test]
fn low_value_tool_work_folds_into_runs_and_the_rest_stays() {
    let items = history();
    let staged = staged(&items);
    let runs = tool_runs(&items, &staged, &[false; 14], 0);
    assert_eq!(
        runs,
        vec![
            // Reasoning leading into a folded call goes with it; earlier cleanups keep their
            // conclusion.
            ToolRun {
                indices: vec![1, 2, 3, 4, 5],
                first_item_id: "rs_1".to_string(),
                last_item_id: "out_b".to_string(),
                text: "Removed tool calls: exec ×2.\n- exec (out_b): Ran pytest: 3 passed"
                    .to_string(),
            },
            // The unjudged result breaks the run; the newest work group is never folded.
            ToolRun {
                indices: vec![9, 10],
                first_item_id: "call_d".to_string(),
                last_item_id: "out_d".to_string(),
                text:
                    "Removed tool calls: apply_patch ×1.\n- apply_patch (out_d): Config is valid."
                        .to_string(),
            },
        ]
    );
    // Records before `from` are left to the summary, and trimmed reasoning stays gone.
    let mut dropped = [false; 14];
    dropped[1] = true;
    assert_eq!(
        tool_runs(&items, &staged, &dropped, 2)
            .into_iter()
            .map(|run| run.indices)
            .collect::<Vec<_>>(),
        vec![vec![2, 3, 4, 5], vec![9, 10]]
    );
    assert_eq!(
        tool_runs(&items, &staged, &dropped, 6)
            .into_iter()
            .map(|run| run.indices)
            .collect::<Vec<_>>(),
        vec![vec![9, 10]]
    );
}
