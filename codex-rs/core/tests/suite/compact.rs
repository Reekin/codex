use anyhow::Result;
use anyhow::anyhow;
use codex_core::TurnInputRequest;
use codex_core::config::Config;
use codex_features::Feature;
use codex_history::RolloutItem;
use codex_login::CodexAuth;
use codex_model_provider_info::ModelProviderInfo;
use codex_model_provider_info::built_in_model_providers;
use codex_models_manager::bundled_models_response;
use codex_protocol::config_types::CollaborationMode;
use codex_protocol::config_types::ModeKind;
use codex_protocol::config_types::Settings;
use codex_protocol::items::TurnItem;
use codex_protocol::models::PermissionProfile;
use codex_protocol::openai_models::ModelInfo;
use codex_protocol::openai_models::ModelsResponse;
use codex_protocol::protocol::AskForApproval;
use codex_protocol::protocol::CodexErrorInfo;
use codex_protocol::protocol::EventMsg;
use codex_protocol::protocol::ItemCompletedEvent;
use codex_protocol::protocol::ItemStartedEvent;
use codex_protocol::protocol::Op;
use codex_protocol::protocol::ThreadSettingsOverrides;
use codex_protocol::user_input::UserInput;
use codex_utils_absolute_path::AbsolutePathBuf;
use codex_utils_path_uri::PathUri;
use core_test_support::PathBufExt;
use core_test_support::context_snapshot;
use core_test_support::context_snapshot::ContextSnapshotOptions;
use core_test_support::responses;
use core_test_support::responses::ev_reasoning_item;
use core_test_support::responses::mount_models_once;
use core_test_support::skip_if_no_network;
use core_test_support::test_codex::local_requests;
use core_test_support::test_codex::test_codex;
use core_test_support::test_codex::turn_permission_fields;
use core_test_support::wait_for_event;
use std::path::PathBuf;

use core_test_support::responses::ev_assistant_message;
use core_test_support::responses::ev_completed;
use core_test_support::responses::ev_completed_with_tokens;
use core_test_support::responses::mount_response_sequence;
use core_test_support::responses::mount_sse_once;
use core_test_support::responses::mount_sse_sequence;
use core_test_support::responses::sse;
use core_test_support::responses::sse_failed;
use core_test_support::responses::sse_response;
use core_test_support::responses::start_mock_server;
use pretty_assertions::assert_eq;
use serde_json::Value;
use serde_json::json;
use std::fs;
use std::path::Path;
use std::sync::Arc;
use tempfile::TempDir;
use wiremock::MockServer;
#[path = "compact_local.rs"]
mod local;
#[path = "compact_local_support.rs"]
pub(crate) mod local_support;
#[path = "compact_local_tool_cleanup.rs"]
mod tool_cleanup;

// --- Test helpers -----------------------------------------------------------

pub(super) const FIRST_REPLY: &str = "FIRST_REPLY";
const THIRD_USER_MSG: &str = "next turn";
const FINAL_REPLY: &str = "FINAL_REPLY";
const PRETURN_CONTEXT_DIFF_CWD: &str = "/tmp/PRETURN_CONTEXT_DIFF_CWD";
const GLOBAL_AGENTS_FILENAME: &str = "AGENTS.md";
const NEW_GLOBAL_INSTRUCTIONS: &str = "new global instructions";
const OLD_GLOBAL_INSTRUCTIONS: &str = "old global instructions";
const REMOTE_V2_SUMMARY: &str = "global-instructions-remote-v2-summary";

fn disabled_permission_user_turn(
    text: impl Into<String>,
    cwd: PathBuf,
    model: String,
) -> TurnInputRequest {
    let (sandbox_policy, permission_profile) =
        turn_permission_fields(PermissionProfile::Disabled, cwd.as_path());
    TurnInputRequest::user_input(vec![UserInput::Text {
        text: text.into(),
        text_elements: Vec::new(),
    }])
    .with_thread_settings(ThreadSettingsOverrides {
        environments: Some(local_requests(cwd.abs())),
        approval_policy: Some(AskForApproval::Never),
        sandbox_policy: Some(sandbox_policy),
        permission_profile,
        collaboration_mode: Some(CollaborationMode {
            mode: ModeKind::Default,
            settings: Settings {
                model,
                reasoning_effort: None,
                developer_instructions: None,
            },
        }),
        ..Default::default()
    })
}

fn set_test_compact_prompt(config: &mut Config) {
    config.compact_prompt = Some("Preserve pending work and verification status.".to_string());
}

fn body_contains_text(body: &str, text: &str) -> bool {
    body.contains(&json_fragment(text))
}

fn json_fragment(text: &str) -> String {
    serde_json::to_string(text)
        .expect("serialize text to JSON")
        .trim_matches('"')
        .to_string()
}

fn read_hook_inputs(path: &Path) -> Vec<Value> {
    let text = fs::read_to_string(path).expect("failed to read hook input log");
    text.lines()
        .filter(|line| !line.trim().is_empty())
        .map(|line| serde_json::from_str(line).expect("failed to parse hook input log line"))
        .collect()
}

fn python_hook_command(script_path: &Path) -> String {
    format!("python3 \"{}\"", script_path.display())
}

fn write_unsupported_blocking_pre_compact_hook(home: &Path) {
    let script_path = home.join("pre_compact_block.py");
    let log_path = home.join("pre_compact_block_log.jsonl");
    let script = format!(
        r#"import json
from pathlib import Path
import sys

payload = json.load(sys.stdin)
with Path(r"{log_path}").open("a", encoding="utf-8") as handle:
    handle.write(json.dumps(payload) + "\n")

print(json.dumps({{"decision": "block", "reason": "blocked by policy"}}))
"#,
        log_path = log_path.display(),
    );
    let hooks = json!({
        "hooks": {
            "PreCompact": [{
                "matcher": "manual",
                "hooks": [{
                    "type": "command",
                    "command": python_hook_command(&script_path),
                    "statusMessage": "checking compact policy",
                }]
            }]
        }
    });

    fs::write(&script_path, script).expect("write pre compact hook script");
    fs::write(home.join("hooks.json"), hooks.to_string()).expect("write hooks.json");
}

fn write_matching_compact_hooks(home: &Path) {
    let auto_script_path = home.join("pre_compact_auto.py");
    let auto_log_path = home.join("pre_compact_auto_log.jsonl");
    let manual_post_script_path = home.join("post_compact_manual.py");
    let manual_post_log_path = home.join("post_compact_manual_log.jsonl");
    let auto_script = format!(
        r#"import json
from pathlib import Path
import sys

payload = json.load(sys.stdin)
with Path(r"{auto_log_path}").open("a", encoding="utf-8") as handle:
    handle.write(json.dumps(payload) + "\n")
"#,
        auto_log_path = auto_log_path.display(),
    );
    let manual_post_script = format!(
        r#"import json
from pathlib import Path
import sys

payload = json.load(sys.stdin)
with Path(r"{manual_post_log_path}").open("a", encoding="utf-8") as handle:
    handle.write(json.dumps(payload) + "\n")
"#,
        manual_post_log_path = manual_post_log_path.display(),
    );
    let hooks = json!({
        "hooks": {
            "PreCompact": [{
                "matcher": "auto",
                "hooks": [{
                    "type": "command",
                    "command": python_hook_command(&auto_script_path),
                }]
            }],
            "PostCompact": [{
                "matcher": "manual",
                "hooks": [{
                    "type": "command",
                    "command": python_hook_command(&manual_post_script_path),
                }]
            }]
        }
    });

    fs::write(&auto_script_path, auto_script).expect("write auto pre compact hook script");
    fs::write(&manual_post_script_path, manual_post_script)
        .expect("write manual post compact hook script");
    fs::write(home.join("hooks.json"), hooks.to_string()).expect("write hooks.json");
}

fn openai_model_provider(server: &MockServer) -> ModelProviderInfo {
    let mut provider =
        built_in_model_providers(/* openai_base_url */ /*openai_base_url*/ None)["openai"].clone();
    provider.base_url = Some(format!("{}/v1", server.uri()));
    provider.supports_websockets = false;
    provider
}

fn invalid_request_response(message: impl Into<String>) -> wiremock::ResponseTemplate {
    wiremock::ResponseTemplate::new(/*status*/ 400).set_body_json(json!({
        "detail": message.into(),
    }))
}

fn model_not_found_response(model: &str) -> wiremock::ResponseTemplate {
    wiremock::ResponseTemplate::new(/*status*/ 404).set_body_json(json!({
        "error": {
            "message": format!("Model not found {model}"),
            "type": "invalid_request_error",
            "param": "model",
            "code": null,
        }
    }))
}

fn write_global_file(
    home: &TempDir,
    filename: &str,
    contents: impl AsRef<[u8]>,
) -> Result<AbsolutePathBuf> {
    let path = home.path().join(filename);
    std::fs::write(&path, contents)?;
    Ok(path.abs())
}

fn instruction_fragments(request: &responses::ResponsesRequest) -> Vec<String> {
    request
        .message_input_texts("user")
        .into_iter()
        .filter(|text| text.starts_with("# AGENTS.md instructions"))
        .collect()
}

fn instruction_fragments_in_items(items: &[Value]) -> Vec<String> {
    items
        .iter()
        .filter(|item| {
            item.get("type").and_then(Value::as_str) == Some("message")
                && item.get("role").and_then(Value::as_str) == Some("user")
        })
        .filter_map(|item| item.get("content").and_then(Value::as_array))
        .flatten()
        .filter_map(|span| span.get("text").and_then(Value::as_str))
        .filter(|text| text.starts_with("# AGENTS.md instructions"))
        .map(str::to_string)
        .collect()
}

fn expected_instruction_fragment(contents: &str) -> String {
    format!("# AGENTS.md instructions\n\n<INSTRUCTIONS>\n{contents}\n</INSTRUCTIONS>")
}

fn assert_single_instruction_fragment(request: &responses::ResponsesRequest, expected: &str) {
    assert_eq!(instruction_fragments(request), vec![expected.to_string()]);
}

fn replacement_history_from_rollout(path: &Path) -> Result<Vec<Value>> {
    let rollout_text = fs::read_to_string(path)?;
    let mut replacement_history = None;
    for line in rollout_text
        .lines()
        .map(str::trim)
        .filter(|line| !line.is_empty())
    {
        let entry = codex_rollout::parse_rollout_line(line)?;
        if let RolloutItem::Compacted(compacted) = entry.item
            && let Some(items) = compacted.replacement_history
        {
            replacement_history = Some(
                items
                    .into_iter()
                    .map(|envelope| serde_json::to_value(envelope.item))
                    .collect::<std::result::Result<Vec<_>, _>>()?,
            );
        }
    }
    replacement_history.ok_or_else(|| anyhow!("expected rollout replacement history"))
}

fn remote_v2_compaction_response() -> String {
    responses::sse(vec![
        json!({
            "type": "response.output_item.done",
            "item": {
                "type": "compaction",
                "encrypted_content": REMOTE_V2_SUMMARY,
            }
        }),
        responses::ev_completed("remote-v2-compact-response"),
    ])
}

fn model_info_with_context_window(slug: &str, context_window: i64) -> ModelInfo {
    let models_response = bundled_models_response().expect("bundled models.json should parse");
    let mut model_info = models_response
        .models
        .into_iter()
        .find(|model| model.slug == "gpt-5.5")
        .expect("model missing from models.json");
    model_info.slug = slug.to_string();
    model_info.context_window = Some(context_window);
    model_info.max_context_window = Some(context_window);
    model_info.comp_hash = None;
    model_info
}

fn model_info_with_optional_comp_hash(slug: &str, comp_hash: Option<&str>) -> ModelInfo {
    let mut model_info = model_info_with_context_window(slug, /*context_window*/ 273_000);
    model_info.comp_hash = comp_hash.map(str::to_string);
    model_info
}

async fn assert_compaction_uses_turn_lifecycle_id(codex: &std::sync::Arc<codex_core::CodexThread>) {
    let mut turn_started_id = None;
    let mut turn_completed_id = None;
    let mut compact_started_id = None;
    let mut compact_completed_id = None;

    while turn_completed_id.is_none() {
        let event = codex.next_event().await.expect("next event");
        match event.msg {
            EventMsg::TurnStarted(_) => turn_started_id = Some(event.id.clone()),
            EventMsg::ItemStarted(ItemStartedEvent {
                item: TurnItem::ContextCompaction(_),
                ..
            }) => compact_started_id = Some(event.id.clone()),
            EventMsg::ItemCompleted(ItemCompletedEvent {
                item: TurnItem::ContextCompaction(_),
                ..
            }) => compact_completed_id = Some(event.id.clone()),
            EventMsg::Error(error) => panic!("unexpected compaction error: {error:?}"),
            EventMsg::TurnComplete(_) => turn_completed_id = Some(event.id.clone()),
            _ => {}
        }
    }

    let turn_started_id = turn_started_id.expect("turn started id");
    let turn_completed_id = turn_completed_id.expect("turn complete id");

    assert_eq!(
        turn_completed_id, turn_started_id,
        "turn start and complete should use the same event id"
    );
    assert_eq!(
        compact_started_id,
        Some(turn_started_id.clone()),
        "compaction item start should use the turn event id"
    );
    assert_eq!(
        compact_completed_id,
        Some(turn_started_id),
        "compaction item completion should use the turn event id"
    );
}
fn context_snapshot_options() -> ContextSnapshotOptions {
    ContextSnapshotOptions::default().rewrite_known_segments()
}

fn format_labeled_requests_snapshot(
    scenario: &str,
    sections: &[(&str, &core_test_support::responses::ResponsesRequest)],
) -> String {
    context_snapshot::format_labeled_requests_snapshot(
        scenario,
        sections,
        &context_snapshot_options(),
    )
}

#[test_case::test_case(false; "success")]
#[test_case::test_case(true; "failure")]
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn reasoning_effort_override_remote_v2_compaction_resets_pinned_effort(
    fail_compaction: bool,
) -> Result<()> {
    use codex_protocol::openai_models::ReasoningEffort;

    skip_if_no_network!(Ok(()));

    let server = start_mock_server().await;
    let request_log = mount_sse_sequence(
        &server,
        vec![
            sse(vec![
                ev_assistant_message("m1", "first reply"),
                ev_completed("r1"),
            ]),
            sse(vec![
                ev_assistant_message("m2", "second reply"),
                ev_completed("r2"),
            ]),
            if fail_compaction {
                sse_failed("r3", "server_error", "compaction failed")
            } else {
                remote_v2_compaction_response()
            },
            sse(vec![
                ev_assistant_message("m4", "third reply"),
                ev_completed("r4"),
            ]),
            sse(vec![ev_completed("r5")]),
            sse(vec![ev_completed("r6")]),
        ],
    )
    .await;

    let test = test_codex()
        .with_auth(CodexAuth::create_dummy_chatgpt_auth_for_testing())
        .with_model_info_override("gpt-5.4", |model| {
            model.use_responses_lite = true;
            model.supports_reasoning_effort_updates = true;
        })
        .with_config(move |config| {
            config.model_provider.stream_max_retries = Some(0);
            config.model_reasoning_effort = Some(ReasoningEffort::Medium);
            config
                .features
                .enable(Feature::ReasoningEffortOverride)
                .expect("test config should allow feature update");
            set_test_compact_prompt(config);
        })
        .build_with_auto_env(&server)
        .await?;

    test.submit_text_turn("first message").await?;
    core_test_support::submit_thread_settings(
        &test.codex,
        ThreadSettingsOverrides {
            effort: Some(Some(ReasoningEffort::High)),
            ..Default::default()
        },
    )
    .await?;
    test.submit_text_turn("second message").await?;
    test.codex.submit(Op::Compact).await?;
    if fail_compaction {
        wait_for_event(&test.codex, |event| matches!(event, EventMsg::Error(_))).await;
    }
    wait_for_event(&test.codex, |event| {
        matches!(event, EventMsg::TurnComplete(_))
    })
    .await;
    test.submit_text_turn("after compaction").await?;
    test.submit_text_turn("unchanged effort").await?;
    core_test_support::submit_thread_settings(
        &test.codex,
        ThreadSettingsOverrides {
            effort: Some(Some(ReasoningEffort::Low)),
            ..Default::default()
        },
    )
    .await?;
    test.submit_text_turn("changed effort").await?;

    let requests = request_log.requests();
    assert_eq!(requests.len(), 6);
    let first_body = requests[0].body_json();
    let second_body = requests[1].body_json();
    assert_eq!(first_body["reasoning"]["effort"], "medium");
    assert_eq!(second_body["reasoning"]["effort"], "medium");
    assert_eq!(requests[2].body_json()["reasoning"]["effort"], "medium");
    let post_compaction_effort = if fail_compaction { "medium" } else { "high" };
    assert_eq!(
        requests[3..]
            .iter()
            .map(|request| request.body_json()["reasoning"]["effort"].clone())
            .collect::<Vec<_>>(),
        vec![json!(post_compaction_effort); 3],
    );
    let updates = requests
        .iter()
        .map(|request| {
            request
                .input()
                .into_iter()
                .filter(|item| item["type"] == "configuration_update")
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();
    let medium = json!({
        "type": "configuration_update",
        "reasoning": {"effort": "medium"},
    });
    let high = json!({
        "type": "configuration_update",
        "reasoning": {"effort": "high"},
    });
    let low = json!({
        "type": "configuration_update",
        "reasoning": {"effort": "low"},
    });
    let retained = if fail_compaction {
        vec![medium.clone(), high.clone()]
    } else {
        vec![]
    };
    let mut changed = retained.clone();
    changed.push(low);
    assert_eq!(
        updates,
        [
            vec![medium.clone()],
            vec![medium.clone(), high.clone()],
            vec![medium, high],
            retained.clone(),
            retained,
            changed,
        ]
    );

    Ok(())
}

#[test_case::test_case(json!({"input_tokens": 120_000, "output_tokens": 30_000, "total_tokens": 150_000}); "reported_usage")]
#[test_case::test_case(Value::Null; "missing_usage")]
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn incomplete_response_during_compaction_is_terminal_and_counted(
    reported_usage: Value,
) -> Result<()> {
    skip_if_no_network!(Ok(()));
    for (provider, expected_error) in [(
        openai_model_provider as fn(&MockServer) -> ModelProviderInfo,
        "Error running remote compact task: Incomplete response returned, reason: max_output_tokens",
    )] {
        let server = start_mock_server().await;
        let previous_model = "gpt-5.4";
        let selected_model = "gpt-5.2";
        let incomplete = sse(vec![
            ev_assistant_message("discarded", "DISCARDED_INCOMPLETE_SUMMARY"),
            json!({
                "type": "response.incomplete",
                "response": {
                    "id": "incomplete-compaction",
                    "incomplete_details": {"reason": "max_output_tokens"},
                    "usage": reported_usage
                }
            }),
        ]);
        let response_mock = mount_sse_sequence(
            &server,
            vec![
                sse(vec![
                    ev_assistant_message("initial", "Initial reply."),
                    ev_completed_with_tokens("initial", /*total_tokens*/ 120_000),
                ]),
                incomplete.clone(),
                incomplete,
            ],
        )
        .await;
        let model_provider = provider(&server);
        let test = test_codex()
            .with_model(previous_model)
            .with_auth(CodexAuth::create_dummy_chatgpt_auth_for_testing())
            .with_config(move |config| {
                config.model_provider = model_provider;
                config.model_catalog = Some(ModelsResponse {
                    models: vec![
                        model_info_with_context_window(
                            previous_model,
                            /*context_window*/ 273_000,
                        ),
                        model_info_with_context_window(
                            selected_model,
                            /*context_window*/ 125_000,
                        ),
                    ],
                });
                config.model_provider.stream_max_retries = Some(2);
                set_test_compact_prompt(config);
            })
            .build_with_auto_env(&server)
            .await?;
        // A real large history remains installed after failure. Server usage also
        // includes request overhead that the local history estimate does not count.
        test.submit_turn(&"x".repeat(440_000)).await?;
        // A smaller selected model forces compaction with the previous model.
        // An incomplete result must stop both retries and fallback to the selected model.
        test.codex
            .start_or_steer_turn(
                TurnInputRequest::user_input(vec![UserInput::Text {
                    text: "continue with the smaller model".into(),
                    text_elements: Vec::new(),
                }])
                .with_thread_settings(ThreadSettingsOverrides {
                    collaboration_mode: Some(CollaborationMode {
                        mode: ModeKind::Default,
                        settings: Settings {
                            model: selected_model.to_string(),
                            reasoning_effort: None,
                            developer_instructions: None,
                        },
                    }),
                    ..Default::default()
                }),
            )
            .await?;
        let mut errors = Vec::new();
        wait_for_event(&test.codex, |event| {
            match event {
                EventMsg::Error(error) => errors.push(error.message.clone()),
                EventMsg::StreamError(value) => panic!("unexpected compaction retry: {value:?}"),
                EventMsg::RawResponseCompleted(value) => {
                    panic!("incomplete response marked complete: {value:?}")
                }
                _ => {}
            }
            matches!(event, EventMsg::TurnComplete(_))
        })
        .await;
        assert_eq!(errors, vec![expected_error]);
        assert_eq!(
            test.codex
                .token_usage_info()
                .await
                .expect("partial usage")
                .total_token_usage
                .total_tokens,
            if reported_usage.is_null() {
                120_000
            } else {
                270_000
            },
        );
        assert_eq!(
            response_mock
                .requests()
                .iter()
                .map(|request| request.body_json()["model"].clone())
                .collect::<Vec<_>>(),
            vec![json!(previous_model), json!(previous_model)],
        );
        // Failure has not made the original oversized history fit the smaller model.
        // The next turn must attempt compaction again without retaining partial output.
        test.codex
            .start_or_steer_turn(TurnInputRequest::user_input(vec![UserInput::Text {
                text: "continue after the failed compaction".into(),
                text_elements: Vec::new(),
            }]))
            .await?;
        wait_for_event(&test.codex, |event| {
            matches!(event, EventMsg::TurnComplete(_))
        })
        .await;
        let requests = response_mock.requests();
        assert_eq!(requests.len(), 3);
        assert_eq!(requests[2].body_json()["model"], json!(previous_model));
        assert!(!requests[2].inputs_of_type("compaction_trigger").is_empty());
        assert!(!body_contains_text(
            &requests[2].body_json().to_string(),
            "DISCARDED_INCOMPLETE_SUMMARY",
        ));
    }
    Ok(())
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn auto_compact_runs_after_resume_when_token_usage_is_over_limit() {
    skip_if_no_network!();

    let server = start_mock_server().await;

    let limit = 200_000;
    let over_limit_tokens = 250_000;
    let remote_summary = REMOTE_V2_SUMMARY;

    let mut builder = test_codex().with_config(move |config| {
        set_test_compact_prompt(config);
        config.model_auto_compact_token_limit = Some(limit);
    });
    let initial = builder.build(&server).await.unwrap();

    // A single over-limit completion should not auto-compact until the next user message.
    let initial_response = mount_sse_once(
        &server,
        sse(vec![
            ev_assistant_message("m1", FIRST_REPLY),
            ev_completed_with_tokens("r1", over_limit_tokens),
        ]),
    )
    .await;
    initial.submit_turn("OVER_LIMIT_TURN").await.unwrap();

    let initial_requests = initial_response.requests();
    assert_eq!(initial_requests.len(), 1);
    assert!(
        initial_requests[0]
            .inputs_of_type("compaction_trigger")
            .is_empty(),
        "remote compaction should not run before the next user message"
    );

    let mut resume_builder = test_codex().with_config(move |config| {
        set_test_compact_prompt(config);
        config.model_auto_compact_token_limit = Some(limit);
    });
    let resumed = resume_builder.restart(&server, &initial).await.unwrap();

    let follow_up_user = "AFTER_RESUME_USER";
    let compact_turn = remote_v2_compaction_response();
    let sse_follow_up = sse(vec![
        ev_assistant_message("m2", FINAL_REPLY),
        ev_completed("r2"),
    ]);
    let response_mock = mount_sse_sequence(&server, vec![compact_turn, sse_follow_up]).await;

    resumed
        .codex
        .start_or_steer_turn(disabled_permission_user_turn(
            follow_up_user,
            resumed.cwd.path().to_path_buf(),
            resumed.session_configured.model.clone(),
        ))
        .await
        .unwrap();

    wait_for_event(&resumed.codex, |event| {
        matches!(event, EventMsg::ContextCompacted(_))
    })
    .await;
    wait_for_event(&resumed.codex, |event| {
        matches!(event, EventMsg::TurnComplete(_))
    })
    .await;

    let requests = response_mock.requests();
    assert_eq!(
        requests.len(),
        2,
        "remote compaction should run once after resume before sampling"
    );
    assert_eq!(
        requests[0].path(),
        "/v1/responses",
        "remote compaction should use the responses endpoint"
    );
    assert_eq!(requests[0].inputs_of_type("compaction_trigger").len(), 1);
    let follow_up_body = requests[1].body_json().to_string();
    assert!(follow_up_body.contains(follow_up_user));
    assert!(follow_up_body.contains(remote_summary));
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn pre_sampling_compact_falls_back_from_retired_previous_model_after_rename() {
    skip_if_no_network!();

    let server = MockServer::start().await;
    let retired_model = "gpt-5.5";
    let previous_model_family = "gpt-5.5";
    let renamed_model = "gpt-5.6";
    let mut previous_model_info = model_info_with_optional_comp_hash("gpt-5.4", Some("hash-a"));
    previous_model_info.slug = previous_model_family.to_string();
    let mut renamed_model_info = model_info_with_optional_comp_hash("gpt-5.4", Some("hash-b"));
    renamed_model_info.slug = renamed_model.to_string();

    let models_mock = mount_models_once(
        &server,
        ModelsResponse {
            models: vec![previous_model_info, renamed_model_info],
        },
    )
    .await;

    let request_log = mount_response_sequence(
        &server,
        vec![
            sse_response(sse(vec![
                ev_assistant_message("m1", "before switch"),
                ev_completed_with_tokens("r1", /*total_tokens*/ 100),
            ])),
            invalid_request_response(format!(
                "The '{retired_model}' model is not supported when using Codex with a ChatGPT account."
            )),
            sse_response(sse(vec![
                json!({
                    "type": "response.output_item.done",
                    "item": {
                        "type": "compaction",
                        "encrypted_content": "RETIRED_MODEL_SUMMARY",
                    }
                }),
                ev_completed_with_tokens("r2", /*total_tokens*/ 10),
            ])),
            sse_response(sse(vec![
                ev_assistant_message("m3", "after switch"),
                ev_completed_with_tokens("r3", /*total_tokens*/ 100),
            ])),
        ],
    )
    .await;

    let model_provider = openai_model_provider(&server);
    let mut initial_builder = test_codex()
        .with_auth(CodexAuth::create_dummy_chatgpt_auth_for_testing())
        .with_model(retired_model)
        .with_config(move |config| {
            config.model_provider = model_provider;
            set_test_compact_prompt(config);
        });
    let initial = initial_builder
        .build(&server)
        .await
        .expect("build initial test codex");
    let home = initial.home.clone();
    let rollout_path = initial
        .session_configured
        .rollout_path
        .clone()
        .expect("rollout path");

    initial
        .codex
        .start_or_steer_turn(disabled_permission_user_turn(
            "before switch",
            initial.cwd.path().to_path_buf(),
            retired_model.to_string(),
        ))
        .await
        .expect("submit first user turn");
    wait_for_event(&initial.codex, |event| {
        matches!(event, EventMsg::TurnComplete(_))
    })
    .await;

    initial
        .codex
        .submit(Op::Shutdown)
        .await
        .expect("shutdown initial session");
    wait_for_event(&initial.codex, |event| {
        matches!(event, EventMsg::ShutdownComplete)
    })
    .await;

    let model_provider = openai_model_provider(&server);
    let mut resumed_builder = test_codex()
        .with_auth(CodexAuth::create_dummy_chatgpt_auth_for_testing())
        .with_model(retired_model)
        .with_config(move |config| {
            config.model_provider = model_provider;
            set_test_compact_prompt(config);
        });
    let resumed = resumed_builder
        .resume(&server, home, rollout_path)
        .await
        .expect("resume codex");

    resumed
        .codex
        .start_or_steer_turn(disabled_permission_user_turn(
            "after switch",
            resumed.cwd.path().to_path_buf(),
            renamed_model.to_string(),
        ))
        .await
        .expect("submit renamed-model turn");
    assert_compaction_uses_turn_lifecycle_id(&resumed.codex).await;

    let requests = request_log.requests();
    assert_eq!(models_mock.requests().len(), 1);
    assert_eq!(
        requests.len(),
        4,
        "the renamed-model turn should retry compaction and then sample"
    );
    assert_eq!(
        requests[0].body_json()["model"].as_str(),
        Some(retired_model)
    );
    assert_eq!(
        requests[1].body_json()["model"].as_str(),
        Some(retired_model)
    );
    assert_eq!(
        requests[2].body_json()["model"].as_str(),
        Some(renamed_model)
    );
    assert_eq!(
        requests[3].body_json()["model"].as_str(),
        Some(renamed_model)
    );
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn pre_sampling_compact_falls_back_when_previous_model_is_not_found() {
    skip_if_no_network!();

    let server = MockServer::start().await;
    let retired_model = "gpt-5.5";
    let previous_model_family = "gpt-5.5";
    let renamed_model = "gpt-5.6";
    let mut previous_model_info = model_info_with_optional_comp_hash("gpt-5.4", Some("hash-a"));
    previous_model_info.slug = previous_model_family.to_string();
    let mut renamed_model_info = model_info_with_optional_comp_hash("gpt-5.4", Some("hash-b"));
    renamed_model_info.slug = renamed_model.to_string();

    let models_mock = mount_models_once(
        &server,
        ModelsResponse {
            models: vec![previous_model_info, renamed_model_info],
        },
    )
    .await;

    let request_log = mount_response_sequence(
        &server,
        vec![
            sse_response(sse(vec![
                ev_assistant_message("m1", "before switch"),
                ev_completed_with_tokens("r1", /*total_tokens*/ 100),
            ])),
            model_not_found_response(retired_model),
            sse_response(sse(vec![
                json!({
                    "type": "response.output_item.done",
                    "item": {
                        "type": "compaction",
                        "encrypted_content": "RETIRED_MODEL_SUMMARY",
                    }
                }),
                ev_completed_with_tokens("r2", /*total_tokens*/ 10),
            ])),
            sse_response(sse(vec![
                ev_assistant_message("m3", "after switch"),
                ev_completed_with_tokens("r3", /*total_tokens*/ 100),
            ])),
        ],
    )
    .await;

    let model_provider = openai_model_provider(&server);
    let mut initial_builder = test_codex()
        .with_auth(CodexAuth::create_dummy_chatgpt_auth_for_testing())
        .with_model(retired_model)
        .with_config(move |config| {
            config.model_provider = model_provider;
            set_test_compact_prompt(config);
        });
    let initial = initial_builder
        .build(&server)
        .await
        .expect("build initial test codex");
    let home = initial.home.clone();
    let rollout_path = initial
        .session_configured
        .rollout_path
        .clone()
        .expect("rollout path");

    initial
        .codex
        .start_or_steer_turn(disabled_permission_user_turn(
            "before switch",
            initial.cwd.path().to_path_buf(),
            retired_model.to_string(),
        ))
        .await
        .expect("submit first user turn");
    wait_for_event(&initial.codex, |event| {
        matches!(event, EventMsg::TurnComplete(_))
    })
    .await;

    initial
        .codex
        .submit(Op::Shutdown)
        .await
        .expect("shutdown initial session");
    wait_for_event(&initial.codex, |event| {
        matches!(event, EventMsg::ShutdownComplete)
    })
    .await;

    let mut model_provider = openai_model_provider(&server);
    model_provider.stream_max_retries = Some(0);
    let mut resumed_builder = test_codex()
        .with_auth(CodexAuth::create_dummy_chatgpt_auth_for_testing())
        .with_model(retired_model)
        .with_config(move |config| {
            config.model_provider = model_provider;
            set_test_compact_prompt(config);
        });
    let resumed = resumed_builder
        .resume(&server, home, rollout_path)
        .await
        .expect("resume codex");

    resumed
        .codex
        .start_or_steer_turn(disabled_permission_user_turn(
            "after switch",
            resumed.cwd.path().to_path_buf(),
            renamed_model.to_string(),
        ))
        .await
        .expect("submit renamed-model turn");
    assert_compaction_uses_turn_lifecycle_id(&resumed.codex).await;

    let requests = request_log.requests();
    assert_eq!(models_mock.requests().len(), 1);
    assert_eq!(
        requests.len(),
        4,
        "the renamed-model turn should retry compaction and then sample"
    );
    assert_eq!(
        requests[0].body_json()["model"].as_str(),
        Some(retired_model)
    );
    assert_eq!(
        requests[1].body_json()["model"].as_str(),
        Some(retired_model)
    );
    assert_eq!(
        requests[2].body_json()["model"].as_str(),
        Some(renamed_model)
    );
    assert_eq!(
        requests[3].body_json()["model"].as_str(),
        Some(renamed_model)
    );
}

#[test_case::test_case("openai"; "built in provider")]
#[test_case::test_case("local"; "cloud agent provider alias")]
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn pre_sampling_compact_falls_back_after_previous_model_invalid_request_on_downshift(
    provider_id: &'static str,
) {
    skip_if_no_network!();

    let server = MockServer::start().await;
    let retired_model = "gpt-5.6";
    let previous_model_family = "gpt-5.6";
    let next_model = "gpt-5.5";
    let mut previous_model_info =
        model_info_with_context_window("gpt-5.4", /*context_window*/ 273_000);
    previous_model_info.slug = previous_model_family.to_string();
    previous_model_info.use_responses_lite = true;
    let mut next_model_info =
        model_info_with_context_window("gpt-5.4", /*context_window*/ 125_000);
    next_model_info.slug = next_model.to_string();
    next_model_info.use_responses_lite = false;

    let models_mock = mount_models_once(
        &server,
        ModelsResponse {
            models: vec![previous_model_info, next_model_info],
        },
    )
    .await;

    let request_log = mount_response_sequence(
        &server,
        vec![
            sse_response(sse(vec![
                ev_assistant_message("m1", "before switch"),
                ev_completed_with_tokens("r1", /*total_tokens*/ 120_000),
            ])),
            invalid_request_response("previous-model compaction was rejected"),
            sse_response(sse(vec![
                json!({
                    "type": "response.output_item.done",
                    "item": {
                        "type": "compaction",
                        "encrypted_content": "DOWNSHIFT_SUMMARY",
                    }
                }),
                ev_completed_with_tokens("r2", /*total_tokens*/ 10),
            ])),
            sse_response(sse(vec![
                ev_assistant_message("m3", "after switch"),
                ev_completed_with_tokens("r3", /*total_tokens*/ 100),
            ])),
        ],
    )
    .await;

    let model_provider = openai_model_provider(&server);
    let mut builder = test_codex()
        .with_auth(CodexAuth::create_dummy_chatgpt_auth_for_testing())
        .with_model(retired_model)
        .with_config(move |config| {
            config.model_provider_id = provider_id.to_owned();
            config.model_provider = model_provider;
            config.tool_registry.turn_metadata_includes_tool_info = true;
            set_test_compact_prompt(config);
        });
    let test = builder
        .build_with_auto_env(&server)
        .await
        .expect("build test codex");

    test.codex
        .start_or_steer_turn(disabled_permission_user_turn(
            "before switch",
            test.cwd.path().to_path_buf(),
            retired_model.to_string(),
        ))
        .await
        .expect("submit first user turn");
    wait_for_event(&test.codex, |event| {
        matches!(event, EventMsg::TurnComplete(_))
    })
    .await;

    test.codex
        .start_or_steer_turn(disabled_permission_user_turn(
            "after switch",
            test.cwd.path().to_path_buf(),
            next_model.to_string(),
        ))
        .await
        .expect("submit smaller-model turn");
    assert_compaction_uses_turn_lifecycle_id(&test.codex).await;

    let requests = request_log.requests();
    assert_eq!(models_mock.requests().len(), 1);
    assert_eq!(
        requests.len(),
        4,
        "the smaller-model turn should retry compaction and then sample"
    );
    assert_eq!(
        requests[0].body_json()["model"].as_str(),
        Some(retired_model)
    );
    assert_eq!(
        requests[1].body_json()["model"].as_str(),
        Some(retired_model)
    );
    assert_eq!(requests[2].body_json()["model"].as_str(), Some(next_model));
    assert_eq!(requests[3].body_json()["model"].as_str(), Some(next_model));

    // Preparing the non-Lite fallback must not clear the previous model's inventory
    // before its compaction attempt is sent.
    let [first_metadata, compact_metadata] = [&requests[0], &requests[1]].map(|request| {
        serde_json::from_str::<Value>(
            request.body_json()["client_metadata"]["x-codex-turn-metadata"]
                .as_str()
                .expect("request should include turn metadata"),
        )
        .expect("turn metadata should be valid JSON")
    });
    let inventory = first_metadata["tool_namespaces_info"]
        .as_object()
        .expect("Responses Lite request should include tool inventory");
    assert!(!inventory.is_empty());
    assert_eq!(
        compact_metadata["tool_namespaces_info"],
        first_metadata["tool_namespaces_info"],
    );
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn pre_sampling_compact_falls_back_after_previous_model_stream_retries_are_exhausted() {
    skip_if_no_network!();

    let server = MockServer::start().await;
    let previous_model = "gpt-5.4";
    let selected_model = "gpt-5.2";
    let _models_mock = mount_models_once(
        &server,
        ModelsResponse {
            models: vec![
                model_info_with_context_window(previous_model, /*context_window*/ 273_000),
                model_info_with_context_window(selected_model, /*context_window*/ 125_000),
            ],
        },
    )
    .await;

    let compaction_failure = sse_failed(
        "compact-failure",
        "server_error",
        "previous-model compaction failed",
    );
    let request_log = mount_sse_sequence(
        &server,
        vec![
            sse(vec![
                ev_assistant_message("m1", "before switch"),
                ev_completed_with_tokens("r1", /*total_tokens*/ 120_000),
            ]),
            compaction_failure.clone(),
            compaction_failure.clone(),
            compaction_failure,
            remote_v2_compaction_response(),
            sse(vec![
                ev_assistant_message("m3", "after switch"),
                ev_completed_with_tokens("r3", /*total_tokens*/ 100),
            ]),
        ],
    )
    .await;

    let mut model_provider = openai_model_provider(&server);
    model_provider.stream_max_retries = Some(2);
    let mut builder = test_codex()
        .with_auth(CodexAuth::create_dummy_chatgpt_auth_for_testing())
        .with_model(previous_model)
        .with_config(move |config| {
            config.model_provider = model_provider;
            set_test_compact_prompt(config);
        });
    let test = builder.build(&server).await.expect("build test codex");

    test.codex
        .start_or_steer_turn(disabled_permission_user_turn(
            "before switch",
            test.cwd.path().to_path_buf(),
            previous_model.to_string(),
        ))
        .await
        .expect("submit first user turn");
    wait_for_event(&test.codex, |event| {
        matches!(event, EventMsg::TurnComplete(_))
    })
    .await;

    test.codex
        .start_or_steer_turn(disabled_permission_user_turn(
            "after switch",
            test.cwd.path().to_path_buf(),
            selected_model.to_string(),
        ))
        .await
        .expect("submit selected-model turn");
    assert_compaction_uses_turn_lifecycle_id(&test.codex).await;

    let actual_models = request_log
        .requests()
        .iter()
        .map(|request| {
            request.body_json()["model"]
                .as_str()
                .expect("request model")
                .to_string()
        })
        .collect::<Vec<_>>();
    assert_eq!(
        actual_models,
        vec![
            previous_model, // Initial turn.
            previous_model, // Compaction attempt.
            previous_model, // First retry.
            previous_model, // Second retry.
            selected_model, // Fallback compaction.
            selected_model, // Turn sampling.
        ]
    );
}

/// Both compaction models keep the configured retry budget when the server supplies advice,
/// including the selected model's final attempt.
#[test_case::test_case(2; "selected_model_recovers")]
#[test_case::test_case(3; "selected_model_exhausts_retries")]
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn pre_sampling_compact_advised_errors_fall_back_then_retry_on_selected_model(
    selected_model_failures: usize,
) {
    skip_if_no_network!();

    let server = MockServer::start().await;
    let previous_model = "gpt-5.4";
    let selected_model = "gpt-5.2";
    let _models_mock = mount_models_once(
        &server,
        ModelsResponse {
            models: vec![
                model_info_with_context_window(previous_model, /*context_window*/ 273_000),
                model_info_with_context_window(selected_model, /*context_window*/ 125_000),
            ],
        },
    )
    .await;
    let overload = wiremock::ResponseTemplate::new(/*s*/ 503)
        .insert_header("Retry-After", "0")
        .set_body_json(json!({ "error": { "code": "server_is_overloaded" } }));
    let mut responses = vec![
        sse_response(sse(vec![
            ev_assistant_message("m1", "before switch"),
            ev_completed_with_tokens("r1", /*total_tokens*/ 120_000),
        ])),
        overload.clone(), // Old model: initial attempt and two retries.
        overload.clone(),
        overload.clone(),
    ];
    responses.extend(vec![overload; selected_model_failures]);
    if selected_model_failures < 3 {
        responses.extend([
            sse_response(remote_v2_compaction_response()),
            sse_response(sse(vec![
                ev_assistant_message("m3", "after switch"),
                ev_completed_with_tokens("r3", /*total_tokens*/ 100),
            ])),
        ]);
    }
    let request_log = mount_response_sequence(&server, responses).await;
    let mut model_provider = openai_model_provider(&server);
    model_provider.request_max_retries = Some(0);
    model_provider.stream_max_retries = Some(2);
    let test = test_codex()
        .with_auth(CodexAuth::create_dummy_chatgpt_auth_for_testing())
        .with_model(previous_model)
        .with_config(move |config| {
            config.model_provider = model_provider;
            set_test_compact_prompt(config);
        })
        .build_with_auto_env(&server)
        .await
        .expect("build codex");
    test.codex
        .start_or_steer_turn(disabled_permission_user_turn(
            "before switch",
            test.cwd.path().to_path_buf(),
            previous_model.to_string(),
        ))
        .await
        .expect("submit first turn");
    wait_for_event(&test.codex, |event| {
        matches!(event, EventMsg::TurnComplete(_))
    })
    .await;
    test.codex
        .start_or_steer_turn(disabled_permission_user_turn(
            "after switch",
            test.cwd.path().to_path_buf(),
            selected_model.to_string(),
        ))
        .await
        .expect("submit selected-model turn");
    if selected_model_failures < 3 {
        assert_compaction_uses_turn_lifecycle_id(&test.codex).await;
    } else {
        let mut errors = Vec::new();
        loop {
            match wait_for_event(&test.codex, |_| true).await {
                EventMsg::Error(error) => errors.push(error.codex_error_info),
                EventMsg::TurnComplete(event) => {
                    assert_eq!(
                        event.error.and_then(|error| error.codex_error_info),
                        Some(CodexErrorInfo::ServerOverloaded)
                    );
                    break;
                }
                _ => {}
            }
        }
        assert_eq!(errors, vec![Some(CodexErrorInfo::ServerOverloaded)]);
    }
    let actual_models = request_log
        .requests()
        .iter()
        .map(|request| {
            request.body_json()["model"]
                .as_str()
                .expect("request model")
                .to_string()
        })
        .collect::<Vec<_>>();
    let mut expected_models = vec![
        previous_model, // First turn.
        previous_model, // Old-model compaction attempt and retries.
        previous_model,
        previous_model,
        selected_model, // Selected-model compaction attempt and retries.
        selected_model,
        selected_model,
    ];
    if selected_model_failures < 3 {
        expected_models.push(selected_model); // Sampling after compaction succeeds.
    }
    assert_eq!(actual_models, expected_models);
}

/// Manual compaction keeps the configured retry limit even when the server supplies advice.
#[test_case::test_case(0; "no_retries")]
#[test_case::test_case(2; "two_retries")]
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn manual_remote_compact_advice_respects_retry_limit(max_retries: u64) {
    skip_if_no_network!();

    let server = start_mock_server().await;
    let overload = wiremock::ResponseTemplate::new(/*s*/ 503)
        .insert_header("Retry-After", "0")
        .set_body_json(json!({ "error": { "code": "server_is_overloaded" } }));
    let request_log =
        mount_response_sequence(&server, vec![overload; (max_retries + 1) as usize]).await;
    let mut model_provider = openai_model_provider(&server);
    model_provider.request_max_retries = Some(0);
    model_provider.stream_max_retries = Some(max_retries);
    let test = test_codex()
        .with_auth(CodexAuth::create_dummy_chatgpt_auth_for_testing())
        .with_model("gpt-5.4")
        .with_config(move |config| {
            config.model_provider = model_provider;
            set_test_compact_prompt(config);
        })
        .build_with_auto_env(&server)
        .await
        .expect("build codex");

    test.codex.submit(Op::Compact).await.expect("run /compact");
    let mut errors = Vec::new();
    let mut reconnect_messages = Vec::new();
    loop {
        match wait_for_event(&test.codex, |_| true).await {
            EventMsg::Error(error) => errors.push(error.codex_error_info),
            EventMsg::StreamError(error) => reconnect_messages.push(error.message),
            EventMsg::TurnComplete(event) => {
                assert_eq!(
                    event.error.and_then(|error| error.codex_error_info),
                    Some(CodexErrorInfo::ServerOverloaded)
                );
                break;
            }
            _ => {}
        }
    }
    assert_eq!(errors, vec![Some(CodexErrorInfo::ServerOverloaded)]);
    assert_eq!(
        reconnect_messages,
        (1..=max_retries)
            .map(|retry| format!("Reconnecting... {retry}/{max_retries}"))
            .collect::<Vec<_>>()
    );
    let models = request_log
        .requests()
        .iter()
        .map(|request| {
            request.body_json()["model"]
                .as_str()
                .expect("request model")
                .to_string()
        })
        .collect::<Vec<_>>();
    assert_eq!(models, vec!["gpt-5.4"; (max_retries + 1) as usize]);
}

#[derive(Clone, Copy)]
enum ContinueThread {
    Live,
    Fork,
    Restart,
}

#[test_case::test_case(ContinueThread::Live; "live")]
#[test_case::test_case(ContinueThread::Fork; "fork")]
#[test_case::test_case(ContinueThread::Restart; "cold resume")]
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn auto_compact_accounts_for_encrypted_reasoning(continuation: ContinueThread) {
    skip_if_no_network!();

    let server = start_mock_server().await;

    let pre_last_reasoning_content = "a".repeat(2_400);
    let post_last_reasoning_content = "b".repeat(4_000);

    let first_turn = sse(vec![
        ev_reasoning_item("pre-reasoning", &["pre"], &[&pre_last_reasoning_content]),
        ev_completed_with_tokens("r1", /*total_tokens*/ 10),
    ]);
    let second_turn = sse(vec![
        ev_reasoning_item("post-reasoning", &["post"], &[&post_last_reasoning_content]),
        ev_completed_with_tokens("r2", /*total_tokens*/ 80),
    ]);
    let third_turn = sse(vec![
        ev_assistant_message("m3", THIRD_USER_MSG),
        ev_completed_with_tokens("r3", /*total_tokens*/ 81),
    ]);
    let compact_turn = remote_v2_compaction_response();
    let fourth_turn = sse(vec![
        ev_assistant_message("m4", FINAL_REPLY),
        ev_completed_with_tokens("r4", /*total_tokens*/ 1),
    ]);

    let request_log = mount_sse_sequence(
        &server,
        vec![
            first_turn,
            second_turn,
            third_turn,
            compact_turn,
            fourth_turn,
        ],
    )
    .await;

    let builder = || {
        test_codex()
            .with_auth(CodexAuth::create_dummy_chatgpt_auth_for_testing())
            .with_model_info_override("gpt-5.5", |model| {
                model.use_responses_lite = true;
            })
            .with_config(|config| {
                config.model = Some("gpt-5.5".to_string());
                set_test_compact_prompt(config);
                config.model_auto_compact_token_limit = Some(81);
            })
    };
    let mut test = builder()
        .build_with_auto_env(&server)
        .await
        .expect("build codex");

    for user in ["FIRST_REASONING", "SECOND_REASONING"] {
        test.submit_text_turn(user).await.expect("submit user turn");
    }

    match continuation {
        ContinueThread::Live => {}
        ContinueThread::Restart => {
            test = builder()
                .restart_with_auto_env(&server, &test)
                .await
                .expect("restart");
        }
        ContinueThread::Fork => {
            let path = test
                .session_configured
                .rollout_path
                .clone()
                .expect("rollout path");
            let options = test.start_thread_options().await;
            test.codex.shutdown_and_wait().await.expect("flush rollout");
            test.codex = test
                .thread_manager
                .fork_legacy_thread(usize::MAX, options, path)
                .await
                .expect("fork")
                .thread;
        }
    }

    for user in ["BELOW_COMPACT_LIMIT", "AT_COMPACT_LIMIT"] {
        test.submit_text_turn(user).await.expect("submit user turn");
    }

    let requests = request_log.requests();
    // Responses Lite includes retained reasoning in its usage without a header.
    // 80 must remain below the limit; 81 must trigger compaction before turn 4.
    assert_eq!(
        requests
            .iter()
            .map(|request| request.inputs_of_type("compaction_trigger").len())
            .collect::<Vec<_>>(),
        vec![0, 0, 0, 1, 0],
    );
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn remote_v2_compaction_refreshes_instructions_and_preserves_them_on_cold_resume()
-> Result<()> {
    skip_if_no_network!(Ok(()));

    // Set up an ordinary turn, a remote-v2 compact response, and a post-compaction turn.
    let server = responses::start_mock_server().await;
    let response_mock = responses::mount_sse_sequence(
        &server,
        vec![
            responses::sse(vec![
                responses::ev_response_created("remote-v2-initial-response"),
                responses::ev_completed("remote-v2-initial-response"),
            ]),
            remote_v2_compaction_response(),
            responses::sse(vec![
                responses::ev_response_created("remote-v2-follow-up-response"),
                responses::ev_completed("remote-v2-follow-up-response"),
            ]),
            responses::sse(vec![
                responses::ev_response_created("remote-v2-resumed-response"),
                responses::ev_completed("remote-v2-resumed-response"),
            ]),
        ],
    )
    .await;
    let home = Arc::new(TempDir::new()?);
    let source = write_global_file(
        home.as_ref(),
        GLOBAL_AGENTS_FILENAME,
        OLD_GLOBAL_INSTRUCTIONS,
    )?;
    let mut builder = test_codex()
        .with_home(Arc::clone(&home))
        .with_model_info_override("gpt-5.5", |model| model.use_responses_lite = true)
        .with_auth(CodexAuth::create_dummy_chatgpt_auth_for_testing());
    let test = builder.build(&server).await?;

    // Materialize the old snapshot, rewrite the selected file in place, and compact remotely.
    test.submit_turn("before remote v2 compaction").await?;
    let rewritten_source = write_global_file(
        home.as_ref(),
        GLOBAL_AGENTS_FILENAME,
        NEW_GLOBAL_INSTRUCTIONS,
    )?;
    assert_eq!(source, rewritten_source);
    test.codex.submit(Op::Compact).await?;
    wait_for_event(&test.codex, |event| {
        matches!(event, EventMsg::TurnComplete(_))
    })
    .await;
    test.submit_turn("after remote v2 compaction").await?;
    test.codex.flush_rollout().await?;

    // Compaction summarizes the existing history; the follow-up injects the refreshed instructions.
    let requests = response_mock.requests();
    assert_eq!(requests.len(), 3);
    let old_fragment = expected_instruction_fragment(OLD_GLOBAL_INSTRUCTIONS);
    let new_fragment = expected_instruction_fragment(NEW_GLOBAL_INSTRUCTIONS);
    assert_single_instruction_fragment(&requests[0], &old_fragment);
    assert_single_instruction_fragment(&requests[1], &old_fragment);
    assert_single_instruction_fragment(&requests[2], &new_fragment);
    assert_eq!(
        requests[1].input().last(),
        Some(&json!({"type": "compaction_trigger"})),
        "remote-v2 compact request should append exactly one compaction trigger"
    );
    let rollout_path = test.codex.rollout_path().expect("rollout path");
    let replacement_history = replacement_history_from_rollout(&rollout_path)?;
    assert_eq!(
        instruction_fragments_in_items(&replacement_history),
        vec![new_fragment.clone()],
        "remote-v2 replacement history includes refreshed global instructions"
    );
    assert_eq!(
        test.codex.instruction_sources().await,
        vec![PathUri::from_abs_path(&source)],
        "running thread retains the selected same-path source"
    );
    assert_eq!(
        fs::read_to_string(source.as_path())?,
        NEW_GLOBAL_INSTRUCTIONS,
        "the selected source path should contain the rewritten text"
    );

    // Cold-resume the persisted replacement history with freshly loaded same-path configuration.
    test.codex.submit(Op::Shutdown).await?;
    wait_for_event(&test.codex, |event| {
        matches!(event, EventMsg::ShutdownComplete)
    })
    .await;
    let resumed_cwd = test.config.cwd.clone();
    let mut resume_builder = test_codex()
        .with_home(Arc::clone(&home))
        .with_model_info_override("gpt-5.5", |model| model.use_responses_lite = true)
        .with_auth(CodexAuth::create_dummy_chatgpt_auth_for_testing())
        .with_config(move |config| config.cwd = resumed_cwd);
    let resumed = resume_builder
        .resume(&server, Arc::clone(&home), rollout_path)
        .await?;
    resumed
        .submit_turn("after remote v2 compaction cold resume")
        .await?;

    // Cold resume replays the refreshed context without appending unchanged instructions again.
    let requests = response_mock.requests();
    assert_eq!(requests.len(), 4);
    assert_single_instruction_fragment(&requests[3], &new_fragment);
    let resumed_input = requests[3].input();
    assert_eq!(
        resumed_input.get(..replacement_history.len()),
        Some(replacement_history.as_slice()),
        "remote-v2 cold resume should replay persisted replacement history verbatim"
    );
    let post_compact_input = requests[2].input();
    assert_eq!(
        resumed_input.get(..post_compact_input.len()),
        Some(post_compact_input.as_slice()),
        "remote-v2 cold resume should replay the complete post-compaction structured prefix"
    );
    assert_eq!(
        resumed.codex.instruction_sources().await,
        vec![PathUri::from_abs_path(&source)],
        "cold-resumed thread reports the same rewritten source path"
    );

    Ok(())
}

#[path = "compact_program_tests.rs"]
mod program_tests;
