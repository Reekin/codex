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
use codex_protocol::protocol::EventMsg;
use codex_protocol::protocol::ItemCompletedEvent;
use codex_protocol::protocol::ItemStartedEvent;
use codex_protocol::protocol::Op;
use codex_protocol::protocol::ThreadSettingsOverrides;
use codex_protocol::user_input::UserInput;
use codex_utils_absolute_path::AbsolutePathBuf;
use codex_utils_path_uri::PathUri;
use core_test_support::PathBufExt;
use core_test_support::responses;
use core_test_support::responses::mount_models_once;
use core_test_support::skip_if_no_network;
use core_test_support::test_codex::local_selections;
use core_test_support::test_codex::test_codex;
use core_test_support::test_codex::turn_permission_fields;
use core_test_support::wait_for_event;
use std::path::PathBuf;

use core_test_support::responses::ev_assistant_message;
use core_test_support::responses::ev_completed;
use core_test_support::responses::ev_completed_with_tokens;
use core_test_support::responses::mount_compact_json_once;
use core_test_support::responses::mount_compact_response_sequence;
use core_test_support::responses::mount_sse_once;
use core_test_support::responses::mount_sse_once_match;
use core_test_support::responses::mount_sse_sequence;
use core_test_support::responses::sse;
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

pub(super) const FIRST_REPLY: &str = "FIRST_REPLY";
const SECOND_LARGE_REPLY: &str = "SECOND_LARGE_REPLY";
const FIRST_AUTO_SUMMARY: &str = "FIRST_AUTO_SUMMARY";
const SECOND_AUTO_SUMMARY: &str = "SECOND_AUTO_SUMMARY";
const FINAL_REPLY: &str = "FINAL_REPLY";
const PRETURN_CONTEXT_DIFF_CWD: &str = "/tmp/PRETURN_CONTEXT_DIFF_CWD";
const GLOBAL_AGENTS_FILENAME: &str = "AGENTS.md";
const NEW_GLOBAL_INSTRUCTIONS: &str = "new global instructions";
const OLD_GLOBAL_INSTRUCTIONS: &str = "old global instructions";
const REMOTE_V2_SUMMARY: &str = "global-instructions-remote-v2-summary";

pub(super) fn allow_echo_commands(home: &Path) {
    let rules_dir = home.join("rules");
    fs::create_dir_all(&rules_dir).expect("create exec policy rules directory");
    fs::write(
        rules_dir.join("default.rules"),
        r#"prefix_rule(pattern=["echo"], decision="allow")"#,
    )
    .expect("write echo exec policy rule");
}

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
        environments: Some(local_selections(cwd.abs())),
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
        .find(|model| model.slug == slug)
        .expect("model missing from models.json");
    model_info.context_window = Some(context_window);
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

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn auto_compact_runs_after_resume_when_token_usage_is_over_limit() {
    skip_if_no_network!();

    let server = start_mock_server().await;

    let limit = 200_000;
    let over_limit_tokens = 250_000;
    let remote_summary = "REMOTE_COMPACT_SUMMARY";

    let compacted_history = vec![
        codex_protocol::models::ResponseItem::Message {
            id: None,
            role: "assistant".to_string(),
            content: vec![codex_protocol::models::ContentItem::OutputText {
                text: remote_summary.to_string(),
            }],
            phase: None,
            internal_chat_message_metadata_passthrough: None,
        },
        codex_protocol::models::ResponseItem::Compaction {
            id: None,
            encrypted_content: "ENCRYPTED_COMPACTION_SUMMARY".to_string(),
            internal_chat_message_metadata_passthrough: None,
        },
    ];
    let compact_mock =
        mount_compact_json_once(&server, serde_json::json!({ "output": compacted_history })).await;

    let mut builder = test_codex().with_config(move |config| {
        set_test_compact_prompt(config);
        config.model_auto_compact_token_limit = Some(limit);
        let _ = config.features.disable(Feature::RemoteCompactionV2);
    });
    let initial = builder.build(&server).await.unwrap();

    // A single over-limit completion should not auto-compact until the next user message.
    mount_sse_once(
        &server,
        sse(vec![
            ev_assistant_message("m1", FIRST_REPLY),
            ev_completed_with_tokens("r1", over_limit_tokens),
        ]),
    )
    .await;
    initial.submit_turn("OVER_LIMIT_TURN").await.unwrap();

    assert!(
        compact_mock.requests().is_empty(),
        "remote compaction should not run before the next user message"
    );

    let mut resume_builder = test_codex().with_config(move |config| {
        set_test_compact_prompt(config);
        config.model_auto_compact_token_limit = Some(limit);
        let _ = config.features.disable(Feature::RemoteCompactionV2);
    });
    let resumed = resume_builder.restart(&server, &initial).await.unwrap();

    let follow_up_user = "AFTER_RESUME_USER";
    let sse_follow_up = sse(vec![
        ev_assistant_message("m2", FINAL_REPLY),
        ev_completed("r2"),
    ]);

    let follow_up_matcher = move |req: &wiremock::Request| {
        let body = std::str::from_utf8(&req.body).unwrap_or("");
        body.contains(follow_up_user) && body.contains(remote_summary)
    };
    mount_sse_once_match(&server, follow_up_matcher, sse_follow_up).await;

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

    let compact_requests = compact_mock.requests();
    assert_eq!(
        compact_requests.len(),
        1,
        "remote compaction should run once after resume"
    );
    assert_eq!(
        compact_requests[0].path(),
        "/v1/responses/compact",
        "remote compaction should hit the compact endpoint"
    );
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn pre_sampling_legacy_remote_compact_falls_back_after_previous_model_invalid_request() {
    skip_if_no_network!();

    let server = MockServer::start().await;
    let retired_model = "gpt-5.6";
    let previous_model_family = "gpt-5.6";
    let next_model = "gpt-5.5";
    let mut previous_model_info =
        model_info_with_context_window("gpt-5.4", /*context_window*/ 273_000);
    previous_model_info.slug = previous_model_family.to_string();
    let mut next_model_info =
        model_info_with_context_window("gpt-5.4", /*context_window*/ 125_000);
    next_model_info.slug = next_model.to_string();

    let models_mock = mount_models_once(
        &server,
        ModelsResponse {
            models: vec![previous_model_info, next_model_info],
        },
    )
    .await;
    let request_log = mount_sse_sequence(
        &server,
        vec![
            sse(vec![
                ev_assistant_message("m1", "before switch"),
                ev_completed_with_tokens("r1", /*total_tokens*/ 120_000),
            ]),
            sse(vec![
                ev_assistant_message("m3", "after switch"),
                ev_completed_with_tokens("r3", /*total_tokens*/ 100),
            ]),
        ],
    )
    .await;
    let compact_request_log = mount_compact_response_sequence(
        &server,
        vec![
            invalid_request_response("previous-model compaction was rejected"),
            wiremock::ResponseTemplate::new(/*status*/ 200)
                .insert_header("content-type", "application/json")
                .set_body_json(json!({
                    "output": [{
                        "type": "compaction",
                        "encrypted_content": "DOWNSHIFT_SUMMARY",
                    }],
                })),
        ],
    )
    .await;

    let model_provider = openai_model_provider(&server);
    let mut builder = test_codex()
        .with_auth(CodexAuth::create_dummy_chatgpt_auth_for_testing())
        .with_model(retired_model)
        .with_config(move |config| {
            config.model_provider = model_provider;
            set_test_compact_prompt(config);
            let _ = config.features.disable(Feature::RemoteCompactionV2);
        });
    let test = builder.build(&server).await.expect("build test codex");

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
    let compact_requests = compact_request_log.requests();
    assert_eq!(models_mock.requests().len(), 1);
    assert_eq!(requests.len(), 2);
    assert_eq!(compact_requests.len(), 2);
    assert_eq!(
        requests[0].body_json()["model"].as_str(),
        Some(retired_model)
    );
    assert_eq!(
        compact_requests[0].body_json()["model"].as_str(),
        Some(retired_model)
    );
    assert_eq!(
        compact_requests[1].body_json()["model"].as_str(),
        Some(next_model)
    );
    assert_eq!(requests[1].body_json()["model"].as_str(), Some(next_model));
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn remote_v2_compaction_keeps_creation_time_instructions_after_same_path_mutation()
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
        .with_auth(CodexAuth::create_dummy_chatgpt_auth_for_testing())
        .with_config(|config| {
            let _ = config.features.enable(Feature::RemoteCompactionV2);
        });
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

    // Assert the compact request, installed replacement history, and follow-up all keep the
    // creation-time item despite the file-backed source now containing new text.
    let requests = response_mock.requests();
    assert_eq!(requests.len(), 3);
    let old_fragment = expected_instruction_fragment(OLD_GLOBAL_INSTRUCTIONS);
    assert_single_instruction_fragment(&requests[0], &old_fragment);
    assert_single_instruction_fragment(&requests[1], &old_fragment);
    assert_single_instruction_fragment(&requests[2], &old_fragment);
    assert_eq!(
        requests[1].input().last(),
        Some(&json!({"type": "compaction_trigger"})),
        "remote-v2 compact request should append exactly one compaction trigger"
    );
    let rollout_path = test.codex.rollout_path().expect("rollout path");
    let replacement_history = replacement_history_from_rollout(&rollout_path)?;
    assert_eq!(
        instruction_fragments_in_items(&replacement_history),
        Vec::<String>::new(),
        "remote-v2 replacement history currently omits the global-instruction fragment"
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
        .with_auth(CodexAuth::create_dummy_chatgpt_auth_for_testing())
        .with_config(move |config| {
            config.cwd = resumed_cwd;
            let _ = config.features.enable(Feature::RemoteCompactionV2);
        });
    let resumed = resume_builder
        .resume(&server, Arc::clone(&home), rollout_path)
        .await?;
    resumed
        .submit_turn("after remote v2 compaction cold resume")
        .await?;

    // Cold resume replays the persisted old context, then appends the newly loaded instructions as
    // an explicit replacement.
    let requests = response_mock.requests();
    assert_eq!(requests.len(), 4);
    let replacement_fragment = expected_instruction_fragment(&format!(
        "These AGENTS.md instructions replace all previously provided AGENTS.md instructions.\n\n{NEW_GLOBAL_INSTRUCTIONS}"
    ));
    assert_eq!(
        instruction_fragments(&requests[3]),
        vec![old_fragment.clone(), replacement_fragment]
    );
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
