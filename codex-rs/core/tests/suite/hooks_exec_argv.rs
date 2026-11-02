//! Hook contracts for the `exec_argv` tool: hooks see the authoritative argv beside the
//! display command, rewrites must supply argv, and session/network paths keep attribution.

use std::fs;
use std::path::Path;
use std::sync::Arc;
use std::time::Duration;

use anyhow::Context;
use anyhow::Result;
use codex_config::test_support::CloudConfigBundleFixture;
use codex_core::config::Constrained;
use codex_protocol::models::PermissionProfile;
use codex_protocol::permissions::NetworkSandboxPolicy;
use codex_protocol::protocol::AskForApproval;
use codex_protocol::protocol::EventMsg;
use codex_protocol::protocol::Op;
use core_test_support::hooks::trust_discovered_hooks;
use core_test_support::responses::ev_assistant_message;
use core_test_support::responses::ev_completed;
use core_test_support::responses::ev_function_call;
use core_test_support::responses::ev_response_created;
use core_test_support::responses::mount_sse_sequence;
use core_test_support::responses::sse;
use core_test_support::responses::start_mock_server;
use core_test_support::skip_if_no_network;
use core_test_support::test_codex::test_codex;
use core_test_support::wait_for_event;
use pretty_assertions::assert_eq;
use serde_json::Value;
use serde_json::json;
use tempfile::TempDir;
use tokio::time::sleep;
use tokio::time::timeout;

const EXEC_ARGV_MATCHER: &str = "^exec_argv$";
const PRE_TOOL_USE: &str = "pre_tool_use_hook";
const POST_TOOL_USE: &str = "post_tool_use_hook";
const PERMISSION_REQUEST: &str = "permission_request_hook";

/// A command hook matched to `exec_argv`. Its script logs the received payload, plus the hook
/// process cwd as `process_cwd`, to `<name>_log.jsonl`, then runs `tail` to produce its output.
struct ExecArgvHook {
    event: &'static str,
    name: &'static str,
    tail: String,
}

fn write_exec_argv_hooks(home: &Path, hooks: &[ExecArgvHook]) -> Result<()> {
    let mut events = serde_json::Map::new();
    for hook in hooks {
        let script_path = home.join(format!("{}.py", hook.name));
        let log_path = home.join(format!("{}_log.jsonl", hook.name));
        let script = format!(
            r#"import json
from pathlib import Path
import sys

payload = json.load(sys.stdin)
payload["process_cwd"] = str(Path.cwd())
with Path(r"{log_path}").open("a", encoding="utf-8") as handle:
    handle.write(json.dumps(payload) + "\n")
{tail}
"#,
            log_path = log_path.display(),
            tail = hook.tail,
        );
        fs::write(&script_path, script).with_context(|| format!("write {}", hook.name))?;
        events.insert(
            hook.event.to_string(),
            json!([{
                "matcher": EXEC_ARGV_MATCHER,
                "hooks": [{
                    "type": "command",
                    "command": format!("python3 {}", script_path.display()),
                }]
            }]),
        );
    }
    fs::write(
        home.join("hooks.json"),
        json!({ "hooks": events }).to_string(),
    )
    .context("write hooks.json")?;
    Ok(())
}

/// Python that prints `output` as the hook's JSON result.
fn print_hook_output(output: Value) -> String {
    let literal = serde_json::to_string(&output.to_string()).expect("serialize hook output");
    format!("print({literal})")
}

fn write_allow_permission_request_hook(home: &Path) -> Result<()> {
    write_exec_argv_hooks(
        home,
        &[ExecArgvHook {
            event: "PermissionRequest",
            name: PERMISSION_REQUEST,
            tail: print_hook_output(json!({
                "hookSpecificOutput": {
                    "hookEventName": "PermissionRequest",
                    "decision": { "behavior": "allow" }
                }
            })),
        }],
    )
}

fn write_updating_pre_tool_use_hook(home: &Path, updated_input: &Value) -> Result<()> {
    write_exec_argv_hooks(
        home,
        &[ExecArgvHook {
            event: "PreToolUse",
            name: PRE_TOOL_USE,
            tail: print_hook_output(json!({
                "hookSpecificOutput": {
                    "hookEventName": "PreToolUse",
                    "permissionDecision": "allow",
                    "updatedInput": updated_input
                }
            })),
        }],
    )
}

fn write_post_tool_use_context_hook(home: &Path, additional_context: &str) -> Result<()> {
    write_exec_argv_hooks(
        home,
        &[ExecArgvHook {
            event: "PostToolUse",
            name: POST_TOOL_USE,
            tail: print_hook_output(json!({
                "hookSpecificOutput": {
                    "hookEventName": "PostToolUse",
                    "additionalContext": additional_context
                }
            })),
        }],
    )
}

fn write_logging_pre_and_blocking_post_tool_use_hooks(home: &Path, feedback: &str) -> Result<()> {
    let feedback_json =
        serde_json::to_string(feedback).context("serialize post tool use feedback")?;
    write_exec_argv_hooks(
        home,
        &[
            ExecArgvHook {
                event: "PreToolUse",
                name: PRE_TOOL_USE,
                tail: String::new(),
            },
            ExecArgvHook {
                event: "PostToolUse",
                name: POST_TOOL_USE,
                tail: format!("sys.stderr.write({feedback_json} + \"\\n\")\nraise SystemExit(2)"),
            },
        ],
    )
}

fn read_hook_inputs(home: &Path, name: &str) -> Result<Vec<Value>> {
    let log_path = home.join(format!("{name}_log.jsonl"));
    fs::read_to_string(&log_path)
        .with_context(|| format!("read hook log {}", log_path.display()))?
        .lines()
        .filter(|line| !line.trim().is_empty())
        .map(|line| serde_json::from_str(line).context("parse hook log line"))
        .collect()
}

/// Asserts the complete hook-facing tool input, as exec_argv adds `argv` beside `command`.
fn assert_permission_request_hook_payload(
    hook_input: &Value,
    tool_name: &str,
    expected_tool_input: &Value,
) {
    assert_eq!(hook_input["hook_event_name"], "PermissionRequest");
    assert_eq!(hook_input["tool_name"], tool_name);
    assert_eq!(&hook_input["tool_input"], expected_tool_input);
    assert!(hook_input.get("approval_attempt").is_none());
    assert!(hook_input.get("sandbox_permissions").is_none());
    assert!(hook_input.get("additional_permissions").is_none());
    assert!(hook_input.get("justification").is_none());
    assert!(hook_input.get("host").is_none());
    assert!(hook_input.get("protocol").is_none());
}

#[tokio::test]
async fn permission_request_hook_sees_exec_argv_identity_and_authoritative_argv() -> Result<()> {
    skip_if_no_network!(Ok(()));

    let server = start_mock_server().await;
    let call_id = "permissionrequest-exec-argv";
    let literal_args = vec!["permission-approved".to_string()];
    let argv = super::exec_argv_test_helper_argv(/*delay_ms*/ 0, literal_args.clone())?;
    let justification = "run the argv-native permission test helper";
    let args = json!({
        "argv": argv.clone(),
        "sandbox_permissions": "require_escalated",
        "justification": justification,
    });
    let responses = mount_sse_sequence(
        &server,
        vec![
            sse(vec![
                ev_response_created("resp-permission-exec-argv-1"),
                ev_function_call(call_id, "exec_argv", &serde_json::to_string(&args)?),
                ev_completed("resp-permission-exec-argv-1"),
            ]),
            sse(vec![
                ev_response_created("resp-permission-exec-argv-2"),
                ev_assistant_message("msg-permission-exec-argv", "done"),
                ev_completed("resp-permission-exec-argv-2"),
            ]),
        ],
    )
    .await;

    let mut builder = test_codex()
        .with_pre_build_hook(|home| {
            write_allow_permission_request_hook(home)
                .expect("failed to write exec_argv permission request hook fixture");
        })
        .with_config(trust_discovered_hooks)
        .with_config(super::enable_exec_argv);
    let test = builder.build(&server).await?;

    test.submit_turn_with_approval_and_permission_profile(
        "run exec_argv after hook approval",
        AskForApproval::OnRequest,
        PermissionProfile::read_only(),
    )
    .await?;

    let requests = responses.requests();
    assert_eq!(requests.len(), 2);
    let output = requests[1]
        .function_call_output_text(call_id)
        .context("missing exec_argv function output")?;
    assert!(output.contains(&serde_json::to_string(&literal_args)?));

    let hook_inputs = read_hook_inputs(test.codex_home_path(), PERMISSION_REQUEST)?;
    assert_eq!(hook_inputs.len(), 1);
    assert_permission_request_hook_payload(
        &hook_inputs[0],
        "exec_argv",
        &json!({
            "command": codex_shell_command::parse_command::shlex_join(&argv),
            "argv": argv,
            "description": justification,
        }),
    );
    assert!(
        hook_inputs[0].get("tool_use_id").is_none(),
        "PermissionRequest input should not include a tool_use_id"
    );

    Ok(())
}

#[tokio::test]
async fn managed_network_permission_request_preserves_exec_argv_attribution() -> Result<()> {
    skip_if_no_network!(Ok(()));

    let python = which::which("python").or_else(|_| which::which("python3"))?;
    let argv = vec![
        python.to_string_lossy().into_owned(),
        "-c".to_string(),
        "import urllib.request; opener = urllib.request.build_opener(urllib.request.ProxyHandler()); print('OK:' + opener.open('http://codex-network-test.invalid', timeout=2).read().decode(errors='replace'))".to_string(),
    ];
    let command = codex_shell_command::parse_command::shlex_join(&argv);
    let call_id = "permissionrequest-network-exec-argv";

    let server = start_mock_server().await;
    let home = Arc::new(TempDir::new()?);
    fs::write(
        home.path().join("config.toml"),
        r#"default_permissions = "workspace"

[permissions.workspace.filesystem]
":minimal" = "read"

[permissions.workspace.network]
enabled = true
mode = "limited"
allow_local_binding = true
"#,
    )?;
    let args = json!({ "argv": argv.clone(), "workdir": "network-workdir" });
    let responses = mount_sse_sequence(
        &server,
        vec![
            sse(vec![
                ev_response_created("resp-network-hook-1"),
                ev_function_call(call_id, "exec_argv", &serde_json::to_string(&args)?),
                ev_completed("resp-network-hook-1"),
            ]),
            sse(vec![
                ev_response_created("resp-network-hook-2"),
                ev_assistant_message("msg-network-hook", "done"),
                ev_completed("resp-network-hook-2"),
            ]),
        ],
    )
    .await;

    let approval_policy = AskForApproval::OnRequest;
    let permission_profile = PermissionProfile::workspace_write_with(
        &[],
        NetworkSandboxPolicy::Enabled,
        /*exclude_tmpdir_env_var*/ false,
        /*exclude_slash_tmp*/ false,
    );
    let permission_profile_for_config = permission_profile.clone();
    let test = test_codex()
        .with_home(Arc::clone(&home))
        .with_pre_build_hook(|home| {
            write_allow_permission_request_hook(home)
                .expect("failed to write permission request hook test fixture");
        })
        .with_cloud_config_bundle(
            CloudConfigBundleFixture::loader_with_enterprise_requirement(
                r#"
[experimental_network]
enabled = true
allow_local_binding = true
allowed_domains = ["**.invalid"]
review_domains = ["codex-network-test.invalid"]
"#,
            ),
        )
        .with_config(move |config| {
            trust_discovered_hooks(config);
            super::enable_exec_argv(config);
            config.approvals_reviewer = codex_config::types::ApprovalsReviewer::AutoReview;
            config.permissions.approval_policy = Constrained::allow_any(approval_policy);
            config
                .permissions
                .set_permission_profile(permission_profile_for_config)
                .expect("set permission profile");
        })
        .build(&server)
        .await?;
    let execution_cwd = test.workspace_path_uri("network-workdir")?;
    fs::create_dir(execution_cwd.to_abs_path()?).context("create network command workdir")?;

    test.submit_turn_with_approval_and_permission_profile(
        "run the shell command after the network permission hook",
        approval_policy,
        permission_profile,
    )
    .await?;
    timeout(Duration::from_secs(10), async {
        loop {
            if test
                .codex_home_path()
                .join("permission_request_hook_log.jsonl")
                .exists()
            {
                break;
            }
            sleep(Duration::from_millis(100)).await;
        }
    })
    .await
    .expect("expected network approval hook to run");
    assert!(
        timeout(
            Duration::from_secs(2),
            wait_for_event(&test.codex, |event| matches!(
                event,
                EventMsg::ExecApprovalRequest(_)
            ))
        )
        .await
        .is_err(),
        "expected the network approval hook to bypass the approval prompt"
    );

    let hook_inputs = read_hook_inputs(test.codex_home_path(), PERMISSION_REQUEST)?;
    assert_eq!(hook_inputs.len(), 1);
    assert_permission_request_hook_payload(
        &hook_inputs[0],
        "exec_argv",
        &json!({
            "command": command,
            "argv": argv,
            "description": "network-access http://codex-network-test.invalid:80",
        }),
    );
    assert_eq!(
        json!({
            "cwd": hook_inputs[0]["cwd"],
            "process_cwd": hook_inputs[0]["process_cwd"],
        }),
        json!({
            "cwd": execution_cwd.inferred_native_path_string(),
            "process_cwd": dunce::canonicalize(&test.config.cwd)?.to_string_lossy(),
        }),
    );
    let requests = responses.requests();
    assert_eq!(
        requests
            .iter()
            .filter(|request| {
                request.body_json()["client_metadata"]["x-openai-subagent"].as_str()
                    == Some("guardian")
            })
            .count(),
        0
    );
    test.codex.submit(Op::Shutdown {}).await?;
    wait_for_event(&test.codex, |event| {
        matches!(event, EventMsg::ShutdownComplete)
    })
    .await;

    Ok(())
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn exec_argv_background_write_stdin_preserves_post_hook_contract() -> Result<()> {
    skip_if_no_network!(Ok(()));

    let server = start_mock_server().await;
    let start_call_id = "posttooluse-exec-argv-session-start";
    let poll_call_id = "posttooluse-exec-argv-session-poll";
    let literal_args = vec!["exec-argv-session-post-output".to_string()];
    let delay_ms = if cfg!(windows) { 12_000 } else { 750 };
    let argv = super::exec_argv_test_helper_argv(delay_ms, literal_args.clone())?;
    let start_args = json!({
        "argv": argv.clone(),
        "tty": false,
        "yield_time_ms": 250,
    });
    let poll_args = json!({
        "session_id": 1000,
        "chars": "",
        "yield_time_ms": 30_000,
    });
    let feedback = "blocked by exec_argv session post hook";
    let responses = mount_sse_sequence(
        &server,
        vec![
            sse(vec![
                ev_response_created("resp-exec-argv-session-1"),
                ev_function_call(
                    start_call_id,
                    "exec_argv",
                    &serde_json::to_string(&start_args)?,
                ),
                ev_completed("resp-exec-argv-session-1"),
            ]),
            sse(vec![
                ev_response_created("resp-exec-argv-session-2"),
                ev_function_call(
                    poll_call_id,
                    "write_stdin",
                    &serde_json::to_string(&poll_args)?,
                ),
                ev_completed("resp-exec-argv-session-2"),
            ]),
            sse(vec![
                ev_response_created("resp-exec-argv-session-3"),
                ev_assistant_message("msg-exec-argv-session", "done"),
                ev_completed("resp-exec-argv-session-3"),
            ]),
        ],
    )
    .await;
    let mut builder = test_codex()
        .with_pre_build_hook(move |home| {
            write_logging_pre_and_blocking_post_tool_use_hooks(home, feedback)
                .expect("failed to write exec_argv session hook fixture");
        })
        .with_config(trust_discovered_hooks)
        .with_config(super::enable_exec_argv);
    let test = builder.build(&server).await?;

    test.submit_turn("run the argv-native background session")
        .await?;

    let requests = responses.requests();
    assert_eq!(requests.len(), 3);
    assert_eq!(
        requests[2]
            .function_call_output_text(poll_call_id)
            .as_deref(),
        Some(feedback)
    );

    let expected_tool_input = json!({
        "command": codex_shell_command::parse_command::shlex_join(&argv),
        "argv": argv,
    });
    let pre_hook_inputs = read_hook_inputs(test.codex_home_path(), PRE_TOOL_USE)?;
    assert_eq!(pre_hook_inputs.len(), 1);
    assert_eq!(pre_hook_inputs[0]["tool_name"], "exec_argv");
    assert_eq!(pre_hook_inputs[0]["tool_use_id"], start_call_id);
    assert_eq!(pre_hook_inputs[0]["tool_input"], expected_tool_input);

    let post_hook_inputs = read_hook_inputs(test.codex_home_path(), POST_TOOL_USE)?;
    assert_eq!(post_hook_inputs.len(), 1);
    assert_eq!(post_hook_inputs[0]["hook_event_name"], "PostToolUse");
    assert_eq!(post_hook_inputs[0]["tool_name"], "exec_argv");
    assert_eq!(post_hook_inputs[0]["tool_use_id"], start_call_id);
    assert_eq!(post_hook_inputs[0]["tool_input"], expected_tool_input);
    assert!(
        post_hook_inputs[0]["tool_response"]
            .as_str()
            .is_some_and(|output| output.contains(
                &serde_json::to_string(&literal_args).expect("serialize literal arguments")
            )),
        "PostToolUse should receive the completed exec_argv output"
    );

    Ok(())
}

#[tokio::test]
async fn pre_tool_use_rewrites_exec_argv_before_real_execution() -> Result<()> {
    skip_if_no_network!(Ok(()));

    let server = start_mock_server().await;
    let call_id = "pretooluse-exec-argv-rewrite";
    let marker_dir = TempDir::new()?;
    let original_marker = marker_dir.path().join("original");
    let rewritten_marker = marker_dir.path().join("rewritten");
    let python = which::which("python").or_else(|_| which::which("python3"))?;
    let writer = "from pathlib import Path; import sys; Path(sys.argv[1]).write_text(sys.argv[2])";
    let argv_writing = |marker: &Path, content: &str| {
        vec![
            python.to_string_lossy().into_owned(),
            "-c".to_string(),
            writer.to_string(),
            marker.to_string_lossy().into_owned(),
            content.to_string(),
        ]
    };
    let original_argv = argv_writing(&original_marker, "original");
    let rewritten_argv = argv_writing(&rewritten_marker, "rewritten");
    let args = json!({
        "argv": original_argv.clone(),
        "tty": false,
        "yield_time_ms": 1_000,
    });
    let responses = mount_sse_sequence(
        &server,
        vec![
            sse(vec![
                ev_response_created("resp-pre-exec-argv-rewrite-1"),
                ev_function_call(call_id, "exec_argv", &serde_json::to_string(&args)?),
                ev_completed("resp-pre-exec-argv-rewrite-1"),
            ]),
            sse(vec![
                ev_response_created("resp-pre-exec-argv-rewrite-2"),
                ev_assistant_message("msg-pre-exec-argv-rewrite", "done"),
                ev_completed("resp-pre-exec-argv-rewrite-2"),
            ]),
        ],
    )
    .await;
    let updated_input = json!({ "argv": rewritten_argv });
    let mut builder = test_codex()
        .with_pre_build_hook(move |home| {
            write_updating_pre_tool_use_hook(home, &updated_input)
                .expect("failed to write exec_argv rewrite hook fixture");
        })
        .with_config(trust_discovered_hooks)
        .with_config(super::enable_exec_argv);
    let test = builder.build(&server).await?;

    test.submit_turn_with_permission_profile(
        "run the rewritten argv-native command",
        PermissionProfile::Disabled,
    )
    .await?;

    let requests = responses.requests();
    assert_eq!(requests.len(), 2);
    requests[1].function_call_output(call_id);
    assert!(
        !original_marker.exists(),
        "the original argv must not execute after hook rewrite"
    );
    assert_eq!(fs::read_to_string(&rewritten_marker)?, "rewritten");

    let hook_inputs = read_hook_inputs(test.codex_home_path(), PRE_TOOL_USE)?;
    assert_eq!(hook_inputs.len(), 1);
    assert_eq!(hook_inputs[0]["hook_event_name"], "PreToolUse");
    assert_eq!(hook_inputs[0]["tool_name"], "exec_argv");
    assert_eq!(hook_inputs[0]["tool_use_id"], call_id);
    assert_eq!(hook_inputs[0]["tool_input"]["argv"], json!(original_argv));

    Ok(())
}

#[tokio::test]
async fn pre_tool_use_command_only_rewrite_rejects_exec_argv_without_execution() -> Result<()> {
    skip_if_no_network!(Ok(()));

    let server = start_mock_server().await;
    let call_id = "pretooluse-exec-argv-command-only";
    let marker_dir = TempDir::new()?;
    let marker = marker_dir.path().join("command-only-rewrite-ran");
    let python = which::which("python").or_else(|_| which::which("python3"))?;
    let argv = vec![
        python.to_string_lossy().into_owned(),
        "-c".to_string(),
        "from pathlib import Path; import sys; Path(sys.argv[1]).write_text('ran')".to_string(),
        marker.to_string_lossy().into_owned(),
    ];
    let args = json!({
        "argv": argv.clone(),
        "tty": false,
        "yield_time_ms": 1_000,
    });
    let responses = mount_sse_sequence(
        &server,
        vec![
            sse(vec![
                ev_response_created("resp-pre-exec-argv-command-only-1"),
                ev_function_call(call_id, "exec_argv", &serde_json::to_string(&args)?),
                ev_completed("resp-pre-exec-argv-command-only-1"),
            ]),
            sse(vec![
                ev_response_created("resp-pre-exec-argv-command-only-2"),
                ev_assistant_message("msg-pre-exec-argv-command-only", "done"),
                ev_completed("resp-pre-exec-argv-command-only-2"),
            ]),
        ],
    )
    .await;
    let updated_input = json!({ "command": "echo must-not-be-reparsed" });
    let mut builder = test_codex()
        .with_pre_build_hook(move |home| {
            write_updating_pre_tool_use_hook(home, &updated_input)
                .expect("failed to write exec_argv command-only rewrite hook fixture");
        })
        .with_config(trust_discovered_hooks)
        .with_config(super::enable_exec_argv);
    let test = builder.build(&server).await?;

    test.submit_turn_with_permission_profile(
        "reject the command-only argv-native rewrite",
        PermissionProfile::Disabled,
    )
    .await?;

    let requests = responses.requests();
    assert_eq!(requests.len(), 2);
    let output = requests[1]
        .function_call_output_text(call_id)
        .context("missing command-only rewrite rejection")?;
    assert!(
        output.contains("updatedInput for exec_argv without array field `argv`"),
        "unexpected command-only rewrite output: {output:?}"
    );
    assert!(
        !marker.exists(),
        "command-only rewrite must not execute argv"
    );

    let hook_inputs = read_hook_inputs(test.codex_home_path(), PRE_TOOL_USE)?;
    assert_eq!(hook_inputs.len(), 1);
    assert_eq!(hook_inputs[0]["tool_name"], "exec_argv");
    assert_eq!(hook_inputs[0]["tool_input"]["argv"], json!(argv));

    Ok(())
}

#[tokio::test]
async fn post_tool_use_records_exec_argv_identity_and_authoritative_argv() -> Result<()> {
    skip_if_no_network!(Ok(()));

    let server = start_mock_server().await;
    let call_id = "posttooluse-exec-argv";
    let literal_args = vec!["post-argv-output".to_string()];
    let argv = super::exec_argv_test_helper_argv(/*delay_ms*/ 0, literal_args.clone())?;
    let args = json!({
        "argv": argv.clone(),
        "tty": false,
        "yield_time_ms": 1_000,
    });
    let responses = mount_sse_sequence(
        &server,
        vec![
            sse(vec![
                ev_response_created("resp-post-exec-argv-1"),
                ev_function_call(call_id, "exec_argv", &serde_json::to_string(&args)?),
                ev_completed("resp-post-exec-argv-1"),
            ]),
            sse(vec![
                ev_response_created("resp-post-exec-argv-2"),
                ev_assistant_message("msg-post-exec-argv", "done"),
                ev_completed("resp-post-exec-argv-2"),
            ]),
        ],
    )
    .await;
    let mut builder = test_codex()
        .with_pre_build_hook(|home| {
            write_post_tool_use_context_hook(home, "argv post context")
                .expect("failed to write exec_argv post hook fixture");
        })
        .with_config(trust_discovered_hooks)
        .with_config(super::enable_exec_argv);
    let test = builder.build(&server).await?;

    test.submit_turn("run exec_argv with a post hook").await?;

    assert_eq!(responses.requests().len(), 2);
    let hook_inputs = read_hook_inputs(test.codex_home_path(), POST_TOOL_USE)?;
    assert_eq!(hook_inputs.len(), 1);
    assert_eq!(hook_inputs[0]["hook_event_name"], "PostToolUse");
    assert_eq!(hook_inputs[0]["tool_name"], "exec_argv");
    assert_eq!(hook_inputs[0]["tool_use_id"], call_id);
    assert_eq!(hook_inputs[0]["tool_input"]["argv"], json!(argv));
    assert!(
        hook_inputs[0]["tool_response"]
            .as_str()
            .is_some_and(|output| output.contains(
                &serde_json::to_string(&literal_args).expect("serialize literal arguments")
            ))
    );

    Ok(())
}
