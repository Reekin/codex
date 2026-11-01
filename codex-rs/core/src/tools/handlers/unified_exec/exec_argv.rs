//! `exec_argv`: unified exec that passes the model's argv directly to process creation.
//!
//! Parsing, hook rewrite, and diagnostics live here; cwd, permission, approval, and
//! process plumbing are shared with `exec_command` through `PreparedExecCommand`.

use std::path::Path;
#[cfg(any(windows, test))]
use std::path::PathBuf;
use std::sync::Arc;

use codex_protocol::models::AdditionalPermissionProfile;
use codex_tools::ToolName;
use codex_tools::ToolSpec;
use codex_utils_output_truncation::TruncationPolicy;
use codex_utils_output_truncation::approx_token_count;
use codex_utils_path_uri::PathUri;
use codex_utils_string::truncate_middle_chars;
use serde::Deserialize;
use serde_json::Value;

use crate::function_tool::FunctionCallError;
use crate::maybe_emit_implicit_skill_invocation;
use crate::sandboxing::SandboxPermissions;
use crate::tools::context::ExecCommandToolOutput;
use crate::tools::context::ToolEnvironments;
use crate::tools::context::ToolInvocation;
use crate::tools::context::ToolPayload;
use crate::tools::context::boxed_tool_output;
use crate::tools::handlers::ToolEnvironmentArgs;
use crate::tools::handlers::apply_patch::intercept_apply_patch;
use crate::tools::handlers::function_tool_environment;
use crate::tools::handlers::parse_arguments;
use crate::tools::handlers::resolve_sandbox_permissions;
use crate::tools::handlers::rewrite_function_arguments;
use crate::tools::handlers::selected_executor;
use crate::tools::registry::CoreToolRuntime;
use crate::tools::registry::PostToolUsePayload;
use crate::tools::registry::PreToolUsePayload;
use crate::tools::registry::ToolExecutor;
use crate::tools::sandboxing::ExecHookMetadata;
use crate::unified_exec::ExecCommandRequest;
use crate::unified_exec::UnifiedExecContext;
use crate::unified_exec::UnifiedExecError;
use crate::unified_exec::UnifiedExecProcessManager;
use crate::unified_exec::generate_chunk_id;
use crate::unified_exec::output::CapturedStream;

use super::super::shell_spec::CommandToolOptions;
use super::super::shell_spec::create_exec_argv_tool_with_environment_id;
use super::CommandLaunch;
use super::ExecCommandWorkdirArgs;
use super::ResolvedCommand;
use super::default_exec_yield_time_ms;
use super::exec_command::EXEC_COMMAND_REJECTION_MAX_BYTES;
use super::exec_command::emit_unified_exec_tty_metric;
use super::post_unified_exec_tool_use_payload;
use super::preparation::ExecArgumentsForEnvironment;
use super::preparation::PreparedExecCommand;
use super::preparation::parse_exec_arguments;
use super::preparation::reject_disabled_tty;
use super::preparation::resolve_exec_permissions;

#[derive(Debug, Deserialize)]
struct ExecArgvArgs {
    argv: Vec<String>,
    #[serde(default)]
    tty: bool,
    #[serde(default = "default_exec_yield_time_ms")]
    yield_time_ms: u64,
    #[serde(default)]
    max_output_tokens: Option<usize>,
    #[serde(default)]
    sandbox_permissions: Option<SandboxPermissions>,
    #[serde(default)]
    additional_permissions: Option<AdditionalPermissionProfile>,
    #[serde(default)]
    justification: Option<String>,
    #[serde(default)]
    prefix_rule: Option<Vec<String>>,
}

/// The only input PreToolUse needs, so hooks see argv even when permission fields
/// cannot be resolved without an executor cwd.
#[derive(Deserialize)]
struct ExecArgvHookArgs {
    argv: Vec<String>,
}

#[derive(Clone, Copy)]
pub(crate) struct ExecArgvHandlerOptions {
    pub(crate) allow_tty: bool,
    pub(crate) exec_permission_approvals_enabled: bool,
    pub(crate) include_environment_id: bool,
}

pub struct ExecArgvHandler {
    options: ExecArgvHandlerOptions,
}

impl Default for ExecArgvHandler {
    fn default() -> Self {
        Self {
            options: ExecArgvHandlerOptions {
                allow_tty: true,
                exec_permission_approvals_enabled: false,
                include_environment_id: false,
            },
        }
    }
}

impl ExecArgvHandler {
    pub(crate) fn new(options: ExecArgvHandlerOptions) -> Self {
        Self { options }
    }

    async fn handle_call(
        &self,
        invocation: ToolInvocation,
    ) -> Result<Box<dyn crate::tools::context::ToolOutput>, FunctionCallError> {
        let ToolInvocation {
            environments,
            session,
            step_context,
            cancellation_token,
            tracker,
            call_id,
            payload,
            ..
        } = invocation;
        let ToolPayload::Function { arguments } = payload else {
            return Err(FunctionCallError::RespondToModel(
                "exec_argv handler received unsupported payload".to_string(),
            ));
        };

        let manager: &UnifiedExecProcessManager = &session.services.unified_exec_manager;
        let context = UnifiedExecContext::new(
            session.clone(),
            step_context.clone(),
            cancellation_token,
            call_id.clone(),
        );
        let turn_environment = selected_executor(&environments, &step_context.environments)?;
        let ExecArgumentsForEnvironment {
            args,
            cwd,
            native_cwd,
        } = parse_exec_arguments::<ExecArgvArgs>(&arguments, &step_context, turn_environment)?;
        reject_disabled_tty(&session, args.tty)?;
        let sandbox_permissions =
            resolve_sandbox_permissions(args.sandbox_permissions, args.justification.as_deref())?;
        let resolved = ResolvedCommand {
            command: validate_argv(args.argv)?,
            launch: CommandLaunch::Argv,
        };
        let hook_metadata = resolved.hook_metadata();
        maybe_emit_implicit_skill_invocation(
            &session,
            &step_context,
            hook_metadata.command(),
            &cwd,
            native_cwd.as_ref(),
            &turn_environment.selection.environment_id,
        )
        .await;
        let permissions = resolve_exec_permissions(
            &session,
            &step_context,
            turn_environment,
            &cwd,
            sandbox_permissions,
            args.additional_permissions,
        )?;
        let prepared = PreparedExecCommand {
            turn_environment: turn_environment.clone(),
            resolved,
            cwd,
            permissions,
            justification: args.justification,
            prefix_rule: args.prefix_rule,
            tty: args.tty,
            yield_time_ms: args.yield_time_ms,
            max_output_tokens: args.max_output_tokens,
            completion_timeout: None,
        };
        let max_output_tokens = prepared.max_output_tokens;
        let truncation_policy: TruncationPolicy =
            step_context.settings.model_info.truncation_policy.into();
        let fs = prepared.turn_environment.environment.get_filesystem();
        let intercepted_patch = intercept_apply_patch(
            &prepared.resolved.command,
            &prepared.cwd,
            fs.as_ref(),
            prepared.turn_environment.clone(),
            context.session.clone(),
            Arc::clone(&context.step_context),
            context.cancellation_token.clone(),
            Some(&tracker),
            &context.call_id,
            "exec_argv",
        )
        .await;
        if let Some(output) = intercepted_patch? {
            return Ok(boxed_tool_output(ExecCommandToolOutput {
                event_call_id: String::new(),
                chunk_id: String::new(),
                wall_time: std::time::Duration::ZERO,
                raw_output: CapturedStream {
                    text: output.into_text(),
                    omitted_bytes: 0,
                },
                truncation_policy,
                max_output_tokens,
                process_id: None,
                exit_code: None,
                original_token_count: None,
                hook_metadata: None,
                tool_cwd: None,
            }));
        }

        emit_unified_exec_tty_metric(&step_context.session_telemetry, prepared.tty);
        crate::tools::lifecycle::notify_command_start(
            context.session.as_ref(),
            context.step_context.turn.as_ref(),
            &context.call_id,
            &prepared.resolved.command,
            &prepared.cwd,
            fs.as_ref(),
        )
        .await;
        // Preparation can be cancelled. Reserve a process only once it is done.
        let process_id = manager.allocate_process_id().await;
        let argv0 = prepared.resolved.command[0].clone();
        let tool_cwd =
            (!prepared.turn_environment.environment.is_remote()).then(|| prepared.cwd.clone());
        let request = ExecCommandRequest {
            prepared,
            process_id,
            network: context.step_context.turn.network.clone(),
        };
        match manager.exec_command(request, &context).await {
            Ok(response) => Ok(boxed_tool_output(response)),
            Err(UnifiedExecError::SandboxDenied {
                output,
                original_token_count,
                output_omitted_bytes,
                ..
            }) => {
                let output_text = output.aggregated_output.text;
                let original_token_count =
                    original_token_count.unwrap_or_else(|| approx_token_count(&output_text));
                Ok(boxed_tool_output(ExecCommandToolOutput {
                    event_call_id: context.call_id.clone(),
                    chunk_id: generate_chunk_id(),
                    wall_time: output.duration,
                    raw_output: CapturedStream {
                        text: output_text,
                        omitted_bytes: output_omitted_bytes.map_or(0, std::num::NonZeroUsize::get),
                    },
                    truncation_policy,
                    max_output_tokens,
                    // Sandbox denial is terminal, so there is no live
                    // process for write_stdin to resume.
                    process_id: None,
                    exit_code: Some(output.exit_code),
                    original_token_count: Some(original_token_count),
                    hook_metadata: Some(hook_metadata),
                    tool_cwd,
                }))
            }
            Err(err) => Err(FunctionCallError::RespondToModel(format_exec_argv_error(
                hook_metadata.command(),
                &argv0,
                native_cwd.as_deref(),
                &err,
            ))),
        }
    }
}

impl ToolExecutor<ToolInvocation> for ExecArgvHandler {
    fn tool_name(&self) -> ToolName {
        ToolName::plain("exec_argv")
    }

    fn spec(&self) -> ToolSpec {
        let mut spec = create_exec_argv_tool_with_environment_id(
            CommandToolOptions {
                exec_permission_approvals_enabled: self.options.exec_permission_approvals_enabled,
            },
            self.options.include_environment_id,
        );
        if !self.options.allow_tty
            && let ToolSpec::Function(spec) = &mut spec
        {
            spec.parameters
                .properties
                .get_or_insert_default()
                .remove("tty");
        }
        spec
    }

    fn supports_parallel_tool_calls(&self) -> bool {
        true
    }

    fn handle<'a>(&'a self, invocation: ToolInvocation) -> codex_tools::ToolExecutorFuture<'a>
    where
        ToolInvocation: 'a,
    {
        Box::pin(self.handle_call(invocation))
    }
}

impl CoreToolRuntime for ExecArgvHandler {
    fn environments_for_input(
        &self,
        invocation: &ToolInvocation,
    ) -> Result<ToolEnvironments, FunctionCallError> {
        function_tool_environment(invocation, |args: ToolEnvironmentArgs| args.environment_id)
    }

    fn matches_kind(&self, payload: &ToolPayload) -> bool {
        matches!(payload, ToolPayload::Function { .. })
    }

    fn pre_tool_use_payload(&self, invocation: &ToolInvocation) -> Option<PreToolUsePayload> {
        let ToolPayload::Function { arguments } = &invocation.payload else {
            return None;
        };
        let args = parse_arguments::<ExecArgvHookArgs>(arguments).ok()?;
        let metadata = ExecHookMetadata::argv(&validate_argv(args.argv).ok()?);
        Some(PreToolUsePayload {
            tool_name: metadata.tool_name,
            tool_input: metadata.tool_input,
        })
    }

    fn pre_tool_use_cwd(
        &self,
        invocation: &ToolInvocation,
    ) -> Result<Option<PathUri>, FunctionCallError> {
        let ToolPayload::Function { arguments } = &invocation.payload else {
            return Ok(None);
        };
        let args: ExecCommandWorkdirArgs = parse_arguments(arguments)?;
        let environment = selected_executor(
            &invocation.environments,
            &invocation.step_context.environments,
        )?;
        let cwd = args.resolve_cwd(environment)?;
        // Hook scripts execute on the host, so remote paths cannot become the hook cwd.
        Ok((!environment.environment.is_remote()).then_some(cwd))
    }

    fn with_updated_hook_input(
        &self,
        mut invocation: ToolInvocation,
        updated_input: Value,
    ) -> Result<ToolInvocation, FunctionCallError> {
        let ToolPayload::Function { arguments } = invocation.payload else {
            return Err(FunctionCallError::RespondToModel(
                "hook input rewrite received unsupported exec_argv payload".to_string(),
            ));
        };
        let argv = updated_hook_argv(&updated_input)?;
        invocation.payload = ToolPayload::Function {
            arguments: rewrite_function_arguments(&arguments, "exec_argv", |arguments| {
                arguments.insert("argv".to_string(), serde_json::json!(argv));
            })?,
        };
        Ok(invocation)
    }

    fn post_tool_use_payload(
        &self,
        invocation: &ToolInvocation,
        result: &dyn crate::tools::context::ToolOutput,
    ) -> Option<PostToolUsePayload> {
        post_unified_exec_tool_use_payload(invocation, result)
    }
}

fn validate_argv(argv: Vec<String>) -> Result<Vec<String>, FunctionCallError> {
    let Some(program) = argv.first() else {
        return Err(FunctionCallError::RespondToModel(
            "exec_argv requires a non-empty argv array.".to_string(),
        ));
    };
    if program.is_empty() {
        return Err(FunctionCallError::RespondToModel(
            "exec_argv requires argv[0] to name a program.".to_string(),
        ));
    }
    if argv.iter().any(|arg| arg.contains('\0')) {
        return Err(FunctionCallError::RespondToModel(
            "exec_argv arguments must not contain NUL bytes.".to_string(),
        ));
    }
    Ok(argv)
}

/// Hooks change execution only through `updatedInput.argv`; a rendered command
/// is never parsed back into argv.
fn updated_hook_argv(updated_input: &Value) -> Result<Vec<String>, FunctionCallError> {
    let Some(Value::Array(items)) = updated_input.get("argv") else {
        return Err(FunctionCallError::RespondToModel(
            "hook returned updatedInput for exec_argv without array field `argv`".to_string(),
        ));
    };
    let argv = items
        .iter()
        .map(|item| {
            item.as_str().map(str::to_string).ok_or_else(|| {
                FunctionCallError::RespondToModel(
                    "hook returned updatedInput.argv for exec_argv with a non-string argument"
                        .to_string(),
                )
            })
        })
        .collect::<Result<Vec<_>, _>>()?;
    validate_argv(argv)
}

fn format_exec_argv_error(
    display_command: &str,
    argv0: &str,
    cwd: Option<&Path>,
    err: &UnifiedExecError,
) -> String {
    let mut message = truncate_middle_chars(
        &format!("exec_argv failed for `{display_command}`: {err:?}"),
        EXEC_COMMAND_REJECTION_MAX_BYTES,
    );
    if let UnifiedExecError::CreateProcess {
        message: create_process_message,
    } = err
        && create_process_error_may_be_program_lookup(create_process_message)
        && let Some(hint) = cwd.and_then(|cwd| windows_pathext_resolution_hint(argv0, cwd))
    {
        message.push_str("\n\n");
        message.push_str(&hint);
    }
    message
}

fn create_process_error_may_be_program_lookup(message: &str) -> bool {
    let lowercase = message.to_ascii_lowercase();
    lowercase.contains("program not found")
        || lowercase.contains("not found")
        || lowercase.contains("os error 2")
        || message.contains("找不到指定的文件")
}

/// Reports exact PATHEXT candidates for a bare program name. Diagnostics never
/// change or retry the executed argv.
#[cfg(windows)]
fn windows_pathext_resolution_hint(program: &str, cwd: &Path) -> Option<String> {
    let path_entries = std::env::var_os("PATH")
        .map(|path| std::env::split_paths(&path).collect::<Vec<_>>())
        .unwrap_or_default();
    let pathext = std::env::var("PATHEXT").unwrap_or_else(|_| ".COM;.EXE;.BAT;.CMD".to_string());
    let candidates = windows_pathext_candidates(program, cwd, path_entries, &pathext);
    if candidates.is_empty() {
        return None;
    }
    let formatted_candidates = candidates
        .iter()
        .take(5)
        .map(|path| format!("`{}`", path.display()))
        .collect::<Vec<_>>()
        .join(", ");
    Some(format!(
        "Windows note: `exec_argv` does not ask a shell to resolve PATHEXT entries for argv[0]. Found candidate path(s): {formatted_candidates}. For a native executable, use its full path with extension as argv[0] to fix program lookup. For .cmd/.bat/.ps1 scripts or shims, use exec_command with the appropriate shell; a full script path does not remove its shell semantics. Alternatively, launch the underlying native executable directly, for example node script.js. An explicitly launched shell or interpreter still interprets its command or code arguments."
    ))
}

#[cfg(not(windows))]
fn windows_pathext_resolution_hint(_program: &str, _cwd: &Path) -> Option<String> {
    None
}

#[cfg(any(windows, test))]
pub(super) fn windows_pathext_candidates(
    program: &str,
    cwd: &Path,
    path_entries: impl IntoIterator<Item = PathBuf>,
    pathext: &str,
) -> Vec<PathBuf> {
    let program_path = Path::new(program.trim());
    if program_path.as_os_str().is_empty() || program_path.extension().is_some() {
        return Vec::new();
    }
    let extensions = pathext
        .split(';')
        .map(str::trim)
        .filter(|extension| extension.starts_with('.') && extension.len() > 1)
        .collect::<Vec<_>>();
    if extensions.is_empty() {
        return Vec::new();
    }
    let has_path_separator = program.contains('\\') || program.contains('/');
    let bases = if has_path_separator {
        vec![if program_path.is_absolute() {
            program_path.to_path_buf()
        } else {
            cwd.join(program_path)
        }]
    } else {
        path_entries
            .into_iter()
            .map(|path_entry| path_entry.join(program))
            .collect()
    };
    let mut candidates = Vec::new();
    for base in bases {
        for extension in &extensions {
            let Some(mut file_name) = base.file_name().map(std::ffi::OsStr::to_os_string) else {
                continue;
            };
            file_name.push(extension);
            let mut candidate = base.clone();
            candidate.set_file_name(file_name);
            if candidate.is_file() && !candidates.contains(&candidate) {
                candidates.push(candidate);
            }
        }
    }
    candidates
}
