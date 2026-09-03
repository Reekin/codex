use std::path::Path;
#[cfg(any(windows, test))]
use std::path::PathBuf;
use std::sync::Arc;

use codex_features::Feature;
use codex_sandboxing::SandboxManager;
use codex_sandboxing::SandboxType;
use codex_sandboxing::SandboxablePreference;
use codex_tools::ToolName;
use codex_tools::ToolSpec;
use codex_utils_output_truncation::approx_token_count;
use codex_utils_path_uri::PathConvention;
use serde_json::Value;

use crate::exec_policy::prompt_is_rejected_by_policy;
use crate::function_tool::FunctionCallError;
use crate::maybe_emit_implicit_skill_invocation;
use crate::tools::context::ExecCommandToolOutput;
use crate::tools::context::ToolInvocation;
use crate::tools::context::ToolPayload;
use crate::tools::context::boxed_tool_output;
use crate::tools::handlers::apply_granted_turn_permissions;
use crate::tools::handlers::apply_patch::intercept_apply_patch;
use crate::tools::handlers::file_system_sandbox_policy_context_for_cwd;
use crate::tools::handlers::implicit_granted_permissions;
use crate::tools::handlers::normalize_and_validate_additional_permissions;
use crate::tools::handlers::parse_arguments;
use crate::tools::handlers::parse_arguments_with_base_path;
use crate::tools::handlers::resolve_sandbox_permissions;
use crate::tools::handlers::resolve_tool_environment;
use crate::tools::handlers::rewrite_function_arguments;
use crate::tools::handlers::shell_spec::CommandToolOptions;
use crate::tools::handlers::shell_spec::create_exec_argv_tool_with_environment_id;
use crate::tools::hook_names::HookToolName;
use crate::tools::registry::CoreToolRuntime;
use crate::tools::registry::PostToolUsePayload;
use crate::tools::registry::PreToolUsePayload;
use crate::tools::registry::ToolExecutor;
use crate::tools::sandboxing::PermissionRequestPayload;
use crate::unified_exec::ExecCommandRequest;
use crate::unified_exec::UnifiedExecContext;
use crate::unified_exec::UnifiedExecError;
use crate::unified_exec::UnifiedExecLaunchMode;
use crate::unified_exec::UnifiedExecProcessManager;
use crate::unified_exec::generate_chunk_id;

use super::ExecArgvArgs;
use super::ExecCommandEnvironmentArgs;
use super::exec_command::emit_unified_exec_tty_metric;
use super::post_unified_exec_tool_use_payload;
use super::shell_mode_for_environment;

#[derive(Clone, Copy)]
pub(crate) struct ExecArgvHandlerOptions {
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
            session,
            turn,
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
            call_id,
        );
        let environment_args: ExecCommandEnvironmentArgs = parse_arguments(&arguments)?;
        let Some(turn_environment) = resolve_tool_environment(
            &step_context.environments,
            environment_args.environment_id.as_deref(),
        )?
        else {
            return Err(FunctionCallError::RespondToModel(
                "unified exec is unavailable in this session".to_string(),
            ));
        };
        let native_environment_cwd = turn_environment.cwd().clone();
        let cwd = environment_args
            .workdir
            .as_deref()
            .filter(|workdir| !workdir.is_empty())
            .map_or_else(
                || Ok(native_environment_cwd.clone()),
                |workdir| native_environment_cwd.join(workdir),
            )
            .map_err(|err| FunctionCallError::RespondToModel(err.to_string()))?;
        let environment = Arc::clone(&turn_environment.environment);
        let fs = environment.get_filesystem();
        let requires_host_native_cwd = !environment.is_remote()
            && SandboxManager::new().select_initial(
                turn_environment.permission_profile(),
                SandboxablePreference::Auto,
                turn_environment.config().windows_sandbox_level,
                turn.network.is_some(),
            ) != SandboxType::None;
        let cwd_uses_native_convention =
            cwd.infer_path_convention() == Some(PathConvention::native());
        let native_cwd = match cwd.to_abs_path() {
            Ok(cwd) if cwd_uses_native_convention => Some(cwd),
            _ if !requires_host_native_cwd => None,
            Err(err) => return Err(FunctionCallError::RespondToModel(err.to_string())),
            Ok(_) => {
                return Err(FunctionCallError::RespondToModel(format!(
                    "path URI `{cwd}` does not use the host's native {} path convention",
                    PathConvention::native()
                )));
            }
        };
        let args: ExecArgvArgs = match native_cwd.as_ref() {
            Some(native_cwd) => parse_arguments_with_base_path(&arguments, native_cwd)?,
            None => parse_arguments(&arguments)?,
        };
        let command = validate_argv(args.argv)?;
        let hook_command = codex_shell_command::parse_command::shlex_join(&command);
        let hook_metadata =
            PermissionRequestPayload::exec_argv(hook_command.clone(), command.clone());
        maybe_emit_implicit_skill_invocation(
            session.as_ref(),
            context.step_context.turn.as_ref(),
            &hook_command,
            &cwd,
            native_cwd.as_ref(),
            &turn_environment.selection.environment_id,
        )
        .await;
        let ExecArgvArgs {
            tty,
            yield_time_ms,
            max_output_tokens,
            sandbox_permissions,
            additional_permissions,
            justification,
            prefix_rule,
            ..
        } = args;
        let sandbox_permissions =
            resolve_sandbox_permissions(sandbox_permissions, justification.as_deref())?;
        let process_id = manager.allocate_process_id().await;
        let exec_permission_approvals_enabled =
            session.features().enabled(Feature::ExecPermissionApprovals);
        let requested_additional_permissions = additional_permissions.clone();
        let sandbox_context =
            turn_environment.sandbox_context(/*additional_permissions*/ None);
        let Some(permission_context) =
            file_system_sandbox_policy_context_for_cwd(&sandbox_context, &cwd)
        else {
            manager.release_process_id(process_id).await;
            return Err(FunctionCallError::RespondToModel(
                "selected environment sandbox context is missing cwd".to_string(),
            ));
        };
        let effective_additional_permissions = apply_granted_turn_permissions(
            context.session.as_ref(),
            turn_environment,
            &cwd,
            sandbox_permissions,
            additional_permissions,
        )
        .await;
        let additional_permissions_allowed = exec_permission_approvals_enabled
            || (session.features().enabled(Feature::RequestPermissionsTool)
                && effective_additional_permissions.permissions_preapproved);
        let approval_policy = context.step_context.settings.approval_policy();
        if effective_additional_permissions
            .sandbox_permissions
            .requests_sandbox_override()
            && !effective_additional_permissions.permissions_preapproved
            && prompt_is_rejected_by_policy(approval_policy, /*prompt_is_rule*/ false).is_some()
        {
            manager.release_process_id(process_id).await;
            return Err(FunctionCallError::RespondToModel(format!(
                "approval policy is {approval_policy:?}; reject command — you cannot ask for escalated permissions if the approval policy is {approval_policy:?}"
            )));
        }
        let normalized_additional_permissions = match implicit_granted_permissions(
            sandbox_permissions,
            requested_additional_permissions.as_ref(),
            &effective_additional_permissions,
        )
        .map_or_else(
            || {
                normalize_and_validate_additional_permissions(
                    additional_permissions_allowed,
                    approval_policy,
                    effective_additional_permissions.sandbox_permissions,
                    effective_additional_permissions.additional_permissions,
                    effective_additional_permissions.permissions_preapproved,
                    &permission_context,
                )
            },
            |permissions| Ok(Some(permissions)),
        ) {
            Ok(normalized) => normalized,
            Err(err) => {
                manager.release_process_id(process_id).await;
                return Err(FunctionCallError::RespondToModel(err));
            }
        };

        let intercepted_patch = intercept_apply_patch(
            &command,
            &cwd,
            fs.as_ref(),
            turn_environment.clone(),
            context.session.clone(),
            Arc::clone(&context.step_context),
            context.cancellation_token.clone(),
            Some(&tracker),
            &context.call_id,
            "exec_argv",
        )
        .await;
        if intercepted_patch.is_err() {
            manager.release_process_id(process_id).await;
        }
        if let Some(output) = intercepted_patch? {
            manager.release_process_id(process_id).await;
            return Ok(boxed_tool_output(ExecCommandToolOutput {
                event_call_id: String::new(),
                chunk_id: String::new(),
                wall_time: std::time::Duration::ZERO,
                raw_output: output.into_text().into_bytes(),
                truncation_policy: turn.model_info().truncation_policy.into(),
                max_output_tokens,
                process_id: None,
                exit_code: None,
                original_token_count: None,
                output_omitted_bytes: None,
                hook_metadata: None,
            }));
        }

        emit_unified_exec_tty_metric(&turn.session_telemetry, tty);
        let argv0 = command[0].clone();
        let request = ExecCommandRequest {
            command,
            launch_mode: UnifiedExecLaunchMode::Argv,
            tool_name: ToolName::plain("exec_argv"),
            hook_metadata: hook_metadata.clone(),
            process_id,
            yield_time_ms,
            max_output_tokens,
            cwd,
            sandbox_cwd: native_environment_cwd,
            turn_environment: turn_environment.clone(),
            shell_mode: shell_mode_for_environment(
                &turn.unified_exec_shell_mode,
                environment.as_ref(),
            ),
            network: context.step_context.turn.network.clone(),
            tty,
            sandbox_permissions: effective_additional_permissions.sandbox_permissions,
            additional_permissions: normalized_additional_permissions,
            additional_permissions_preapproved: effective_additional_permissions
                .permissions_preapproved,
            justification,
            prefix_rule,
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
                    raw_output: output_text.into_bytes(),
                    truncation_policy: turn.model_info().truncation_policy.into(),
                    max_output_tokens,
                    process_id: None,
                    exit_code: Some(output.exit_code),
                    original_token_count: Some(original_token_count),
                    output_omitted_bytes,
                    hook_metadata: Some(hook_metadata),
                }))
            }
            Err(err) => Err(FunctionCallError::RespondToModel(format_exec_argv_error(
                &hook_command,
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
        create_exec_argv_tool_with_environment_id(
            CommandToolOptions {
                allow_login_shell: false,
                exec_permission_approvals_enabled: self.options.exec_permission_approvals_enabled,
            },
            self.options.include_environment_id,
        )
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
    fn matches_kind(&self, payload: &ToolPayload) -> bool {
        matches!(payload, ToolPayload::Function { .. })
    }

    fn pre_tool_use_payload(&self, invocation: &ToolInvocation) -> Option<PreToolUsePayload> {
        let ToolPayload::Function { arguments } = &invocation.payload else {
            return None;
        };
        let args = parse_arguments::<ExecArgvArgs>(arguments).ok()?;
        let argv = validate_argv(args.argv).ok()?;
        Some(PreToolUsePayload {
            tool_name: HookToolName::new("exec_argv"),
            tool_input: PermissionRequestPayload::exec_argv(
                codex_shell_command::parse_command::shlex_join(&argv),
                argv,
            )
            .tool_input,
        })
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
    hook_command: &str,
    argv0: &str,
    cwd: Option<&Path>,
    err: &UnifiedExecError,
) -> String {
    let mut message = format!("exec_argv failed for `{hook_command}`: {err:?}");
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
    let first = candidates[0].display();
    Some(format!(
        "Windows note: `exec_argv` does not ask a shell to resolve PATHEXT entries for argv[0]. Found candidate command shim(s): {formatted_candidates}. Try using the full path with its extension, for example `{first}` as argv[0]."
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
