use std::sync::Arc;

use crate::function_tool::FunctionCallError;
use crate::maybe_emit_implicit_skill_invocation;
use crate::tools::context::ExecCommandToolOutput;
use crate::tools::context::ToolInvocation;
use crate::tools::context::ToolPayload;
use crate::tools::context::boxed_tool_output;
use crate::tools::handlers::apply_granted_turn_permissions;
use crate::tools::handlers::apply_patch::intercept_apply_patch;
use crate::tools::handlers::implicit_granted_permissions;
use crate::tools::handlers::normalize_and_validate_additional_permissions;
use crate::tools::handlers::parse_arguments;
use crate::tools::handlers::parse_arguments_with_base_path;
use crate::tools::handlers::rewrite_function_arguments;
use crate::tools::hook_names::HookToolName;
use crate::tools::registry::CoreToolRuntime;
use crate::tools::registry::PostToolUsePayload;
use crate::tools::registry::PreToolUsePayload;
use crate::tools::registry::ToolExecutor;
use crate::unified_exec::ExecCommandRequest;
use crate::unified_exec::UnifiedExecContext;
use crate::unified_exec::UnifiedExecError;
use crate::unified_exec::UnifiedExecHookMetadata;
use crate::unified_exec::UnifiedExecProcessManager;
use crate::unified_exec::generate_chunk_id;
use codex_features::Feature;
use codex_tools::ToolName;
use codex_tools::ToolSpec;
use codex_utils_output_truncation::approx_token_count;
use codex_utils_path_uri::PathConvention;
use serde_json::Value;

use super::ExecArgvArgs;
use super::ExecCommandEnvironmentArgs;
use super::exec_command::emit_unified_exec_tty_metric;
use super::post_unified_exec_tool_use_payload_with_name;
use crate::tools::handlers::shell_spec::CommandToolOptions;
use crate::tools::handlers::shell_spec::create_exec_argv_tool_with_environment_id;

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

    fn handle(&self, invocation: ToolInvocation) -> codex_tools::ToolExecutorFuture<'_> {
        Box::pin(self.handle_call(invocation))
    }
}

impl ExecArgvHandler {
    async fn handle_call(
        &self,
        invocation: ToolInvocation,
    ) -> Result<Box<dyn crate::tools::context::ToolOutput>, FunctionCallError> {
        let ToolInvocation {
            session,
            turn,
            tracker,
            call_id,
            payload,
            ..
        } = invocation;

        let arguments = match payload {
            ToolPayload::Function { arguments } => arguments,
            _ => {
                return Err(FunctionCallError::RespondToModel(
                    "exec_argv handler received unsupported payload".to_string(),
                ));
            }
        };

        let manager: &UnifiedExecProcessManager = &session.services.unified_exec_manager;
        let context = UnifiedExecContext::new(session.clone(), turn.clone(), call_id.clone());
        let environment_args: ExecCommandEnvironmentArgs = parse_arguments(&arguments)?;
        let Some(turn_environment) = super::super::resolve_tool_environment(
            turn.as_ref(),
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
        let cwd_uses_native_convention =
            cwd.infer_path_convention() == Some(PathConvention::native());
        let native_cwd = match cwd.to_abs_path() {
            Ok(cwd) if cwd_uses_native_convention => cwd,
            Err(err) => return Err(FunctionCallError::RespondToModel(err.to_string())),
            Ok(_) => {
                return Err(FunctionCallError::RespondToModel(format!(
                    "path URI `{cwd}` does not use the host's native {} path convention",
                    PathConvention::native()
                )));
            }
        };
        let args: ExecArgvArgs = parse_arguments_with_base_path(&arguments, &native_cwd)?;
        let command = validate_argv(args.argv)?;
        let hook_command = codex_shell_command::parse_command::shlex_join(&command);
        let hook_metadata =
            UnifiedExecHookMetadata::exec_argv(hook_command.clone(), command.clone());
        maybe_emit_implicit_skill_invocation(
            session.as_ref(),
            context.turn.as_ref(),
            &hook_command,
            &native_cwd,
        )
        .await;
        let process_id = manager.allocate_process_id().await;

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

        let exec_permission_approvals_enabled =
            session.features().enabled(Feature::ExecPermissionApprovals);
        let requested_additional_permissions = additional_permissions.clone();
        let permission_cwd = &native_cwd;
        let effective_additional_permissions = apply_granted_turn_permissions(
            context.session.as_ref(),
            &turn_environment.environment_id,
            permission_cwd.as_path(),
            sandbox_permissions,
            additional_permissions,
        )
        .await;
        let additional_permissions_allowed = exec_permission_approvals_enabled
            || (session.features().enabled(Feature::RequestPermissionsTool)
                && effective_additional_permissions.permissions_preapproved);

        if effective_additional_permissions
            .sandbox_permissions
            .requests_sandbox_override()
            && !effective_additional_permissions.permissions_preapproved
            && !matches!(
                context.turn.approval_policy.value(),
                codex_protocol::protocol::AskForApproval::OnRequest
            )
        {
            let approval_policy = context.turn.approval_policy.value();
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
                    context.turn.approval_policy.value(),
                    effective_additional_permissions.sandbox_permissions,
                    effective_additional_permissions.additional_permissions,
                    effective_additional_permissions.permissions_preapproved,
                    permission_cwd,
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

        if let Some(output) = intercept_apply_patch(
            &command,
            &cwd,
            fs.as_ref(),
            turn_environment.clone(),
            context.session.clone(),
            context.turn.clone(),
            Some(&tracker),
            &context.call_id,
            "exec_argv",
        )
        .await?
        {
            manager.release_process_id(process_id).await;
            return Ok(boxed_tool_output(ExecCommandToolOutput {
                event_call_id: String::new(),
                chunk_id: String::new(),
                wall_time: std::time::Duration::ZERO,
                raw_output: output.into_text().into_bytes(),
                truncation_policy: turn.model_info.truncation_policy.into(),
                max_output_tokens,
                process_id: None,
                exit_code: None,
                original_token_count: None,
                hook_command: None,
                hook_tool_name: None,
                hook_input: None,
            }));
        }

        emit_unified_exec_tty_metric(&turn.session_telemetry, tty);
        match manager
            .exec_command(
                ExecCommandRequest {
                    command,
                    shell_type: None,
                    tool_name: ToolName::plain("exec_argv"),
                    hook_command: hook_command.clone(),
                    hook_metadata: hook_metadata.clone(),
                    process_id,
                    yield_time_ms,
                    max_output_tokens,
                    cwd,
                    sandbox_cwd: native_environment_cwd,
                    turn_environment: turn_environment.clone(),
                    shell_mode: turn.unified_exec_shell_mode.clone(),
                    network: context.turn.network.clone(),
                    tty,
                    sandbox_permissions: effective_additional_permissions.sandbox_permissions,
                    additional_permissions: normalized_additional_permissions,
                    additional_permissions_preapproved: effective_additional_permissions
                        .permissions_preapproved,
                    justification,
                    prefix_rule,
                },
                &context,
            )
            .await
        {
            Ok(response) => Ok(boxed_tool_output(response)),
            Err(UnifiedExecError::SandboxDenied { output, .. }) => {
                let output_text = output.aggregated_output.text;
                let original_token_count = approx_token_count(&output_text);
                Ok(boxed_tool_output(ExecCommandToolOutput {
                    event_call_id: context.call_id.clone(),
                    chunk_id: generate_chunk_id(),
                    wall_time: output.duration,
                    raw_output: output_text.into_bytes(),
                    truncation_policy: turn.model_info.truncation_policy.into(),
                    max_output_tokens,
                    process_id: None,
                    exit_code: Some(output.exit_code),
                    original_token_count: Some(original_token_count),
                    hook_command: Some(hook_command),
                    hook_tool_name: Some("exec_argv".to_string()),
                    hook_input: Some(hook_metadata.tool_input),
                }))
            }
            Err(err) => Err(FunctionCallError::RespondToModel(format!(
                "exec_argv failed for `{hook_command}`: {err:?}"
            ))),
        }
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
            tool_name: exec_argv_hook_name(),
            tool_input: serde_json::json!({
                "command": codex_shell_command::parse_command::shlex_join(&argv),
                "argv": argv,
            }),
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
        post_unified_exec_tool_use_payload_with_name(invocation, result, exec_argv_hook_name())
    }
}

fn exec_argv_hook_name() -> HookToolName {
    HookToolName::new("exec_argv")
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
    let mut argv = Vec::with_capacity(items.len());
    for item in items {
        let Some(arg) = item.as_str() else {
            return Err(FunctionCallError::RespondToModel(
                "hook returned updatedInput.argv for exec_argv with a non-string argument"
                    .to_string(),
            ));
        };
        argv.push(arg.to_string());
    }
    validate_argv(argv)
}
