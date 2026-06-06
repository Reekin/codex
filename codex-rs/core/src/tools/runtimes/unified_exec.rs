/*
Runtime: unified exec

Handles approval + sandbox orchestration for unified exec requests, delegating to
the process manager to spawn PTYs once an ExecRequest is prepared.
*/
use crate::command_canonicalization::canonicalize_command_for_approval;
use crate::exec::ExecCapturePolicy;
use crate::exec::ExecExpiration;
use crate::guardian::GuardianApprovalRequest;
use crate::guardian::GuardianNetworkAccessTrigger;
use crate::guardian::review_approval_request;
use crate::sandboxing::ExecOptions;
use crate::sandboxing::ExecServerEnvConfig;
use crate::sandboxing::SandboxPermissions;
use crate::shell::ShellType;
use crate::tools::flat_tool_name;
use crate::tools::network_approval::NetworkApprovalMode;
use crate::tools::network_approval::NetworkApprovalSpec;
use crate::tools::runtimes::build_sandbox_command;
use crate::tools::runtimes::disable_powershell_profile_for_elevated_windows_sandbox;
use crate::tools::runtimes::exec_env_for_sandbox_permissions;
use crate::tools::runtimes::maybe_wrap_shell_lc_with_snapshot;
use crate::tools::runtimes::shell::zsh_fork_backend;
use crate::tools::sandboxing::Approvable;
use crate::tools::sandboxing::ApprovalCtx;
use crate::tools::sandboxing::ExecApprovalRequirement;
use crate::tools::sandboxing::PermissionRequestPayload;
use crate::tools::sandboxing::SandboxAttempt;
use crate::tools::sandboxing::Sandboxable;
use crate::tools::sandboxing::ToolCtx;
use crate::tools::sandboxing::ToolError;
use crate::tools::sandboxing::ToolRuntime;
use crate::tools::sandboxing::managed_network_for_sandbox_permissions;
use crate::tools::sandboxing::with_cached_approval;
use crate::unified_exec::NoopSpawnLifecycle;
use crate::unified_exec::UnifiedExecError;
use crate::unified_exec::UnifiedExecHookMetadata;
use crate::unified_exec::UnifiedExecProcess;
use crate::unified_exec::UnifiedExecProcessManager;
use codex_exec_server::Environment;
use codex_network_proxy::NetworkProxy;
use codex_protocol::error::CodexErr;
use codex_protocol::error::SandboxErr;
use codex_protocol::models::AdditionalPermissionProfile;
use codex_protocol::protocol::ReviewDecision;
use codex_sandboxing::SandboxablePreference;
use codex_shell_command::powershell::prefix_powershell_script_with_utf8;
use codex_tools::UnifiedExecShellMode;
use codex_utils_absolute_path::AbsolutePathBuf;
use futures::future::BoxFuture;
use std::collections::HashMap;
use std::sync::Arc;
use tokio_util::sync::CancellationToken;

/// Request payload used by the unified-exec runtime after approvals and
/// sandbox preferences have been resolved for the current turn.
#[derive(Clone, Debug)]
pub struct UnifiedExecRequest {
    pub command: Vec<String>,
    pub shell_type: Option<ShellType>,
    pub hook_metadata: UnifiedExecHookMetadata,
    pub process_id: i32,
    pub cwd: AbsolutePathBuf,
    pub sandbox_cwd: AbsolutePathBuf,
    pub environment: Arc<Environment>,
    pub env: HashMap<String, String>,
    pub exec_server_env_config: Option<ExecServerEnvConfig>,
    pub explicit_env_overrides: HashMap<String, String>,
    pub network: Option<NetworkProxy>,
    pub tty: bool,
    pub sandbox_permissions: SandboxPermissions,
    pub additional_permissions: Option<AdditionalPermissionProfile>,
    #[cfg(unix)]
    pub additional_permissions_preapproved: bool,
    pub justification: Option<String>,
    pub exec_approval_requirement: ExecApprovalRequirement,
}

/// Cache key for approval decisions that can be reused across equivalent
/// unified-exec launches.
#[derive(serde::Serialize, Clone, Debug, Eq, PartialEq, Hash)]
pub struct UnifiedExecApprovalKey {
    pub command: Vec<String>,
    pub cwd: AbsolutePathBuf,
    pub tty: bool,
    pub sandbox_permissions: SandboxPermissions,
    pub additional_permissions: Option<AdditionalPermissionProfile>,
}

/// Runtime adapter that keeps policy and sandbox orchestration on the
/// unified-exec side while delegating process startup to the manager.
pub struct UnifiedExecRuntime<'a> {
    manager: &'a UnifiedExecProcessManager,
    shell_mode: UnifiedExecShellMode,
}

fn unified_exec_options(
    network_denial_cancellation_token: Option<CancellationToken>,
) -> ExecOptions {
    let mut expiration = ExecExpiration::DefaultTimeout;
    if let Some(cancellation) = network_denial_cancellation_token {
        expiration = expiration.with_cancellation(cancellation);
    }
    ExecOptions {
        expiration,
        capture_policy: ExecCapturePolicy::ShellTool,
    }
}

impl<'a> UnifiedExecRuntime<'a> {
    /// Creates a runtime bound to the shared unified-exec process manager.
    pub fn new(manager: &'a UnifiedExecProcessManager, shell_mode: UnifiedExecShellMode) -> Self {
        Self {
            manager,
            shell_mode,
        }
    }
}

impl Sandboxable for UnifiedExecRuntime<'_> {
    fn sandbox_preference(&self) -> SandboxablePreference {
        SandboxablePreference::Auto
    }

    fn escalate_on_failure(&self) -> bool {
        true
    }
}

impl Approvable<UnifiedExecRequest> for UnifiedExecRuntime<'_> {
    type ApprovalKey = UnifiedExecApprovalKey;

    fn approval_keys(&self, req: &UnifiedExecRequest) -> Vec<Self::ApprovalKey> {
        vec![UnifiedExecApprovalKey {
            command: canonicalize_command_for_approval(&req.command),
            cwd: req.cwd.clone(),
            tty: req.tty,
            sandbox_permissions: req.sandbox_permissions,
            additional_permissions: req.additional_permissions.clone(),
        }]
    }

    fn start_approval_async<'b>(
        &'b mut self,
        req: &'b UnifiedExecRequest,
        ctx: ApprovalCtx<'b>,
    ) -> BoxFuture<'b, ReviewDecision> {
        let keys = self.approval_keys(req);
        let session = ctx.session;
        let turn = ctx.turn;
        let call_id = ctx.call_id.to_string();
        let command = req.command.clone();
        let cwd = req.cwd.clone();
        let retry_reason = ctx.retry_reason.clone();
        let reason = retry_reason.clone().or_else(|| req.justification.clone());
        let guardian_review_id = ctx.guardian_review_id.clone();
        Box::pin(async move {
            if let Some(review_id) = guardian_review_id {
                return review_approval_request(
                    session,
                    turn,
                    review_id,
                    GuardianApprovalRequest::ExecCommand {
                        id: call_id,
                        command,
                        cwd: cwd.clone(),
                        sandbox_permissions: req.sandbox_permissions,
                        additional_permissions: req.additional_permissions.clone(),
                        justification: req.justification.clone(),
                        tty: req.tty,
                    },
                    retry_reason,
                )
                .await;
            }
            with_cached_approval(&session.services, "unified_exec", keys, || async move {
                let available_decisions = None;
                session
                    .request_command_approval(
                        turn,
                        call_id,
                        /*approval_id*/ None,
                        command,
                        cwd.clone(),
                        reason,
                        ctx.network_approval_context.clone(),
                        req.exec_approval_requirement
                            .proposed_execpolicy_amendment()
                            .cloned(),
                        req.additional_permissions.clone(),
                        available_decisions,
                    )
                    .await
            })
            .await
        })
    }

    fn exec_approval_requirement(
        &self,
        req: &UnifiedExecRequest,
    ) -> Option<ExecApprovalRequirement> {
        Some(req.exec_approval_requirement.clone())
    }

    fn permission_request_payload(
        &self,
        req: &UnifiedExecRequest,
    ) -> Option<PermissionRequestPayload> {
        Some(PermissionRequestPayload {
            tool_name: req.hook_metadata.tool_name.clone(),
            tool_input: req
                .hook_metadata
                .with_description(req.justification.clone()),
        })
    }

    fn sandbox_permissions(&self, req: &UnifiedExecRequest) -> SandboxPermissions {
        req.sandbox_permissions
    }
}

impl<'a> ToolRuntime<UnifiedExecRequest, UnifiedExecProcess> for UnifiedExecRuntime<'a> {
    fn sandbox_cwd<'b>(&self, req: &'b UnifiedExecRequest) -> Option<&'b AbsolutePathBuf> {
        Some(&req.sandbox_cwd)
    }

    fn network_approval_spec(
        &self,
        req: &UnifiedExecRequest,
        ctx: &ToolCtx,
    ) -> Option<NetworkApprovalSpec> {
        let network =
            managed_network_for_sandbox_permissions(req.network.as_ref(), req.sandbox_permissions)?;
        Some(NetworkApprovalSpec {
            network: Some(network.clone()),
            mode: NetworkApprovalMode::Deferred,
            trigger: GuardianNetworkAccessTrigger {
                call_id: ctx.call_id.clone(),
                tool_name: flat_tool_name(&ctx.tool_name).into_owned(),
                command: req.command.clone(),
                cwd: req.cwd.clone(),
                sandbox_permissions: req.sandbox_permissions,
                additional_permissions: req.additional_permissions.clone(),
                justification: req.justification.clone(),
                tty: Some(req.tty),
            },
            permission_request_payload: PermissionRequestPayload {
                tool_name: req.hook_metadata.tool_name.clone(),
                tool_input: req.hook_metadata.tool_input.clone(),
            },
        })
    }

    async fn run(
        &mut self,
        req: &UnifiedExecRequest,
        attempt: &SandboxAttempt<'_>,
        ctx: &ToolCtx,
    ) -> Result<UnifiedExecProcess, ToolError> {
        let base_command = &req.command;
        let session_shell = ctx.session.user_shell();
        let managed_network =
            managed_network_for_sandbox_permissions(req.network.as_ref(), req.sandbox_permissions);
        let mut env = exec_env_for_sandbox_permissions(&req.env, req.sandbox_permissions);
        if let Some(network) = managed_network {
            network.apply_to_env(&mut env);
        }
        let environment_is_remote = req.environment.is_remote();
        let command = command_with_optional_shell_snapshot(
            base_command,
            req.shell_type.as_ref(),
            environment_is_remote,
            session_shell.as_ref(),
            &req.cwd,
            &req.explicit_env_overrides,
            &env,
        );
        let command = disable_powershell_profile_for_elevated_windows_sandbox(
            &command,
            req.shell_type.as_ref(),
            attempt.sandbox,
            attempt.windows_sandbox_level,
        );
        let command = maybe_prefix_powershell_script_with_utf8_for_shell_type(
            command,
            req.shell_type.clone(),
        );

        if req.shell_type.is_some()
            && let UnifiedExecShellMode::ZshFork(zsh_fork_config) = &self.shell_mode
        {
            let command =
                build_sandbox_command(&command, &req.cwd, &env, req.additional_permissions.clone())
                    .map_err(|_| ToolError::Rejected("missing command line for PTY".to_string()))?;
            let options = unified_exec_options(attempt.network_denial_cancellation_token.clone());
            let mut exec_env = attempt
                .env_for(command, options, managed_network)
                .map_err(|err| ToolError::Codex(err.into()))?;
            exec_env.exec_server_env_config = req.exec_server_env_config.clone();
            match zsh_fork_backend::maybe_prepare_unified_exec(
                req,
                attempt,
                ctx,
                exec_env,
                zsh_fork_config,
            )
            .await?
            {
                Some(prepared) => {
                    if req.environment.is_remote() {
                        return Err(ToolError::Rejected(
                            "unified_exec zsh-fork is not supported for remote environments"
                                .to_string(),
                        ));
                    }
                    return self
                        .manager
                        .open_session_with_exec_env(
                            req.process_id,
                            &prepared.exec_request,
                            req.tty,
                            prepared.spawn_lifecycle,
                            req.environment.as_ref(),
                        )
                        .await
                        .map_err(|err| match err {
                            UnifiedExecError::SandboxDenied { output, .. } => {
                                ToolError::Codex(CodexErr::Sandbox(SandboxErr::Denied {
                                    output: Box::new(output),
                                    network_policy_decision: None,
                                }))
                            }
                            other => ToolError::Rejected(other.to_string()),
                        });
                }
                None => {
                    tracing::warn!(
                        "UnifiedExec ZshFork backend specified, but conditions for using it were not met, falling back to direct execution",
                    );
                }
            }
        }
        let command =
            build_sandbox_command(&command, &req.cwd, &env, req.additional_permissions.clone())
                .map_err(|_| ToolError::Rejected("missing command line for PTY".to_string()))?;
        let options = unified_exec_options(attempt.network_denial_cancellation_token.clone());
        let mut exec_env = attempt
            .env_for(command, options, managed_network)
            .map_err(|err| ToolError::Codex(err.into()))?;
        exec_env.exec_server_env_config = req.exec_server_env_config.clone();
        self.manager
            .open_session_with_exec_env(
                req.process_id,
                &exec_env,
                req.tty,
                Box::new(NoopSpawnLifecycle),
                req.environment.as_ref(),
            )
            .await
            .map_err(|err| match err {
                UnifiedExecError::SandboxDenied { output, .. } => {
                    ToolError::Codex(CodexErr::Sandbox(SandboxErr::Denied {
                        output: Box::new(output),
                        network_policy_decision: None,
                    }))
                }
                other => ToolError::Rejected(other.to_string()),
            })
    }
}

fn command_with_optional_shell_snapshot(
    base_command: &[String],
    shell_type: Option<&ShellType>,
    environment_is_remote: bool,
    session_shell: &crate::shell::Shell,
    cwd: &AbsolutePathBuf,
    explicit_env_overrides: &HashMap<String, String>,
    env: &HashMap<String, String>,
) -> Vec<String> {
    if environment_is_remote || shell_type.is_none() {
        base_command.to_vec()
    } else {
        maybe_wrap_shell_lc_with_snapshot(
            base_command,
            session_shell,
            cwd,
            explicit_env_overrides,
            env,
        )
    }
}

fn maybe_prefix_powershell_script_with_utf8_for_shell_type(
    command: Vec<String>,
    shell_type: Option<ShellType>,
) -> Vec<String> {
    if shell_type == Some(ShellType::PowerShell) {
        prefix_powershell_script_with_utf8(&command)
    } else {
        command
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::exec::DEFAULT_EXEC_COMMAND_TIMEOUT_MS;
    use crate::session::tests::make_session_and_context;
    use crate::shell::Shell;
    use crate::shell_snapshot::ShellSnapshot;
    use crate::tools::hook_names::HookToolName;
    use crate::tools::sandboxing::ToolRuntime;
    use codex_exec_server::Environment;
    use codex_network_proxy::ConfigReloader;
    use codex_network_proxy::ConfigState;
    use codex_network_proxy::NetworkProxyConfig;
    use codex_network_proxy::NetworkProxyConstraints;
    use codex_network_proxy::NetworkProxyState;
    use codex_tools::ToolName;
    use std::path::PathBuf;
    use std::sync::Arc;
    use std::time::Duration;
    use tempfile::tempdir;
    use tokio::sync::watch;

    struct StaticReloader;

    #[async_trait::async_trait]
    impl ConfigReloader for StaticReloader {
        fn source_label(&self) -> String {
            "test config state".to_string()
        }

        async fn maybe_reload(&self) -> anyhow::Result<Option<ConfigState>> {
            Ok(None)
        }

        async fn reload_now(&self) -> anyhow::Result<ConfigState> {
            Err(anyhow::anyhow!("force reload is not supported in tests"))
        }
    }

    async fn test_network_proxy() -> NetworkProxy {
        let state = codex_network_proxy::build_config_state(
            NetworkProxyConfig::default(),
            NetworkProxyConstraints::default(),
        )
        .expect("build network proxy config state");
        NetworkProxy::builder()
            .state(Arc::new(NetworkProxyState::with_reloader(
                state,
                Arc::new(StaticReloader),
            )))
            .managed_by_codex(/*managed_by_codex*/ false)
            .http_addr("127.0.0.1:43128".parse().expect("http proxy addr"))
            .socks_addr("127.0.0.1:48081".parse().expect("socks proxy addr"))
            .build()
            .await
            .expect("build test network proxy")
    }

    #[test]
    fn unified_exec_options_combines_default_timeout_with_network_denial_cancellation() {
        let cancellation = CancellationToken::new();
        let options = unified_exec_options(Some(cancellation.clone()));

        assert_eq!(options.capture_policy, ExecCapturePolicy::ShellTool);
        match options.expiration {
            ExecExpiration::TimeoutOrCancellation {
                timeout,
                cancellation: actual,
            } => {
                assert_eq!(
                    timeout,
                    Duration::from_millis(DEFAULT_EXEC_COMMAND_TIMEOUT_MS)
                );
                cancellation.cancel();
                assert!(actual.is_cancelled());
            }
            other => panic!("expected timeout-or-cancellation expiration, got {other:?}"),
        }
    }

    #[test]
    fn shell_type_none_does_not_rewrite_powershell_argv() {
        let command = vec![
            "powershell.exe".to_string(),
            "-NoProfile".to_string(),
            "-Command".to_string(),
            "Write-Output test".to_string(),
        ];

        assert_eq!(
            maybe_prefix_powershell_script_with_utf8_for_shell_type(command.clone(), None),
            command
        );
    }

    #[test]
    fn shell_type_none_skips_shell_snapshot_wrapping() {
        let cwd_dir = tempdir().expect("create cwd temp dir");
        let snapshot_dir = tempdir().expect("create snapshot temp dir");
        let snapshot_path = snapshot_dir.path().join("snapshot.sh");
        std::fs::write(&snapshot_path, "# Snapshot file\n").expect("write snapshot");
        let cwd =
            AbsolutePathBuf::try_from(cwd_dir.path().to_path_buf()).expect("absolute temp dir");
        let snapshot_path =
            AbsolutePathBuf::try_from(snapshot_path).expect("absolute snapshot path");
        let (_tx, shell_snapshot) = watch::channel(Some(Arc::new(ShellSnapshot {
            path: snapshot_path,
            cwd: cwd.clone(),
        })));
        let session_shell = Shell {
            shell_type: ShellType::Zsh,
            shell_path: PathBuf::from("/bin/zsh"),
            shell_snapshot,
        };
        let command = vec![
            "printf".to_string(),
            "%s".to_string(),
            "$HOME | cat".to_string(),
        ];

        assert_eq!(
            command_with_optional_shell_snapshot(
                &command,
                None,
                /*environment_is_remote*/ false,
                &session_shell,
                &cwd,
                &HashMap::new(),
                &HashMap::new(),
            ),
            command
        );
    }

    #[tokio::test]
    async fn exec_argv_permission_request_uses_exec_argv_hook_identity_and_input() {
        let cwd_dir = tempdir().expect("create cwd temp dir");
        let cwd =
            AbsolutePathBuf::try_from(cwd_dir.path().to_path_buf()).expect("absolute temp dir");
        let manager = UnifiedExecProcessManager::default();
        let runtime = UnifiedExecRuntime::new(&manager, UnifiedExecShellMode::Direct);
        let hook_metadata = UnifiedExecHookMetadata::exec_argv(
            "printf '%s' '$HOME | cat'".to_string(),
            vec![
                "printf".to_string(),
                "%s".to_string(),
                "$HOME | cat".to_string(),
            ],
        );
        let request = UnifiedExecRequest {
            command: vec![
                "printf".to_string(),
                "%s".to_string(),
                "$HOME | cat".to_string(),
            ],
            shell_type: None,
            hook_metadata,
            process_id: 1002,
            cwd: cwd.clone(),
            sandbox_cwd: cwd,
            environment: Arc::new(Environment::default_for_tests()),
            env: HashMap::new(),
            exec_server_env_config: None,
            explicit_env_overrides: HashMap::new(),
            network: None,
            tty: false,
            sandbox_permissions: SandboxPermissions::UseDefault,
            additional_permissions: None,
            #[cfg(unix)]
            additional_permissions_preapproved: false,
            justification: Some("Need literal argv execution".to_string()),
            exec_approval_requirement: ExecApprovalRequirement::Skip {
                bypass_sandbox: false,
                proposed_execpolicy_amendment: None,
            },
        };

        let payload = runtime
            .permission_request_payload(&request)
            .expect("permission request payload");

        assert_eq!(payload.tool_name, HookToolName::new("exec_argv"));
        assert_eq!(
            payload.tool_input,
            serde_json::json!({
                "command": "printf '%s' '$HOME | cat'",
                "argv": ["printf", "%s", "$HOME | cat"],
                "description": "Need literal argv execution",
            })
        );
    }

    #[tokio::test]
    async fn exec_argv_network_approval_spec_uses_exec_argv_permission_payload() {
        let (session, turn) = make_session_and_context().await;
        let session = Arc::new(session);
        let turn = Arc::new(turn);
        let cwd_dir = tempdir().expect("create cwd temp dir");
        let cwd =
            AbsolutePathBuf::try_from(cwd_dir.path().to_path_buf()).expect("absolute temp dir");
        let manager = UnifiedExecProcessManager::default();
        let runtime = UnifiedExecRuntime::new(&manager, UnifiedExecShellMode::Direct);
        let hook_metadata = UnifiedExecHookMetadata::exec_argv(
            "python3 -c 'print(1)'".to_string(),
            vec![
                "python3".to_string(),
                "-c".to_string(),
                "print(1)".to_string(),
            ],
        );
        let request = UnifiedExecRequest {
            command: vec![
                "python3".to_string(),
                "-c".to_string(),
                "print(1)".to_string(),
            ],
            shell_type: None,
            hook_metadata,
            process_id: 1003,
            cwd: cwd.clone(),
            sandbox_cwd: cwd,
            environment: Arc::new(Environment::default_for_tests()),
            env: HashMap::new(),
            exec_server_env_config: None,
            explicit_env_overrides: HashMap::new(),
            network: Some(test_network_proxy().await),
            tty: false,
            sandbox_permissions: SandboxPermissions::UseDefault,
            additional_permissions: None,
            #[cfg(unix)]
            additional_permissions_preapproved: false,
            justification: Some("Need managed network".to_string()),
            exec_approval_requirement: ExecApprovalRequirement::Skip {
                bypass_sandbox: false,
                proposed_execpolicy_amendment: None,
            },
        };
        let ctx = ToolCtx {
            session,
            turn,
            call_id: "call-argv-network".to_string(),
            tool_name: ToolName::plain("exec_argv"),
        };

        let spec = runtime
            .network_approval_spec(&request, &ctx)
            .expect("network approval spec");

        assert_eq!(spec.mode, NetworkApprovalMode::Deferred);
        assert_eq!(
            spec.permission_request_payload.tool_name,
            HookToolName::new("exec_argv")
        );
        assert_eq!(
            spec.permission_request_payload.tool_input,
            serde_json::json!({
                "command": "python3 -c 'print(1)'",
                "argv": ["python3", "-c", "print(1)"],
            })
        );
        assert_eq!(
            spec.permission_request_payload
                .with_description("network-access http://example.test:80".to_string())
                .tool_input,
            serde_json::json!({
                "command": "python3 -c 'print(1)'",
                "argv": ["python3", "-c", "print(1)"],
                "description": "network-access http://example.test:80",
            })
        );
    }

    #[tokio::test]
    async fn unified_exec_uses_the_trusted_sandbox_cwd() {
        let cwd_dir = tempdir().expect("create process temp dir");
        let sandbox_dir = tempdir().expect("create sandbox temp dir");
        let cwd =
            AbsolutePathBuf::try_from(cwd_dir.path().to_path_buf()).expect("absolute temp dir");
        let sandbox_cwd = AbsolutePathBuf::try_from(sandbox_dir.path().to_path_buf())
            .expect("absolute sandbox temp dir");
        let manager = UnifiedExecProcessManager::default();
        let runtime = UnifiedExecRuntime::new(&manager, UnifiedExecShellMode::Direct);
        let request = UnifiedExecRequest {
            command: vec!["pwd".to_string()],
            shell_type: Some(ShellType::Sh),
            hook_metadata: UnifiedExecHookMetadata::bash("pwd".to_string()),
            process_id: 1000,
            cwd,
            sandbox_cwd: sandbox_cwd.clone(),
            environment: Arc::new(Environment::default_for_tests()),
            env: HashMap::new(),
            exec_server_env_config: None,
            explicit_env_overrides: HashMap::new(),
            network: None,
            tty: false,
            sandbox_permissions: SandboxPermissions::UseDefault,
            additional_permissions: None,
            #[cfg(unix)]
            additional_permissions_preapproved: false,
            justification: None,
            exec_approval_requirement: ExecApprovalRequirement::Skip {
                bypass_sandbox: false,
                proposed_execpolicy_amendment: None,
            },
        };

        assert_eq!(runtime.sandbox_cwd(&request), Some(&sandbox_cwd));
    }
}
