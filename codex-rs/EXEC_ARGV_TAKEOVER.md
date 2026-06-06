# exec_argv Takeover Acceptance

## Scope

Implement a high-value `exec_argv` tool for unified exec. The tool runs one external program with an argv vector and reuses the existing unified exec process manager for sandboxing, approval, PTY handling, output streaming, remote execution, and process continuation.

The tool does not interpret shell syntax. Commands that require pipes, redirection, glob expansion, environment assignment syntax, shell builtins, or shell control flow continue to use `exec_command`.

## Design

- Add a model-visible `exec_argv` tool when unified exec is enabled.
- Require `argv: Vec<String>` and reject empty argv or empty `argv[0]`.
- Preserve existing unified exec controls: `workdir`, `tty`, `yield_time_ms`, `max_output_tokens`, `sandbox_permissions`, `additional_permissions`, `justification`, and `prefix_rule`.
- Include `environment_id` only when the current tool environment mode exposes multiple environments.
- Convert argv to a display command with `shlex_join` for logs, hooks, approval display, and user-visible command history.
- Mark argv invocations as non-shell-derived so shell snapshots, zsh-fork preparation, and shell-derived PowerShell script rewriting do not wrap or mutate the command.
- Run through the existing `UnifiedExecProcessManager`; do not add a new process backend.
- Use `exec_argv` as the hook identity so existing `Bash` hook rewrites do not automatically apply to argv-native calls.
- Expose hook input with `command` for display/audit and `argv` for the precise argument vector.
- Allow hook rewrites only when `updatedInput.argv` is an array of strings. Do not parse a rewritten `command` string back into argv.
- Keep `exec_command` registered for shell scripts and `write_stdin` registered for ongoing unified exec sessions.

## Non-Goals

- Do not replace `exec_command`.
- Do not emulate shell pipelines, redirects, globbing, or variable expansion.
- Do not auto-wrap `.cmd` or `.bat` files with `cmd /c`.
- Do not add per-call env overrides in this change.
- Do not change app-server `command/exec` or exec-server protocols.
- Do not change the unified exec process manager unless a test proves it is required.

## Acceptance

- `exec_argv` appears in the visible tool list whenever unified exec is visible.
- `exec_argv` is absent when shell tools have no active environment.
- `exec_argv` has an argv-first schema and shares the unified exec output schema.
- `exec_argv` executes argv directly without wrapping through the user's shell.
- `exec_argv` preserves the original argv when the program is PowerShell or pwsh; UTF-8 script prefixing remains limited to shell-derived PowerShell calls.
- `exec_argv` rejects empty argv and empty program names before allocating a process.
- `exec_argv` preserves workdir resolution against the selected turn environment.
- `exec_argv` supports sandbox and approval parameters through the existing unified exec path.
- `exec_argv` supports `tty` and returns process ids consistently with `exec_command`.
- `write_stdin` can continue sessions started by `exec_argv`.
- `apply_patch` interception still runs for argv invocations.
- Pre-tool hooks use the `exec_argv` hook identity and receive both `command` and `argv`.
- Pre-tool hook rewrites using `updatedInput.argv` update the invocation.
- Pre-tool hook rewrites that only provide `command` fail instead of being reparsed.
- Permission-request hooks use the `exec_argv` hook identity and receive both `command` and `argv`.
- Managed-network deferred permission hooks preserve the originating `exec_argv` hook identity and argv metadata while adding the `network-access ...` description.
- Post-tool hooks use the `exec_argv` hook identity and receive both `command` and `argv`.
- Post-tool hooks for sessions started by `exec_argv` keep the original `exec_argv` identity and argv metadata when completion is observed through `write_stdin`.
- Existing `exec_command`, `shell_command`, and `write_stdin` behavior remains covered by existing tests.

## Verification

- Run `just fmt` from `codex-rs`.
- Run `cargo test -p codex-core exec_argv_`.
- Run `cargo test -p codex-core shell_type_none_`.
- Run `cargo test -p codex-core exec_argv_network_approval_spec_uses_exec_argv_permission_payload`.
- Run `cargo test -p codex-core tools::network_approval::tests`.
- Run `cargo test -p codex-core tools::handlers::shell_spec::tests`.
- Run `cargo test -p codex-core tools::handlers::unified_exec::tests`.
- Run `cargo test -p codex-core shell_family_registers_visible_unified_exec_and_hidden_legacy_shell`.
- Run `cargo test -p codex-core environment_count_controls_environment_backed_tools`.
- Run `cargo test -p codex-core --test all prompt_tools_are_consistent_across_requests`.
- On a non-Windows runner, run the `exec_argv` hook suite coverage in `codex-rs/core/tests/suite/hooks.rs` and the environment-backed tool exposure coverage in `codex-rs/core/tests/suite/tools.rs`.
