# Feature Exec Argv

## Feature Metadata

- **Feature name**: `exec_argv` argv-native unified exec tool
- **Feature branch pattern**: `ft/exec-argv-<upstream-version>`; for example `ft/exec-argv-0.133`
- **Target integration branch pattern**: `integration-<upstream-version>`
- **Primary owner / purpose**: Provide an argv-first execution surface for agents that need literal process arguments without shell interpretation.
- **Current upstream base**: Record the upstream stable tag used by the active integration branch before each port.
- **Direct previous feature branch**: Use the latest completed versioned `ft/exec-argv-<upstream-version>` branch as the first code reference.

## 1. Requirement Goal

`exec_argv` gives the model a tool that runs one external program from an argv vector. It is distinct from shell execution. The model uses `exec_argv` when arguments must remain literal and uses `exec_command` when it needs pipes, redirects, glob expansion, shell variables, shell builtins, environment assignment syntax, or shell control flow.

The stable contract is:

- `argv` is a non-empty array of strings.
- `argv[0]` names the program to execute.
- Arguments are passed literally and are not reparsed by a shell.
- Display strings such as `command` are for logs, hooks, approval display, and audit only; they are not the source of execution truth.
- PreToolUse, PermissionRequest, managed-network PermissionRequest, PostToolUse, and `write_stdin` completion all preserve `tool_name: "exec_argv"` and `tool_input.argv`.
- Hook rewrites may update execution only through `updatedInput.argv`.
- Hook rewrites that only provide `updatedInput.command` fail instead of being parsed back into argv.
- `exec_command` remains available for shell syntax and keeps the historical `Bash` hook identity.
- On Windows, failed direct process creation for a bare `argv[0]` should diagnose likely `PATHEXT` shim candidates such as `.CMD`, `.BAT`, or `.EXE` without changing the executed argv.

Non-goals:

- Do not replace `exec_command`.
- Do not emulate shell pipelines, redirects, globbing, variable expansion, shell builtins, or shell control flow.
- Do not auto-wrap `.cmd` or `.bat` files with `cmd /c`.
- Do not silently resolve bare Windows command names to `PATHEXT` matches as execution behavior; diagnostics may suggest exact executable paths, but argv remains authoritative.
- Do not add per-call environment overrides as part of this feature.
- Do not change app-server `command/exec` or exec-server protocols unless upstream has already moved the execution contract there.

## 2. Technical Plan

This is a small tool/runtime feature. It does not need a separate stable-core crate, but it must keep a clear boundary between stable behavior and upstream adapter points.

| Layer | Stable meaning | Upstream adapter surface |
| --- | --- | --- |
| Tool contract | `argv: Vec<String>` executes one external program without shell parsing. | Tool schema and model-visible tool registration. |
| Execution semantics | argv-native requests skip shell-derived wrapping and mutation. | Unified exec runtime shell snapshot, zsh-fork, PowerShell prefix, sandbox command construction. |
| Hook metadata | Hook-facing identity and input are `exec_argv` plus `{ command, argv }`. | Hook payload APIs and output-to-post-hook plumbing. |
| Rewrite | `updatedInput.argv` is authoritative; `updatedInput.command` is not accepted for execution. | Tool registry rewrite and handler argument mutation. |
| Approval | PermissionRequest uses the originating hook metadata. | `Approvable` / sandbox approval payload construction. |
| Managed network | Deferred network approvals preserve the original PermissionRequest payload and only append `network-access ...` as `description`. | `NetworkApprovalSpec`, active network approval registration, blocked-request approval handling. |
| Session continuation | Sessions started by `exec_argv` keep original call id, hook identity, and argv through `write_stdin` completion. | Process/session state and completion output construction. |
| Exposure | `exec_argv` appears with unified exec and disappears when no environment-backed tools are available. | Tool family/spec-plan/prompt caching expectations. |
| Windows command discovery diagnostics | Process creation failures for bare `argv[0]` report `PATHEXT` candidates when present, while preserving argv-native execution semantics. | Handler error formatting and Windows path/PATHEXT lookup helper. |

The preferred stable metadata shape is a single value equivalent to:

```text
tool_name = "exec_argv"
tool_input = {
  "command": "<display command>",
  "argv": ["program", "arg"]
}
```

In the current implementation this role is held by `UnifiedExecHookMetadata`.

## 3. Implementation Map

Treat these as current adapter locations, not permanent file-path truth. On a new upstream base, first rediscover the equivalent hook points.

- Tool schema: `codex-rs/core/src/tools/handlers/shell_spec.rs`
- Tool registration: `codex-rs/core/src/tools/tool_family/shell.rs`, `codex-rs/core/src/tools/handlers/mod.rs`, `codex-rs/core/src/tools/handlers/unified_exec.rs`
- Handler: `codex-rs/core/src/tools/handlers/unified_exec/exec_argv.rs`
- Hook metadata: `codex-rs/core/src/unified_exec/mod.rs`
- Runtime adapter: `codex-rs/core/src/tools/runtimes/unified_exec.rs`
- Process/session adapter: `codex-rs/core/src/unified_exec/process_manager.rs`
- PostToolUse output adapter: `codex-rs/core/src/tools/context.rs`, `codex-rs/tools/src/tool_output.rs`
- Managed-network adapter: `codex-rs/core/src/tools/network_approval.rs`, unified exec `network_approval_spec`
- Tool exposure tests: `codex-rs/core/src/tools/spec_plan_tests.rs`, `codex-rs/core/tests/suite/tools.rs`, `codex-rs/core/tests/suite/prompt_caching.rs`
- Hook E2E tests: `codex-rs/core/tests/suite/hooks.rs`

Reference source priority:

1. This feature contract and acceptance section define behavior.
2. The direct previous versioned `ft/exec-argv-<upstream-version>` branch is the first implementation reference.
3. `codex-rs/EXEC_ARGV_TAKEOVER.md` on the feature branch is an implementation-history reference, not the long-term contract.
4. Current upstream architecture decides adapter placement.
5. When old code conflicts with this contract, follow this contract and record the deviation.

## 4. Porting Checklist

1. Add or port the `exec_argv` schema.
2. Register `exec_argv` alongside `exec_command` and `write_stdin` when unified exec is visible.
3. Validate `argv`: reject empty arrays, empty `argv[0]`, and NUL bytes.
4. Convert argv to a display command for logs/UI/hooks, but never use the display command as execution truth.
5. Add or port a hook metadata value equivalent to `UnifiedExecHookMetadata`.
6. Thread hook metadata through the request, runtime, process/session state, output, and post-hook paths.
7. Ensure the argv-native runtime signal, currently `shell_type: None`, bypasses shell snapshot wrapping, zsh-fork preparation, and shell-derived PowerShell UTF-8 prefixing.
8. Ensure PreToolUse sees `tool_name: "exec_argv"` and `{ command, argv }`.
9. Ensure `updatedInput.argv` rewrites the executed argv.
10. Ensure command-only hook rewrites fail.
11. Ensure PermissionRequest sees `tool_name: "exec_argv"` and `{ command, argv, description? }`.
12. Ensure managed-network deferred PermissionRequest preserves the originating payload and only appends the network-access description.
13. Ensure one-shot PostToolUse sees `tool_name: "exec_argv"` and `{ command, argv }`.
14. Ensure `write_stdin` completion for `exec_argv` sessions uses the original `exec_argv` identity and argv metadata.
15. Ensure Windows `program not found` errors for bare `argv[0]` suggest exact `PATHEXT` candidates when candidates exist.
16. Update prompt caching, tool exposure, and spec-plan expectations.
17. Keep downstream packaging and release automation out of the feature branch; carry it on the integration branch.

## 5. Lessons And Pitfalls

- Do not treat successful process execution as sufficient. This feature is mostly about preserving argv-native identity through cross-cutting systems.
- Do not let old `Bash` hook identity leak into `exec_argv` paths. Check PreToolUse, PermissionRequest, managed-network PermissionRequest, PostToolUse, and `write_stdin` completion separately.
- Do not reconstruct hook payloads from display command strings. Carry the structured hook metadata.
- Managed-network approval is deferred. Metadata must be registered before the blocked request happens.
- `write_stdin` completion is a separate output path. Session state must carry original hook metadata.
- PowerShell UTF-8 script prefixing is shell-derived behavior. It must not mutate argv-native requests.
- Windows `.cmd` / `.bat` / `.ps1` shims are a common source of false "not installed" diagnoses. Keep execution exact, but make errors point to discovered `PATHEXT` candidates so the model can retry with an explicit path.
- On Windows, several hook/tool suite tests may be cfg-gated. Use focused lower-level tests locally and run non-Windows E2E tests on an appropriate runner when available.

## 6. Acceptance Standard

P0 gates:

| Gate | Evidence type | Command / file | Notes |
| --- | --- | --- | --- |
| Tool is visible with unified exec and hidden without an environment | Test | `cargo test -p codex-core shell_family_registers_visible_unified_exec_and_hidden_legacy_shell`; `cargo test -p codex-core environment_count_controls_environment_backed_tools` | Also update suite tool exposure expectations. |
| Tool schema is argv-first | Test | `cargo test -p codex-core tools::handlers::shell_spec::tests` | Schema must explain shell syntax belongs in `exec_command`. |
| argv executes literally without shell parsing | Test | `cargo test -p codex-core shell_type_none_` | Shell metacharacters remain literal. |
| Shell-derived transforms are skipped | Test | `cargo test -p codex-core shell_type_none_` | Covers shell snapshot and PowerShell prefix behavior. |
| PreToolUse uses `exec_argv` and includes argv | Test | `cargo test -p codex-core exec_argv_`; non-Windows hooks suite | Bash hooks must not match. |
| Hook rewrite only accepts `updatedInput.argv` | Test | `cargo test -p codex-core exec_argv_`; non-Windows hooks suite | Command-only rewrite fails. |
| PermissionRequest preserves `exec_argv` metadata | Test | `cargo test -p codex-core exec_argv_`; non-Windows hooks suite | Includes argv. |
| Managed-network PermissionRequest preserves argv | Test | `cargo test -p codex-core exec_argv_network_approval_spec_uses_exec_argv_permission_payload`; non-Windows hooks suite | Payload only gains network-access description. |
| PostToolUse preserves `exec_argv` metadata | Test | `cargo test -p codex-core exec_argv_`; non-Windows hooks suite | Includes one-shot output. |
| `write_stdin` completion preserves `exec_argv` metadata | Test | `cargo test -p codex-core exec_argv_`; non-Windows hooks suite | Covers long-running sessions. |
| Windows `PATHEXT` diagnostics suggest explicit shim paths | Test | `cargo test -p codex-core windows_pathext_candidates`; Windows smoke if available | Does not auto-resolve or execute the candidate. |
| Prompt caching and tool exposure expectations are current | Test | `cargo test -p codex-core --test all prompt_tools_are_consistent_across_requests` | Update expected tools. |

Focused validation commands:

```bash
cargo test -p codex-core exec_argv_
cargo test -p codex-core windows_pathext_candidates
cargo test -p codex-core shell_type_none_
cargo test -p codex-core exec_argv_network_approval_spec_uses_exec_argv_permission_payload
cargo test -p codex-core tools::network_approval::tests
cargo test -p codex-core tools::handlers::unified_exec::tests
cargo test -p codex-core tools::handlers::shell_spec::tests
cargo test -p codex-core shell_family_registers_visible_unified_exec_and_hidden_legacy_shell
cargo test -p codex-core environment_count_controls_environment_backed_tools
cargo test -p codex-core --test all prompt_tools_are_consistent_across_requests
just fmt
just fix -p codex-core
```

On non-Windows runners, also execute the relevant `codex-rs/core/tests/suite/hooks.rs` and `codex-rs/core/tests/suite/tools.rs` scenarios. On Windows, these may compile or run as zero-test filtered targets because of `cfg(not(target_os = "windows"))`.

P1 follow-ups that should not block a clean port:

- Consolidate `ExecCommandToolOutput` hook fields into one optional metadata value if the upstream output shape makes that low-risk.
- Extract shared `exec_command` / `exec_argv` launch plumbing if upstream changed enough that duplication becomes a migration hazard.
- Split tool option types if `CommandToolOptions` becomes misleading for argv-native schemas.

## 7. Migration Work Package

Before porting to a new upstream base, fill:

| Section | Required content |
| --- | --- |
| Context | upstream tag, previous feature branch, new feature branch, target integration branch |
| Carry-forward inventory | code files, tests, docs, omitted integration-only pieces |
| Upstream hook map | current schema, registration, handler, runtime, approval, network approval, process/session, post-hook locations |
| Architecture decisions | metadata source of truth, runtime argv-native signal, managed-network payload strategy |
| Phase checklist | port schema/registration, handler, metadata threading, runtime transforms, hooks/approvals, tests |
| Validation mapping | focused commands and unavailable platform tests |
| P0 evidence | command result or explicit unsupported-path decision for every P0 gate |
| Residual risks | remaining P1/P2 cleanup only |

## 8. Integration Handoff

Feature branches should carry only the `exec_argv` behavior, tests, and feature-owned docs. The integration branch should:

- merge or replay the completed versioned `ft/exec-argv-<upstream-version>` branch after feature acceptance;
- keep release packaging and downstream artifact automation separate from the feature branch;
- run integration smoke tests after composing this feature with other personal features;
- preserve this reference file and the `personal-features-port` skill index so future ports start from the contract instead of rediscovering the review history.
