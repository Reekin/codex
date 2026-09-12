# Argv-Native Unified Exec

## Goal

Expose `exec_argv` for launching one external native executable from an argv vector without adding
shell interpretation, while preserving its structured identity through hooks, approvals, background
sessions, and outputs.

## Non-Goals

- Do not replace `exec_command`.
- Do not emulate pipelines, redirects, globbing, variable expansion, shell builtins, environment
  assignment syntax, or shell control flow.
- Do not parse a display command back into argv.
- Do not automatically execute Windows `.cmd`, `.bat`, or PowerShell shims through a shell.
- Do not change unrelated app-server or exec-server protocols.

## Stable Contract

- **REQ-1 Input**: `argv` is a non-empty array of strings and `argv[0]` is a non-empty program
  name. Reject values that cannot be passed safely to process creation.
- **REQ-2 Execution**: Pass argv directly to one process without adding shell interpretation.
  Shell metacharacters remain ordinary arguments to native programs. An explicitly launched shell
  or interpreter still interprets its command or code arguments; argv is not a guarantee that those
  arguments are treated as literal data.
- **REQ-3 Separation**: A rendered command string is for logs, audit, hooks, and UI only. It is
  never execution truth.
- **REQ-4 Tool identity**: PreToolUse, PermissionRequest, managed-network approval, PostToolUse,
  and resumed `write_stdin` completion preserve `tool_name = "exec_argv"` and the authoritative
  argv for that call.
- **REQ-5 Rewrite**: A hook may change execution only through `updatedInput.argv`. A command-only
  rewrite fails instead of being reparsed.
- **REQ-6 Shell independence**: Shell snapshots, shell wrappers, zsh-fork behavior, PowerShell
  script prefixes, and other shell-derived transforms do not mutate argv-native requests.
- **REQ-7 Exposure**: Register `exec_argv` with the unified-exec tool family and hide it whenever
  environment-backed execution tools are unavailable.
- **REQ-8 Windows diagnostics**: If direct process creation fails for a bare program name,
  diagnostics may report exact `PATHEXT` candidates. Diagnostics must not change or retry the
  executed argv. A full native executable path can fix lookup; a full script path does not remove
  shell semantics. Direct `.cmd`, `.bat`, and `.ps1` scripts or shims to `exec_command` with the
  appropriate shell, or suggest the underlying native executable such as `node script.js`.
- **REQ-9 Tool guidance**: Recommend `exec_argv` for external native programs, including
  `node script.js`. Recommend `exec_command` for pipelines, redirects, globbing, variables,
  builtins, control flow, shell initialization, and Windows shell scripts or shims. Explain that
  explicitly invoking a shell or interpreter retains its command or code interpretation.

## Portability Constraints

- **MUST** carry one structured launch value through handler, runtime, sandbox/approval, process
  state, output, and resumed completion.
- **MUST** carry one structured hook metadata value containing tool identity and argv. Do not
  independently mutate compatibility fields derived from it.
- **MUST** distinguish shell and argv launch modes unambiguously. An existing documented enum,
  option variant, or sentinel is acceptable when every consumer shares it and tests prove its
  semantics.
- **MUST** register deferred approval metadata before the blocked managed-network request can be
  observed.
- **MUST** preserve metadata in long-running process/session state so `write_stdin` completion uses
  the originating tool contract.
- **PREFERRED** share generic process-launch plumbing with `exec_command` while keeping parsing,
  rewrite, and shell-transform policy separate.

Do not refactor a coherent carrier or launch-mode representation during a routine port solely to
match the preferred shape.

## Adapter Seams

Rediscover these semantics on every upstream base:

- tool schema, family visibility, and registration;
- argument parsing, validation, and hook rewrite;
- unified-exec request and launch-mode construction;
- shell snapshot and platform-specific command transforms;
- sandbox approval and managed-network approval;
- process/session creation and resumed completion;
- tool output and post-hook payload;
- prompt-caching and exposed-tool expectations;
- platform error formatting.

## P0 Acceptance

### Literal Execution

Run a real process with spaces and shell metacharacters in argv. Prove the child receives the exact
arguments and no shell syntax executes.

### Tool And Rewrite Contract

Prove the schema is argv-first, the tool is exposed only with unified exec, every pre/post hook sees
`exec_argv` plus argv, `updatedInput.argv` changes execution, and command-only rewrite fails.

### Approval Paths

Prove normal and managed-network PermissionRequest preserve the originating tool identity and argv.
The managed-network path may append only its access description.

### Background Completion

Start a long-running `exec_argv` process and complete it through `write_stdin`. Prove the final
output and PostToolUse path retain the originating call ID, tool identity, and authoritative argv.

### Platform Behavior

Prove argv-native requests skip shell-derived transforms. On Windows, prove missing bare commands
report discovered executable/shim paths without executing them automatically, and distinguish
native executable lookup from script shell semantics in both tool guidance and diagnostics.

## Integration Contract

- Compose with existing approval, sandbox, hook, managed-network, and process-session features.
- Keep packaging and downstream release automation out of the feature branch.
- When upstream rewrites unified exec, verify every observer of launch metadata rather than only
  successful process execution.
