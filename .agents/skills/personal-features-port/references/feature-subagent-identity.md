# Subagent Identity Feature

Use this reference for the downstream subagent identity clarification feature.

This is a compact feature guide because the current scope is small compared with chat tree. Expand it into separate `feature-subagent-identity-*` references if the feature grows.

## Feature Metadata

- **Feature name**: subagent identity clarification.
- **Feature branch pattern**: `ft/subagent-identity-labels-<upstream-version>`; for example `ft/subagent-identity-labels-0.133`.
- **Target integration branch pattern**: merge or replay into the versioned integration branch, for example `integration-0.133`.
- **Primary purpose**: prevent spawned subagents from confusing inherited transcript history, root agent state, sibling agents, or wait/list output with their own identity.

## 1. Requirement Goal

Spawned agents must be able to determine their current canonical agent path and interpret multi-agent tool output without ambiguity.

Required behavior:

- subagent developer context states the current canonical agent path when one exists, otherwise clearly states that the path is absent and the agent is still not the `/root` main agent;
- subagent developer context states parent thread id, depth, nickname, and role when available;
- inherited transcript history is described as context, not proof that the current subagent performed those actions;
- full-history forked subagents receive developer start/end boundaries around inherited parent history; the end boundary says the inherited messages are not the current task and the next direct task message is the assignment;
- `list_agents` output identifies the current agent and exposes the current agent name/path;
- `wait_agent` output makes clear which mailbox is being observed;
- waiting on the current agent is rejected where the tool semantics would otherwise be confusing.

Non-goals:

- changing chat tree behavior;
- changing agent scheduling or task ownership semantics;
- adding release workflow or package automation to this feature branch.

## 2. Technical Plan

Keep the feature as a thin multi-agent/tool contract change.

Stable contract:

- `SessionSource::SubAgent` drives identity hints;
- full-history forked subagents receive the identity hint as a developer message in the forked history even when a parent `TurnContextItem` baseline prevents normal initial-context reinjection;
- full-history forked subagents use fork-specific boundary hints around the inherited parent history; ordinary subagent startup context must not claim that earlier startup messages are forked parent history;
- v1/full-history subagents without a canonical `AgentPath` must be told that they have no canonical path and are not `/root`;
- listed-agent output includes an explicit current-agent marker;
- wait output names the current agent/mailbox;
- tests cover v1 and v2 multi-agent tools where both exist.

Adapter surface:

- session source and agent registry fields may move upstream;
- tool schema/spec generation may move upstream;
- wait/list handler namespaces may move upstream.

On a new upstream base, preserve the behavior contract and adapt only the session/tool-spec/list/wait plumbing.

## 3. Implementation Details And References

Current implementation areas from the v0.133 port:

- `codex-rs/core/src/agent/control.rs`;
- `codex-rs/core/src/session/mod.rs`;
- `codex-rs/core/src/session/multi_agents.rs`;
- `codex-rs/core/src/session/tests.rs`;
- `codex-rs/core/src/tools/handlers/multi_agents/wait.rs`;
- `codex-rs/core/src/tools/handlers/multi_agents_spec.rs`;
- `codex-rs/core/src/tools/handlers/multi_agents_spec_tests.rs`;
- `codex-rs/core/src/tools/handlers/multi_agents_tests.rs`;
- `codex-rs/core/src/tools/handlers/multi_agents_v2/list_agents.rs`;
- `codex-rs/core/src/tools/handlers/multi_agents_v2/wait.rs`.

Reference source priority:

1. Behavior in this file.
2. Direct previous versioned `ft/subagent-identity-labels-<upstream-version>` branch.
3. Current upstream multi-agent tool schema and handler architecture.

Do not copy chat tree feature documentation into this feature. If chat-tree docs mention subagent exclusion, that is chat-tree behavior, not this feature's runtime contract.

## 4. Operating Rules

- Keep the feature branch focused on subagent identity runtime/API/test behavior.
- Do not include `.agents/skills/personal-features-port` documentation updates unless the task is explicitly about skill docs.
- Do not include release workflow or packaging changes.
- Preserve backward compatibility for existing tool calls unless the old behavior is the ambiguity being fixed.
- Update tool schema/spec tests whenever model-visible tool output changes.

## 5. Lessons And Pitfalls

- Full-history forked subagents inherit transcript content from parents; identity hints must explicitly say inherited history is context, not current-agent action.
- Do not rely only on `build_initial_context` for subagent identity hints. Full-history fork preserves the parent reference-context baseline, so the child may skip initial context and only receive settings diffs.
- A generic subagent identity hint is not enough for full-history forks. Add fork start/end boundaries around inherited parent history so the model sees that previous `user` messages are historical context and the next direct task message is the current assignment.
- Do not put fork-history wording into the ordinary initial-context identity hint. Non-fork subagents also receive identity context, but there is no inherited parent transcript above that hint.
- In v1 `fork_context` flows, `agent_path` may be absent. Never fall back to `/root` for identity wording in that case; say the path is absent and the child is still not the root main agent.
- `/root`, sibling agents, and child agents may appear in list output; current-agent identity must be explicit rather than inferred from transcript order.
- Waiting for the current agent is easy to misinterpret as waiting for children; reject or explain this path clearly.
- Documentation edits can accidentally mix with feature commits. Keep feature runtime changes separate from chat-tree or personal-features-port documentation.

## 6. Acceptance Standard

P0 gates:

- spawned subagent context includes current canonical agent path when present, or explicit no-path/not-root wording when absent, plus parent/depth identity facts;
- full-history forked subagent history appends the identity hint even when the parent reference-context baseline is preserved;
- full-history forked subagent history wraps inherited parent history with start/end boundary markers, with the end marker before the new task and identifying the next direct task message as the assignment;
- ordinary non-fork subagent initial context does not include fork-history boundary markers;
- list output exposes current agent identity and marks the current agent;
- wait output clarifies the current mailbox/agent being observed;
- self-wait behavior is rejected or otherwise impossible to confuse;
- v1/v2 tool specs and tests match the exposed contract;
- no chat-tree or integration-release files are required for the feature branch to stand alone.

Focused validation:

- `cd codex-rs && cargo test -p codex-core multi_agents`
- If Windows debug stack overflows, rerun with `RUST_MIN_STACK=33554432` and record that environment condition.

P0 evidence table:

| Gate | Evidence | Result | Notes |
| --- | --- | --- | --- |
| Subagent context names current path | session or multi-agent test | pending | |
| `list_agents` marks current agent | handler/spec test | pending | |
| `wait_agent` clarifies mailbox | handler/spec test | pending | |
| self-wait is rejected | handler test | pending | |
| feature branch excludes docs/release plumbing | `git diff --name-only` review | pending | |
