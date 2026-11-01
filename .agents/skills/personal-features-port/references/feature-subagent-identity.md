# Subagent Identity

## Goal

Ensure a spawned subagent can identify itself and interpret inherited history, agent listings, and
wait results without confusing the root agent, parent, siblings, or prior transcript authorship
with its own identity.

## Non-Goals

- Do not change scheduling, ownership, or mailbox routing semantics.
- Do not change agent role/type, model, reasoning, service-tier, or runtime override behavior.
- Do not add release or packaging changes to the feature branch.

## Stable Contract

- **REQ-1 Canonical identity**: When a canonical agent path exists, provide it. Otherwise state that
  the path is absent and the current agent is still not the `/root` main agent.
- **REQ-2 Identity facts**: Provide parent thread, depth, nickname, and role when available.
- **REQ-3 Inherited history**: Explain that inherited transcript messages are context and do not
  prove the current subagent performed those actions.
- **REQ-4 Fork boundaries**: Full-history forks wrap inherited parent history with start/end
  boundaries. The end boundary appears before the new direct task and identifies that task as the
  current assignment.
- **REQ-5 Ordinary startup**: Non-fork subagent startup does not claim that preceding messages are
  inherited parent history.
- **REQ-6 List**: Identity context lets the subagent compare its canonical path with the upstream
  `agent_name` entries returned by `list_agents`. Do not add identity fields to tool output.
- **REQ-7 Wait**: V2 `wait_agent` names the current agent mailbox in its existing `message` field.
- **REQ-8 Self-wait**: Reject waiting on the current agent whenever that call would otherwise be
  confused with waiting for descendants.
- **REQ-9 Protocol compatibility**: Preserve upstream tool definitions, input schemas, and output
  schemas in every supported multi-agent version. Keep identity guidance in bounded context and
  existing textual result fields; keep name/path length validation inside the runtime.

## Portability Constraints

- **MUST** derive model context and wait identity guidance from one authoritative set of identity
  facts. Do not independently infer names from transcript order, thread ID, and registry metadata.
- **MUST** keep rendering separate from identity facts so wording can change without changing
  identity semantics.
- **MUST** inject identity through the repository's normal bounded context/world-state mechanism and
  preserve its deduplication rules.
- **MUST** treat full-history fork boundary insertion as an adapter around current history,
  compaction, and reference-context assembly.
- **MUST** handle missing canonical paths explicitly; never fall back to `/root`.
- **MUST** keep role/runtime override behavior outside this feature contract.
- **PREFERRED** use a small typed value equivalent to path, parent, depth, nickname, and role as the
  shared projection input.

## Adapter Seams

Rediscover these semantics on every upstream base:

- session source and canonical agent path;
- parent/depth/nickname/role metadata;
- initial context and world-state injection;
- full-history fork and compacted-history assembly;
- agent registry list projection;
- v1/v2 list, wait, and self-wait handlers;
- model-visible tool schemas.

## P0 Acceptance

### Startup Identity

Spawn subagents with and without canonical paths. Prove context reports the correct path or explicit
no-path/not-root state and includes available parent, depth, nickname, and role facts.

### Fork Boundaries

Create a full-history fork. Prove inherited messages are bounded, described as context, and followed
by the new assignment. Prove an ordinary non-fork startup has no fork-history boundary wording.

### List And Wait

From a real spawned subagent, compare its contextual identity with list entries. Prove V2 wait
names the caller's mailbox through its existing message field, and V1 self-wait is rejected.

### Protocol Parity

Exercise every supported multi-agent tool version and verify tool definitions and output shapes
match upstream. Verify identity context and full-history boundaries remain available in both
versions without requiring identical list/wait APIs.

## Integration Contract

- Full-history fork behavior may share code with role/runtime features, but identity branches must
  not absorb those contracts.
- Keep integration release automation outside the feature branch.
