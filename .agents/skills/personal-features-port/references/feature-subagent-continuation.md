# Subagent Continuation

## Goal

Let parents handle completed subagent work after finishing a reply, and handle user steering
promptly while waiting for subagents.

## Non-Goals

Background command or code-cell completion, process-restart recovery, changing child execution
lifetimes, and interrupting unrelated tools are outside this feature.

## Stable Contract

- **REQ-1**: In both multi-agent versions, a subagent completion or error delivers its existing
  result to its direct parent. If the parent has naturally finished its turn, a follow-up turn
  processes that result without requiring or fabricating a new user request.
- **REQ-2**: A running parent consumes results through its normal input flow. Results arriving
  during finalization remain eligible for follow-up; each notification is consumed once and
  parent model execution remains serial.
- **REQ-3**: User steering ends an outstanding subagent wait promptly so the parent can process
  that input, without cancelling the child or waiting for its completion or the wait deadline.
  Existing completion and timeout behavior remains available.
- **REQ-4**: Explicit interruption does not automatically restart the parent for a child result.
  Retain the result for subsequent user-initiated work. Closed parents are not resurrected.
- **REQ-5**: Preserve existing child identity, result formats, and bounded context handling.
  Continuations use the parent's configuration and emit the normal turn lifecycle events.

## Portability Constraints

- MUST attach wakeup to actual child completion and parent admission/finalization hooks, so
  completion at the turn boundary is handled without polling or concurrent parent turns.
- MUST return control from the wait tool on steering while keeping the tool-call transcript valid.
- MUST preserve active-turn delivery and stop semantics when upstream changes the input queue.
- MUST retain a result recipient across client unsubscription and idle eviction while child work
  or unconsumed child results still depend on it; release that retention after consumption.

## Adapter Seams

Child terminal-status watchers; session-scoped pending input; idle turn admission; turn completion
and interruption; subagent wait activity subscriptions; model-visible tool descriptions.

## P0 Acceptance

- Finish a parent reply while a child is held by a deterministic model server, then release the
  child. Observe a new parent model request with the child result and no new user submission.
  Exercise both multi-agent versions through the production RPC/model path. Covers REQ-1/5.
- Repeat with the parent unsubscribed and the child held beyond the idle unload delay. Observe
  autonomous result consumption, then normal idle unloading after consumption. Covers REQ-1/2.
- Complete a child during an active parent turn and at its answer boundary. Observe delivery once,
  serialized requests, and completion/start event ordering. Covers REQ-2/5.
- Hold a child, enter a long wait, and steer the parent. Observe the next parent request containing
  steering before releasing the child or reaching the timeout. Cover both versions, plus unchanged
  completion and timeout results. Covers REQ-3.
- Interrupt a parent with an outstanding child, then release the child. Observe no autonomous
  restart; a later user turn can consume the result. Covers REQ-4.

## Integration Contract

Compose with subagent identity, empty-final retry, superseded-turn closure, and local compaction.
Use normal result items and turn boundaries so history cleanup and API projections keep their
existing semantics. Packaging remains integration-owned.

## Maintenance Rules

Keep behavioral requirements here. Record concrete hook paths, refs, commands, and acceptance
evidence in the temporary port brief. Update this contract only for intentional behavior changes.
