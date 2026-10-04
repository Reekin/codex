# Superseded Turn Closure

## Goal

A thread runs one turn at a time. When a process dies mid-turn (crash, power loss, forced kill),
the rollout keeps that turn's `TurnStarted` record with no terminal record. After the thread
resumes and starts another turn, the dead turn must read as interrupted everywhere: thread history
shows it as interrupted, and it is a valid fork anchor that forks through exactly the work it
persisted.

## Non-Goals

- Do not write synthetic terminal records into rollouts; the rule is derived when history is
  projected.
- Do not change the latest turn of a thread. Whether an unfinished latest turn is still running is
  decided by the existing loaded-thread status logic.
- Do not change how live interrupts, `TurnAborted`, or `TurnComplete` are recorded.
- Do not rebuild existing history projections automatically; previously projected rows are
  repaired with a one-time maintenance script outside the product.

## Stable Contract

- **REQ-1 Closure**: When a thread records `TurnStarted` for turn B while an earlier explicit turn A
  is still in progress, A becomes `interrupted`. A turn that already reached a terminal status is
  unchanged, and a repeated `TurnStarted` for the same turn ID does not close it.
- **REQ-2 Boundary**: A closed turn ends immediately before B's `TurnStarted` record. Forking through
  A includes every record persisted before B started and excludes B's `TurnStarted` and everything
  after it.
- **REQ-3 Agreement**: The in-memory history projection (thread reads and rollout-based fork
  truncation) and the SQLite paginated projection (turn lists and paginated forks) report the same
  status for A and accept A as a `lastTurnId` fork anchor.
- **REQ-4 Scope**: Closure applies only within one rollout segment. Turns inherited from a parent
  rollout keep the status recorded by their own segment.
- **REQ-5 First terminal wins**: The inferred interruption is a terminal status. A terminal event for
  A that arrives after B started does not reopen it or turn it into a completion, matching how the
  projections treat other terminal statuses.

## Portability Constraints

- **MUST** derive closure from the persisted `TurnStarted` of the next turn, not from process state,
  timestamps, or loaded-thread status.
- **MUST** apply the rule in both history projections so status and fork boundaries agree.
- **MUST** place the boundary at the next turn's start record, using the same exclusive ordinal and
  byte position a fork-before-B would use.
- **MUST** reuse each projection's normal terminal-turn update path so summary fields (first user
  item, final agent item) are filled as for any interrupted turn.
- **PREFERRED** keep the rule at the existing turn-start handlers of each projection instead of a
  separate repair pass.

## Adapter Seams

- in-memory history builder turn-start handling;
- SQLite projection change-set application for turn starts;
- terminal-turn summary item resolution;
- `lastTurnId` fork validation in both the rollout-truncation and paginated-fork paths.

## P0 Acceptance

### Orphaned Turn Closure

Setup: a rollout with turn A started, A's items, a record between A and B, then turn B started and
completed. Action: build history, read SQLite turn rows, and fork through A on both fork paths.
Result: A is `interrupted` in both projections; the SQLite end position is the position just before
B's `TurnStarted`; both fork paths succeed and the forked history contains A's records and none of
B's. Evidence: unit tests for the builder and the SQLite projection, and a real `thread/fork` with
`lastTurnId` against the packaged app-server on a crash-orphaned rollout.

### Unaffected Turns

Prove an unfinished latest turn stays in progress and is still rejected as a fork anchor, a
completed turn followed by a new turn is unchanged, a repeated `TurnStarted` for the same turn ID
does not close it, and a turn inherited from a parent rollout is not closed by a child segment's
turn start. Evidence: unit tests.

## Integration Contract

- No interaction with other personal features beyond sharing the history projection code.
- Composed smoke: fork through a crash-orphaned turn with the packaged integration binary.
- Maintenance scripts for already-projected rows stay outside the repository.
