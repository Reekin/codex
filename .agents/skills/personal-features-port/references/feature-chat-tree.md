# Chat Tree

## Goal

Represent a main-thread conversation as durable branches of user turns. Let the user inspect the
whole tree and choose the branch that supplies future model context without deleting sibling
branches.

## Non-Goals

- Do not add subagent, side-thread, or internal-session turns to the main chat tree.
- Do not make summaries part of model-visible context.
- Do not let UI state or runtime context caches become durable truth.
- Do not expose a partially supported UI that switches model context while showing another branch.

## Stable Contract

- **REQ-1 Turn creation**: Before a normal main-thread user turn mutates model-visible history,
  capture the current node as parent, create a node for the new turn, and make it current.
- **REQ-2 Node facts**: Preserve stable node ID, nullable parent ID, nullable turn ID, creation
  order, status, and nullable summary. Do not expose an internal root sentinel.
- **REQ-3 Lifecycle**: Support `pending`, `completed`, `interrupted`, `replaced`, and
  `reviewEnded`. Completion and abort paths finalize the node durably.
- **REQ-4 Projection**: The current node defines one root-to-current path. The next real model
  request, `thread/read`, and `thread/turns/list` must use the same visible turns and exclude
  siblings.
- **REQ-5 Selection**: Changing current node preserves every branch, is revision-checked, is
  durable before history changes, survives restart, and is rejected while a task is running.
- **REQ-6 Persistence**: Durable facts must recover nodes, parents, turn mapping, statuses, order,
  current node, revision, and summaries. Runtime context snapshots and UI caches are rebuildable.
- **REQ-7 Summary**: A completed main-thread turn with non-empty assistant output schedules a
  separate asynchronous no-tools summary request. The bounded single-line result is persisted as a
  late update. Failure, cancellation, or empty output never fails or rewinds the completed turn.
- **REQ-8 Isolation**: A thread-spawn subagent that inherits parent history clones the inherited
  selected projection and appends its own turns as child-local continuation nodes without mutating
  the parent tree or creating main-tree summary jobs. Other side threads and internal session
  sources create no tree nodes.
- **REQ-9 UI**: `/chattree` shows the complete tree, marks current and selected rows, supports
  navigation, confirmation, and cancellation, blocks unsafe switching, and refreshes the visible
  transcript immediately after current changes.
- **REQ-10 Public API**: When app-server exists, preserve `chatTree/read`,
  `chatTree/setCurrent`, and `chatTree/updated`, plus selected-branch behavior in existing thread
  transcript APIs.

## Durable And Wire Data

The domain projection contains:

- `version`;
- monotonically increasing `revision`;
- nullable `currentNodeId`;
- `visibleNodeIds`;
- `visibleTurnIds`;
- chronologically ordered `nodes`.

Each wire node contains:

- `nodeId`;
- nullable `parentNodeId`;
- nullable `turnId`;
- `order`;
- `status`;
- nullable `summary`.

Required nullable fields serialize as `null`, not omission. Preserve existing enum strings, order
semantics, method names, and array shapes. Common client-handled error kinds include
`invalidThreadId`, `threadNotLoaded`, `threadNotMaterialized`, `threadStoreInvalidRequest`,
`unknownNode`, `revisionConflict`, `taskRunning`, `chatTreeUnavailable`, and `internal`.

## Portability Constraints

### Domain

- **MUST** keep one pure reducer responsible for node facts, revisions, current selection, ancestor
  projection, visible turn IDs, and overlay flattening.
- **MUST** feed the reducer feature-owned events with primitive stable fields. Do not make it depend
  on a large upstream event enum, session orchestration, model clients, TUI types, or file I/O.
- **MUST** reject and diagnose duplicate nodes, missing parents, unknown current/finalize/summary
  targets, and non-increasing revisions without producing partial state.

### Persistence And Runtime

- **MUST** name exactly one durable authority for every fact. Rollout, sidecar, or a hybrid may be
  used, but a hybrid must assign one authority per field.
- **MUST** rebuild domain state and selected history from durable facts. Treat context snapshots as
  cache; a cache miss triggers deterministic rebuild, and only failure to rebuild is a recovery
  error.
- **MUST** capture parent context before child-only changes. Standalone compaction or other context
  rewrites must update the selected-node cache.
- **MUST** handle rollback by atomically pruning removed nodes and caches and moving current, or
  reject rollback before mutating history.
- **MUST NOT** let `CurrentNodeChanged` open or retarget a replay turn scope.

### Adapters

- **MUST** centralize turn start, completion, abort, persistence ordering, summary ownership, and
  root-session eligibility behind feature-owned lifecycle methods. Task orchestration should only
  call those methods.
- **MUST** scope a thread-spawn subagent's continuation nodes and selection to its cloned child
  tree. Never route those mutations back to the parent thread.
- **MUST** provide one authoritative set-current control path that owns running-task checks,
  revision checks, persistence, and history restoration.
- **MUST** make loaded and unloaded app-server reads, legacy and paginated history, resume paths,
  and supported UIs consume the same projection semantics.
- **MUST NOT** duplicate current, revision, visible-path, default-summary, or row-flattening rules
  in API or UI adapters.
- **PREFERRED** keep upstream model-stream details for summary generation in a feature-owned helper
  so model client API churn does not spread through lifecycle code.

## Summary Ordering

For eligible completed turns:

1. Finalize and durably persist the node.
2. Register summary-job ownership and cancellation.
3. Emit normal turn completion.
4. Start the asynchronous summary request.
5. Normalize a completed response to one non-empty line of at most 96 characters.
6. Persist the late summary update and notify clients.

Late summary updates may change only the node label. Aborted turns and turns without assistant text
keep deterministic bounded fallback labels.

## Adapter Seams

Rediscover these semantics on every upstream base:

- accepted root user turn before model-history mutation;
- completed and aborted turn finalization;
- durable append/flush and cold replay;
- compaction, rollback, resume, and shutdown;
- current-node control and running-task state;
- thread read/list pagination and loaded/unloaded projection;
- app-server RPC registration and notifications;
- TUI command, overlay, stale-revision handling, and transcript refresh;
- model request creation and cancellation for summaries.

## P0 Acceptance

### Branch And Real Model Context

Run A -> B -> C, switch to A, then send D. Prove D is a child of A and capture the next real model
request. It must contain A/D and exclude B/C. `thread/read`, `thread/turns/list`, and
`visibleTurnIds` must agree.

### Lifecycle And Replay

Create completed, interrupted, replaced, and review-ended nodes; change current; apply a late
summary; restart and resume. Prove the full tree, statuses, revision, current, summaries, and
selected history survive. Exercise compaction and the declared rollback strategy.

### Summary And Source Isolation

Prove an eligible completed turn makes a separate no-tools summary request and persists the bounded
result. Prove failure, cancellation, empty output, and missing assistant output do not fail the
turn. Prove late updates leave context unchanged. Prove thread-spawn subagents append child-local
nodes to the inherited selected projection without mutating the parent or creating summary jobs;
other non-root turns create no node.

### Public Paths

Use wire fixtures or integration tests for read, set-current success, unknown node, revision
conflict, and updated notifications. Exercise each supported UI path for open, switch, running-task
rejection, transcript refresh, and stale-revision protection.

Projection-only checks or manual UI inspection alone are insufficient.

## Integration Contract

- Turn-loop features that can replace the terminal assistant message, including empty-final
  recovery, must settle before chat-tree completion captures the message for summary.
- Subagent identity and multi-agent changes must preserve non-root exclusion.
- App-server schema generation, TUI snapshots, and downstream release automation remain owned by
  their normal crate or integration workflows.
