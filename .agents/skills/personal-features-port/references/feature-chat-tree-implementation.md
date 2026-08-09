# Chat Tree Implementation Guide

Follow this guide when implementing chat tree on a new upstream base. It is written as an execution path, not as background notes.

The objective is to produce the same user-facing feature behavior on a changed architecture with the smallest reliable adapter layer. Prefer carrying forward the stable domain/API/test implementation. Do not blindly replay old diffs across changed architecture; when rebase conflicts become architectural, start from the contract, map the new base, bring over reusable pieces, then attach adapters and tests.

## Target Outcome

Use `feature-chat-tree-acceptance.md` as the source of truth for completion. This guide explains how to implement toward that checklist without turning review feedback into architecture churn.

## Reference Technical Design

Use this design as the default implementation target. Change it only when the upstream base makes a part impossible, and document the replacement in the implementation brief.

## Copyable vs Adapter-Specific Work

Future ports should assume most chat-tree logic is copyable and only adapter boundaries require upstream-specific reasoning.

This means a full rewrite is not the default. A normal git rebase is appropriate when the reusable layers apply cleanly and conflicts are localized. A clean reimplementation branch is appropriate when upstream has changed the session, rollout, app-server, or TUI architecture enough that resolving the rebase would amount to guessing adapter semantics.

### Copyable Logic

These parts should be implemented almost mechanically from this guide or copied from the current reference implementation when language/crate layout allows:

- domain data model;
- durable domain event vocabulary;
- reducer API and invariants;
- revision monotonicity rules;
- projection algorithm;
- overlay flattening;
- deterministic fallback summary rules;
- summary prompt shape, normalization rules, and late summary event semantics;
- app-server wire method names and required fields;
- stable error `data.kind` values;
- golden test scenarios and fixture shapes.

If a future implementation changes any of these, treat it as a design deviation and document why it is necessary.

### Adapter-Specific Logic

These parts must be re-discovered for each upstream base:

- where normal user turns start;
- where turns finalize, abort, get replaced, or end review;
- how model-visible history is assembled;
- how runtime context/history is represented;
- how context rewrites such as compaction and rollback are expressed;
- how durable data is appended and flushed;
- how resume/replay reconstructs session state;
- how to spawn background summary work after completed turn finalization without blocking the main turn;
- how app-server registers RPC methods and notifications;
- how the active TUI path refreshes visible transcript.

The migration task is primarily to connect these upstream-specific hooks to the stable domain and API design. Do not redesign the domain unless a hook cannot satisfy a documented invariant.

### Expected Porting Shape

The normal porting loop should be:

1. create the migration branch from the direct previous chat-tree release branch;
2. rebase that branch's downstream commits onto the new upstream stable tag;
3. resolve localized adapter conflicts while preserving reusable domain, app-server wire shapes, fixtures, tests, skill docs, and summary behavior;
4. map changed upstream hooks;
5. write thin adapters around those hooks;
6. run the golden tests;
7. confirm the feature branch is ready to merge or replay into the versioned integration branch;
8. only then adjust design if an invariant cannot be satisfied.

If the implementation starts by changing domain semantics, app-server field names, projection rules, or overlay behavior, stop and justify the deviation before continuing.

### Choosing Rebase vs Clean Port

Prefer **rebase** when:

- domain/projection files apply with minimal conflict;
- app-server method names and wire structs still fit the upstream protocol system;
- test fixtures remain meaningful;
- conflicts are mostly imports, module paths, registration sites, or small API call changes;
- the turn lifecycle and replay model are recognizably the same.

Prefer **clean port with reused modules** when:

- turn lifecycle moved to a different abstraction;
- model-visible history is assembled differently;
- rollout/replay event shape or ordering changed substantially;
- app-server registry/schema generation changed substantially;
- TUI state/rendering moved to a different architecture;
- resolving git conflicts would require mixing old adapter assumptions into new upstream code.

Before choosing a clean port, produce the direct previous branch inventory described in `common-migration-playbook.md`. In both modes, the copyable domain/API/projection/test contract, skill docs, and completed-turn LLM summary behavior should remain stable. The difference is only whether Git can carry those pieces forward mechanically or the agent needs to place them into the new architecture by hand.

### Layering

Implement chat tree as five layers:

1. **Domain**: pure reducer, projection, validation, and overlay flattening.
2. **Core adapter**: connects domain facts to session state, model-visible history, and runtime context cache.
3. **Persistence adapter**: appends and replays durable facts from rollout or sidecar storage.
4. **App-server adapter**: exposes stable read/set-current/notification APIs and selected-branch transcript reads.
5. **UI adapter**: renders projections and submits set-current; it never owns tree truth.

Only the domain layer defines tree semantics. Other layers translate upstream data into domain events and consume domain projections.

### Domain Data Model

The domain model should be small and stable:

```text
type NodeId = String
type TurnId = String
type Revision = u64

ChatTreeNode {
  node_id: NodeId,
  parent_node_id: Option<NodeId>,
  turn_id: Option<TurnId>,
  order: u64,
  status: NodeStatus,
  summary: Option<String>,
}

NodeStatus = Pending | Completed | Interrupted | Replaced | ReviewEnded

ChatTreeState {
  revision: Revision,
  current_node_id: Option<NodeId>,
  nodes_by_id: Map<NodeId, ChatTreeNode>,
  node_order: Vec<NodeId>,
  legacy_visible_turn_ids: Vec<TurnId>,
  diagnostics: Vec<ChatTreeDiagnostic>,
}
```

`legacy_visible_turn_ids` is optional compatibility state for threads that contain linear turns before chat tree was enabled. Do not use it for new chat-tree nodes.

### Durable Domain Events

Use domain-owned events with primitive fields:

```text
ChatTreeEvent::NodeStarted {
  revision,
  node_id,
  parent_node_id,
  turn_id,
  order,
}

ChatTreeEvent::NodeFinalized {
  revision,
  node_id,
  status,
}

ChatTreeEvent::NodeSummaryUpdated {
  revision,
  node_id,
  summary,
}

ChatTreeEvent::CurrentNodeChanged {
  revision,
  node_id,
}

ChatTreeEvent::TreePruned {
  revision,
  surviving_current_node_id,
  removed_node_ids,
}
```

`TreePruned` is needed only if rollback is supported. If rollback is rejected on chat-tree threads, omit it and enforce the rejection before history mutation.

Do not wrap upstream protocol structs in `ChatTreeEvent`. Put conversions in adapter code:

```text
upstream rollout/protocol event -> ChatTreeEvent -> ChatTreeState::apply()
```

### Reducer API

The domain should expose a narrow API:

```text
ChatTreeState::new() -> ChatTreeState
ChatTreeState::apply(event: ChatTreeEvent) -> ApplyOutcome
ChatTreeState::projection() -> ChatTreeProjection
ChatTreeState::overlay_entries() -> Vec<ChatTreeOverlayEntry>
ChatTreeState::contains_node(node_id) -> bool
ChatTreeState::current_revision() -> Revision
```

`ApplyOutcome` should contain:

```text
ApplyOutcome {
  applied: bool,
  change: Option<ChatTreeChange>,
  diagnostic: Option<ChatTreeDiagnostic>,
  projection: ChatTreeProjection,
}
```

Adapters should not manually update revision, current node, visible path, or overlay rows. They call the reducer and consume the outcome.

### Reducer Invariants

`apply()` must enforce:

- event revision is greater than the current revision for durable mutating events;
- duplicate `NodeStarted.node_id` is rejected;
- `NodeStarted.parent_node_id` exists unless it is null;
- `NodeFinalized.node_id` exists;
- `NodeSummaryUpdated.node_id` exists;
- `CurrentNodeChanged.node_id` exists;
- `TreePruned.removed_node_ids` exist or produce diagnostics;
- current node is never set to a removed or unknown node;
- summary changes never change visible turn IDs;
- invalid events are ignored with diagnostics, not partially applied.

The reducer may allow revision gaps to support future rebuild/import events, but it must reject same-revision and lower-revision mutations.

### Projection Algorithm

`projection()` is deterministic:

1. If `current_node_id` is null, return no visible nodes and only explicit legacy visible turns if compatibility mode requires them.
2. Walk from `current_node_id` through `parent_node_id` until root.
3. Detect cycles and missing parents; stop with diagnostics instead of looping forever.
4. Reverse the path to root-to-current order.
5. `visible_node_ids` is the node path.
6. `visible_turn_ids` is `legacy_visible_turn_ids` followed by each path node's non-null `turn_id`.
7. `nodes` is every node in `node_order`, not only visible nodes.
8. Overlay entries are built from `nodes` in chronological sibling order, with depth computed from parent links.

No core, app-server, or TUI code should reimplement this algorithm.

### Core Runtime Cache

Core keeps context outside the domain:

```text
ChatTreeRuntime {
  domain: ChatTreeState,
  history_cache_by_node: Map<NodeId, ContextManager>,
}
```

Cache rules:

- node start stores the parent/current context snapshot for the new node before child-turn mutations;
- turn finalization stores the completed/interrupted node context;
- set-current restores from `history_cache_by_node[node_id]`;
- compaction replaces the current node cache;
- rollback/prune removes caches for removed nodes;
- replay rebuilds caches from turn-scoped rollout records;
- cache misses are adapter failures or rebuild triggers, not domain failures.

If a future base has a good immutable history representation, replace full snapshots with ancestor turn spans or shared-prefix fragments without changing the domain layer.

### Core Operation Sequence

Normal user turn:

```text
parent = domain.current_node_id
node_id = turn_id or generated stable node id
event = NodeStarted(next_revision, node_id, parent, turn_id, next_order)
persist_and_flush_if_required(event)
domain.apply(event)
history_cache_by_node[node_id] = current_context_before_child_updates
record user/context updates
send model request using domain.projection().visible_turn_ids + current new turn
on completion/interruption: persist NodeFinalized and update node cache
```

Set current:

```text
reject if task running
reject if expected_revision != domain.revision
reject if node_id unknown
event = CurrentNodeChanged(next_revision, node_id)
persist and flush event
domain.apply(event)
restore runtime context cache for node_id
return projection
emit notification after projection includes event
```

Replay:

```text
domain = ChatTreeState::new()
active_turn_node = None
for rollout item in order:
  if item is chat-tree event:
    domain.apply(convert(item))
    if item is NodeStarted: active_turn_node = Some(node_id)
    if item is NodeFinalized for active_turn_node: active_turn_node = None
  else if item is turn-scoped context/history and active_turn_node exists:
    update cache for active_turn_node
  else if item is standalone compaction:
    update cache for domain.current_node_id
  else if item is rollback:
    apply prune/rebuild strategy
```

`CurrentNodeChanged` must not set `active_turn_node`.

### App-Server Wire Shape

The app-server adapter converts domain projection to the stable wire contract:

```text
chatTree/read(threadId) -> { threadId, chatTree: ChatTreeProjectionWire }
chatTree/setCurrent(threadId, nodeId, expectedRevision?) -> { threadId, chatTree }
chatTree/updated notification -> { threadId, change, chatTree }
```

Wire projection:

```text
{
  version: 1,
  revision,
  currentNodeId,
  visibleNodeIds,
  visibleTurnIds,
  nodes
}
```

Loaded and unloaded threads must use the same domain projection semantics. `thread/read(includeTurns=true)` and `thread/turns/list` filter transcript turns by `visibleTurnIds` when chat tree is present.

### UI State Shape

UI state should be projection-first:

```text
ChatTreeUiState {
  projection: ChatTreeProjection,
  selected_node_id: Option<NodeId>,
}
```

UI rules:

- rows come from shared overlay entries;
- selection defaults to current node;
- set-current sends `node_id` and current `revision`;
- set-current success replaces projection with server/core projection;
- transcript refresh happens immediately after successful current-node changes;
- notifications with `revision <= projection.revision` are ignored unless they are an identical seed.

## Phase 0: Fill The Migration Work Package

Before changing code, fill the required sections of `feature-chat-tree-migration-template.md`. The work package prevents later review loops from turning into architecture churn and gives other agents a stable handoff artifact.

Required sections before broad implementation edits:

- **Context**: upstream base, previous branch, new feature branch, target integration branch.
- **Reference source priority**: contract, previous feature branch, summary reference, historical lessons, upstream adapter placement.
- **Branch strategy**: rebase or clean port with reused modules.
- **Carry-forward inventory**: copied, adapted, replaced, omitted, and integration-only items.
- **Upstream hook map**: session, turn, replay, app-server, UI, compaction, and rollback hooks with timing evidence.
- **Architecture decisions**: supported UI paths, persistence source, app-server contract, summary implementation, rollback/compaction strategy, runtime context strategy.
- **Current validation commands**: focused tests and schema/snapshot commands expected for this port.

Do not implement until the work package identifies the active session, turn, replay, app-server, and UI hook points.

## Phase 1: Map The Upstream Base

Find the current equivalents of these integration points:

- session/thread state owner;
- user-turn start path;
- turn completion, interruption, replacement, and review-ended paths;
- model-visible history assembly;
- context/history mutation path before model requests;
- compaction and rollback handlers;
- rollout persistence writer;
- resume/replay reconstruction;
- app-server method registry;
- app-server notification routing;
- `thread/read` and paginated turns builder;
- native and app-server TUI slash-command paths;
- visible transcript refresh path.

Output a hook map before editing:

```text
turn start: <file/function>
turn finalize: <file/function>
set-current command path: <file/function>
model-visible history source: <file/function>
durable append + flush: <file/function>
rollout replay: <file/function>
app-server read/setCurrent/updated: <file/function>
TUI overlay + transcript refresh: <file/function>
compaction/rollback: <file/function>
```

If a hook is missing, decide whether to add a small adapter or mark the related feature path unsupported. Do not silently rely on UI-local state or replay side effects.

## Phase 2: Build The Shared Domain First

Install one portable chat-tree domain before wiring UI or app-server behavior.

The domain owns:

- `ChatTreeNode`;
- `ChatTreeEvent`;
- `ChatTreeState`;
- `ChatTreeProjection`;
- `ChatTreeOverlayEntry`;
- revision/current/parent validation;
- ancestor path projection;
- visible turn ID projection;
- overlay flattening;
- deterministic fallback labels.

The domain does not own:

- `ContextManager` or runtime history snapshots;
- TUI rendering types;
- app-server wire structs;
- model client types;
- file I/O or async tasks;
- large upstream event enums.

Use a small stable event enum with primitive fields:

```text
NodeStarted { revision, node_id, parent_node_id, turn_id, order }
NodeFinalized { revision, node_id, status }
NodeSummaryUpdated { revision, node_id, summary }
CurrentNodeChanged { revision, node_id }
TreeRebuilt/BranchPruned { revision, ... } when rollback or rebuild is supported
```

Adapters convert upstream protocol, rollout items, notifications, and UI events into this domain event shape. If upstream renames or reshapes its rollout events, the reducer should not change.

Required domain rules:

- reject duplicate node starts;
- reject missing parent references;
- reject current-node changes to unknown nodes;
- reject finalize/summary updates for unknown nodes;
- reject or diagnose non-increasing durable revisions;
- never let summary updates change visible context;
- produce diagnostics for ignored corrupted facts.

## Phase 3: Wire Core As A Runtime Adapter

Core attaches the domain to the real session/history system.

Required behavior:

1. At normal user-turn start:
   - capture the parent/current node before child-turn-specific context mutations;
   - create and persist `NodeStarted`;
   - make the new node current for future branch creation.
2. Before model request:
   - assemble model-visible history from the selected ancestor path plus the new turn;
   - exclude sibling branch turns.
3. At turn completion/interruption/replacement/review end:
   - persist `NodeFinalized`;
   - update runtime context cache for that node.
4. At set-current:
   - reject unknown nodes, stale expected revisions, and running-task switches;
   - append and flush the durable current-node event before returning success;
   - update runtime history/cache only after durable persistence succeeds.
5. At replay:
   - rebuild domain state from durable facts;
   - rebuild runtime context cache from turn-scoped rollout items;
   - never treat `CurrentNodeChanged` as opening a turn scope.

Full `ContextManager` snapshots are allowed only as a runtime cache. If used, every context mutation path must keep the cache coherent:

- parent snapshots are captured before child-turn mutations;
- standalone compaction updates the current node cache;
- rollback/tree prune removes pruned node caches;
- fork/resume injected context has an explicit mapping;
- missing cache is an adapter recovery failure, not a domain fact.

## Phase 4: Wire Persistence And Replay

Choose exactly one persistent fact source for core tree facts.

### Rollout

Use rollout when upstream already has stable event persistence and app-server thread reads consume rollout.

Requirements:

- append durable chat-tree events in causal order;
- flush before `setCurrent` success;
- replay produces the same projection as live state;
- app-server unloaded reads use the same domain reducer.

### Sidecar

Use a sidecar log when upstream rollout/protocol churn is high or upstream rejects extra rollout fields.

Requirements:

- store stable node/current/summary/rebuild facts;
- key sidecar data by thread ID and turn IDs;
- load sidecar together with thread history for app-server reads;
- preserve archive/export/backup behavior.

### Hybrid

Use hybrid only when rollout must contain minimal current facts and sidecar must contain heavier metadata. Document which source is authoritative for each field.

Do not create two authorities for nodes, current node, or revision.

## Phase 5: Wire App-Server As The Stable External API

External clients must not inspect rollout internals or TUI state.

Implement and preserve:

- `chatTree/read`;
- `chatTree/setCurrent`;
- `chatTree/updated`;
- selected-branch `thread/read`;
- selected-branch `thread/turns/list`;
- stable `data.kind` errors.

Rules:

- `chatTree/read` returns full projection for loaded and resumable threads.
- loaded and unloaded reads use the same domain projection rules.
- `chatTree/setCurrent` persists before success and returns updated projection.
- `chatTree/updated` includes a full projection that already reflects `change`.
- stale or same-revision notifications must not rewind clients.
- non-chat-tree app-server methods must keep their schemas and behavior.

After protocol changes:

- update README/API docs;
- regenerate schema/TypeScript fixtures;
- add wire-level tests for empty tree, populated tree, set-current success, errors, and notifications.

## Phase 6: Wire Supported UI Paths

A UI path is supported only if it can:

- open `/chattree`;
- render shared overlay entries;
- submit set-current with expected revision;
- refresh visible transcript to the selected branch;
- ignore stale projections/notifications;
- block unsafe switching while a task is running.

For app-server TUI:

- seed projection with `chatTree/read`;
- after set-current, use returned projection and refresh transcript with `thread/read` or equivalent selected-branch source;
- on newer `chatTree/updated`, update overlay state and refresh transcript when current branch changes or tree is rebuilt.

For native TUI:

- prefer routing through the same read/set-current/refresh path as app-server TUI;
- otherwise use the shared domain reducer/projector for live events;
- if selected-branch transcript refresh is unavailable, mark native `/chattree` unsupported or redirected.

Never keep a UI path that switches future model context while leaving the old branch transcript on screen.

## Phase 7: Add Summary Last

Branch correctness does not depend on model summaries, but a migration is not feature-complete without completed-turn LLM summary generation.

Recommended order:

1. deterministic fallback label;
2. persisted late summary event;
3. notification and replay support;
4. required async/model summary provider for completed normal turns with non-empty assistant output;
5. cancellation/failure tests.

Implement the async provider using `feature-chat-tree-summary.md`. If only fallback labels are implemented, stop and report the migration as incomplete. Do not document async/model summary as future work for a completed port.

## Phase 8: Validate In Dependency Order

Run the focused validation order from `feature-chat-tree-acceptance.md`. Use `feature-chat-tree-test-matrix.md` for additional cases when a touched layer needs broader coverage.

If a review finds a bug, add the narrowest test that would have caught it before refactoring.

## Phase 9: Triage Reviews And Stop

Use `feature-chat-tree-acceptance.md` for P0/P1/P2 definitions, stop conditions, and acceptance status. Keep SmartTakeover aligned to that checklist.

Do not continue reshaping working code only because a review describes a cleaner hypothetical architecture. Convert the suggestion into a contract failure, a test, or a documented future task.

## Environment And Repository Hygiene

Before treating failures as feature bugs, check the local environment:

- Windows Developer Mode may be required for symlink-heavy dependency/build steps.
- Antivirus may quarantine Rust test/build artifacts during linking; confirm whether the quarantined file is a fresh local build output before continuing.
- `Cargo.lock` may get workspace-version churn from cargo/clippy without dependency changes; inspect and revert noise unless dependencies changed.
- Keep path formats consistent in Windows + Git Bash; convert paths explicitly when passing them to Windows-native tools.

For long migrations, keep a checkpoint:

```text
branch:
upstream base:
supported UI paths:
changed crates/files:
tests run:
known P0:
known P1:
known P2:
environment blockers:
next command:
```

This checkpoint lets the next agent resume without replaying the entire development history.
