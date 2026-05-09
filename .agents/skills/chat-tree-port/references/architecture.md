# Stable Architecture Guidance

Use this design when implementing chat tree on a new upstream architecture.

## Core Principle

Separate chat tree domain rules from Codex integration points.

The domain layer should be portable. Upstream-specific code should be thin adapters.

## Recommended Modules

### ChatTreeEngine

A pure reducer that accepts durable chat tree events and returns state/projections.

Responsibilities:

- validate parent/current references;
- create and finalize nodes;
- update summary;
- set current node;
- compute ancestor path;
- compute visible turn IDs;
- flatten tree for UI overlay.

Avoid:

- async;
- model client calls;
- TUI types;
- app-server protocol types;
- direct file I/O;
- concrete `ContextManager` ownership.

## Reference Implementation Standard

Use this shape as the default target when porting to a new upstream base.

### Pure Domain Layer

Put the chat tree reducer in a low-level crate that core, app-server, and TUI can all depend on.

The domain layer owns:

- `ChatTreeNode`;
- `ChatTreeState`;
- `ChatTreeProjection`;
- `ChatTreeOverlayEntry`;
- revision checks;
- parent/current validation;
- durable event replay;
- visible ancestor path projection;
- visible turn ID projection;
- overlay flattening in chronological sibling order.

The domain layer must not own runtime context snapshots, model history objects, app-server wire types, or TUI rendering types.

The reducer should accept a small, stable chat-tree event enum. Upstream protocol events, rollout items, app-server notifications, and TUI native events should be converted into that enum by adapter code. Do not make the reducer match a large upstream event enum directly.

Recommended event shape:

- use primitive stable fields (`revision`, `node_id`, `parent_node_id`, `turn_id`, `order`, `status`, `summary`);
- do not wrap upstream protocol structs inside the domain event enum;
- keep protocol/rollout conversion in an adapter helper so upstream event renames or payload reshapes do not require changing reducer semantics;
- reject or ignore stale revisions rather than allowing replay or UI seed paths to rewind a projection.

Durable event replay must validate facts before mutating state:

- node-start events with duplicate node IDs are invalid;
- node-start events with missing parents are invalid;
- current-node changes to unknown nodes are invalid;
- finalize and summary events for unknown nodes are invalid;
- invalid durable events should be ignored with diagnostics, not turned into silent partial projections.

### Core Adapter

Core should wrap the pure domain state with runtime-only caches.

Recommended shape:

- `domain: ChatTreeState`;
- `history_snapshots: HashMap<NodeId, ContextManager>` or equivalent runtime cache;
- start/finalize/set-current methods delegate node/current/status/revision rules to the domain;
- context snapshots are read and written only by the core adapter;
- `MissingSnapshot` is an adapter/cache failure, not a domain fact.

On replay, reconstruct the domain state from durable chat tree events and reconstruct cache entries from turn-scoped rollout items. `CurrentNodeChanged` must not open a turn scope; only items inside a matching started turn can mutate a node's context cache.

Context rewrites outside an active turn must still update the selected node's runtime cache. Standalone compaction should replace the current node's cached context, and subsequent turn-context records should update that same cache.

Legacy thread rollback must have an explicit chat-tree strategy. If rollback is supported, replay it through the domain as a visible-branch prune, remove snapshots for pruned nodes, and move current to the newest surviving ancestor. If rollback is not supported, reject it before mutating history.

### App-Server Adapter

App-server should use one projector path for loaded and unloaded threads.

Recommended shape:

- convert core domain projection to app-server wire projection for loaded threads;
- build unloaded rollout projection by feeding durable events into the same domain reducer;
- filter `thread/read` and `thread/turns/list` using the domain projection's `visible_turn_ids`;
- verify `chatTree/updated` notifications are emitted only after the returned projection reflects the durable change;
- send a `treeRebuilt` notification after rollback or any other durable tree-wide rebuild;
- after `chatTree/setCurrent`, return the updated projection and make the selected branch transcript available via `thread/read`/`thread/turns/list`;
- keep stable error `data.kind` values for chat-tree methods.

### TUI Adapter

TUI should not rebuild tree paths or flatten rows manually.

Recommended shape:

- hold the latest `ChatTreeProjection`;
- apply raw live events through the shared reducer/projector when native event delivery is used;
- when a projection has already been seeded from `chatTree/read`, skip or ignore replayed chat-tree events whose revision is not newer than the seeded projection;
- render rows from shared `ChatTreeOverlayEntry` values;
- submit set-current operations with the projection revision.
- after set-current succeeds, refresh the visible transcript from the selected branch before the user sends the next turn.
- when a current-node or tree-rebuilt notification arrives with a newer revision than the local projection, refresh the visible transcript as well as the chat-tree overlay state.

Do not duplicate revision, current-node, status, default-summary, visible-path, or row-flattening rules in TUI code.

If an upstream base no longer supports native event delivery, prefer a `chatTree/read` seed plus `chatTree/updated` notifications over adding another UI-side reducer.

## Supported UI Policy

For each upstream base, explicitly classify UI entrypoints before implementation:

- **Supported**: the path can open `/chattree`, submit set-current, refresh visible transcript to the selected branch, ignore stale projections, and block unsafe switching during running tasks.
- **Redirected**: the path uses another supported adapter, such as routing native UI through app-server read/set-current calls.
- **Unsupported**: the path should not expose `/chattree` as a working feature.

Do not keep a partial UI path that changes future model context but leaves the screen showing an old branch. That state is worse than no UI because users cannot tell which context the next turn will use.

When both native TUI and app-server TUI are supported, they must share the same projection and transcript-refresh contract. If they cannot, make app-server TUI the canonical user-facing path and document the native limitation.

### ChatTreeEvent

Durable event vocabulary should express feature facts directly:

- `NodeStarted { revision, node_id, parent_node_id, turn_id, order }`
- `NodeFinalized { revision, node_id, status }`
- `NodeSummaryUpdated { revision, node_id, summary }`
- `CurrentNodeChanged { revision, node_id }`

The exact enum names can follow upstream style. The point is to avoid deriving core facts from many unrelated UI or turn events when possible.

### ChatTreeProjection

Projection output should be the common input to adapters:

- full tree;
- current node;
- visible ancestor node IDs;
- visible turn IDs;
- overlay entries.

Both native TUI and app-server TUI should share this projection logic.

### SummaryProvider

Summary should be injectable:

- `DeterministicSummaryProvider`: prompt/status-based fallback, always available.
- `ModelSummaryProvider`: optional async model summary.
- `NoopSummaryProvider`: useful during first migration pass.

The feature must work with deterministic/noop summary before model summary is wired.

## Persistence Choices

### Option A: Events in Rollout

Pros:

- one persistent log;
- easy resume replay;
- existing app-server thread reads can reuse rollout.

Cons:

- more sensitive to upstream rollout/protocol churn;
- schema regeneration burden;
- event ordering bugs affect both transcript and chat tree.

Use when upstream rollout is stable and app-server already reads rollout.

### Option B: Sidecar Chat Tree Log

Pros:

- smaller upstream patch surface;
- stable downstream-controlled schema;
- easier to migrate across upstream protocol changes.

Cons:

- must keep sidecar and rollout turn IDs synchronized;
- app-server/thread export needs to load two sources;
- backup/archive behavior needs attention.

Use when upstream protocol churn is high or upstream rejects extra protocol fields.

### Option C: Hybrid

Keep minimal durable node/current events in rollout and heavier metadata/cache in sidecar.

Use only when there is a clear reason. Avoid creating two competing fact sources.

## Context Storage

Avoid long-term reliance on a full `ContextManager` clone per node.

Preferred approaches:

- store `turn_id` spans and rebuild context by ancestor path;
- store immutable shared-prefix context fragments;
- cache reconstructed node contexts with invalidation;
- use full snapshots only as temporary cache or migration shortcut.

If full snapshots are used:

- document them as cache;
- ensure parent snapshots are captured before child-turn mutations;
- test late summary updates do not alter snapshots;
- update the current node cache for standalone context rewrites such as compaction;
- prune caches when rollback removes visible nodes or descendants;
- treat missing cache as an adapter recovery problem, not a domain invariant violation;
- consider memory growth in long branchy sessions.

## Adapter Boundaries

### Core Adapter

Find hooks for:

- new user turn accepted;
- context/history mutation before model request;
- turn completion;
- turn abort;
- session resume/replay;
- shutdown/cancel for async summary jobs.

### App-Server Adapter

Find hooks for:

- registering RPC methods;
- reading thread rollout/history;
- setting current node on loaded thread;
- sending notifications;
- refreshing projected thread turns.

### UI Adapter

Find hooks for:

- slash command registration;
- opening overlay;
- rendering overlay entries;
- submitting set-current operation;
- refreshing visible transcript after set-current/current-node notifications;
- blocking unsafe switches while task is running.

## Refactor Threshold

Refactor into a shared domain module when any of these are true:

- tree flattening appears in more than one file;
- replay logic appears in more than one file;
- current node is stored in more than one runtime cache;
- native TUI and app-server TUI require parallel code;
- tests need to duplicate large event sequences across crates.
