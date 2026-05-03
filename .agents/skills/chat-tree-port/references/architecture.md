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

### ChatTreeEvent

Durable event vocabulary should express feature facts directly:

- `NodeStarted { node_id, parent_node_id, turn_id, order }`
- `NodeFinalized { node_id, status }`
- `NodeSummaryUpdated { node_id, summary }`
- `CurrentNodeChanged { node_id }`

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
- blocking unsafe switches while task is running.

## Refactor Threshold

Refactor into a shared domain module when any of these are true:

- tree flattening appears in more than one file;
- replay logic appears in more than one file;
- current node is stored in more than one runtime cache;
- native TUI and app-server TUI require parallel code;
- tests need to duplicate large event sequences across crates.
