# v116 Reference Review

Use `rebase/v0.116-chat-tree` as the best current behavior reference, not as a patch template.

## What v116 Gets Right

- Implements normal turn nodes.
- Adds `/chattree`.
- Adds branch switching.
- Projects `thread/read` to selected branch.
- Persists current-node changes.
- Adds late summary update events.
- Handles interrupted turns.
- Adds app-server RPC methods.
- Adds app-server notifications.
- Adds native TUI and app-server TUI support.
- Adds resume/replay coverage.

## Main Design Risks

### Snapshot Timing

The v116 implementation creates a child node after some turn context updates are recorded. This risks saving child-turn context into the parent snapshot.

Future ports must capture parent context before child-turn-specific mutations.

### Full Snapshot Per Node

Each node stores a full context snapshot. This is simple but can grow quickly in branchy long sessions.

Prefer path/delta reconstruction or shared-prefix context with cache.

### Scattered Domain Logic

Chat tree rules are spread across:

- core session state;
- task lifecycle;
- rollout replay;
- app-server thread history;
- native TUI;
- app-server TUI.

Future ports should centralize tree construction, projection, and overlay flattening.

### Summary Coupling

v116 uses an async model request after each completed turn to generate node summaries.

Future ports should make summary optional and injectable. Deterministic fallback must be enough for correctness.

### Duplicate Facts

`TurnAborted` can carry chat tree metadata and then a separate node update can be emitted immediately afterward.

Future ports should avoid duplicate events unless the second event truly changes metadata.

### UI State Duplication

Native TUI maintains a local tree from core events. App-server TUI reads tree via app-server RPC and refreshes after notifications.

Future ports should make UI consume a shared projection instead of reimplementing state logic.

### Current Node Fact Sources

v116 has current node in both durable rollout events and live runtime state. App-server reads may overlay live state on rollout-derived state.

Future ports should define fact source priority explicitly and minimize dual authority.

## v116 Files Worth Consulting

- `chat-tree-PRD.md`: concise behavior goal.
- `codex-rs/core/src/state/session.rs`: runtime chat tree state and replay.
- `codex-rs/core/src/tasks/mod.rs`: turn finalize and async summary.
- `codex-rs/core/src/codex.rs`: current-node op, startup restore, node creation hook.
- `codex-rs/protocol/src/protocol.rs`: core op/event payloads.
- `codex-rs/app-server-protocol/src/protocol/common.rs`: method and notification registration.
- `codex-rs/app-server-protocol/src/protocol/v2.rs`: app-server chat tree API types.
- `codex-rs/app-server-protocol/src/protocol/thread_history.rs`: tree builder and branch projection.
- `codex-rs/app-server/src/codex_message_processor.rs`: app-server read/set-current.
- `codex-rs/app-server/src/bespoke_event_handling.rs`: server notification translation.
- `codex-rs/tui/src/chatwidget.rs`: native TUI local event handling.
- `codex-rs/tui/src/pager_overlay.rs`: overlay behavior.
- `codex-rs/tui_app_server/src/app.rs`: app-server TUI tree loading and set-current refresh.

## Behavior Fixes Added After Initial v116 Port

- Replay and app-server TUI parity.
- Summary persistence and live app-server updates.
- Thread turns refresh before chat tree replay.
- Resume fixtures for chat tree summaries.
- Overlay visibility and scroll behavior.
- Current-node resume coverage.
- Late summary projection preservation.
- Stale live current-node defense in app-server `thread/chatTree/read`.

These fixes should become tests in future ports rather than manual QA notes.

## Do Not Copy

- v114 README replacement and demo-only files.
- v116 unrelated Windows `apply_patch` temp environment fix as part of chat tree.
- Duplicated overlay/flatten implementations.
- Full-snapshot-per-node as a permanent architecture without a memory plan.
