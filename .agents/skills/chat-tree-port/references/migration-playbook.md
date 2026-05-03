# Migration Playbook

Follow this when porting chat tree to a new upstream release.

## 1. Prepare Branches

1. Fetch upstream and tags.
2. Identify the latest stable upstream tag.
3. Create a new migration branch from that tag, for example `rebase/vX-chat-tree`.
4. Keep old chat tree branches available for reference.
5. Do not start by applying old patches.

## 2. Map the New Architecture

Find the current upstream equivalents of:

- session state;
- turn lifecycle start/complete/abort;
- model-visible history assembly;
- history recording and compaction;
- rollout persistence;
- resume/replay reconstruction;
- app-server RPC registry;
- app-server notification routing;
- native TUI slash commands;
- app-server TUI slash commands;
- overlay/modal rendering.

Write down the hook points before editing.

## 3. Install Domain Types

Add or locate shared types for:

- nodes;
- durable events;
- current node;
- projection;
- overlay entries.

Keep this layer independent from app-server schema types and ratatui types.

## 4. Wire Core Behavior

Implement:

- create node at normal user-turn start;
- capture parent before child-turn-specific context mutations;
- project model-visible history from current branch;
- finalize completed and aborted turns;
- persist current-node changes;
- replay all durable chat tree facts on resume.

Do not wire LLM summary until deterministic node labels and branch projection work.

## 5. Wire Persistence

Pick rollout, sidecar, or hybrid persistence using `architecture.md`.

Required replay output:

- same nodes;
- same parents;
- same current node;
- same visible turns;
- same summaries when summary events exist.

## 6. Wire API

If app-server exists, expose:

- full tree read;
- set current node;
- node/current notifications.

If app-server has changed radically, preserve command-line or test-level access to equivalent operations so migration can be validated without manual UI.

## 7. Wire UI

Implement `/chattree` in every active user-facing TUI path.

Avoid duplicating tree flattening logic. UI should consume projection entries.

## 8. Add Summary

Add summary only after core branch semantics pass tests.

Preferred sequence:

1. deterministic fallback;
2. optional async model summary;
3. late summary update notification;
4. summary persistence/replay.

## 9. Validate

Use `test-matrix.md`.

Minimum before considering migration complete:

- live branch creation;
- branch switching affects next model context;
- resume restores current node;
- app-server read/set-current works;
- native/app-server TUI behavior matches;
- late summary does not alter context.

## 10. Update Skill References

Add reusable findings to:

- `v116-review.md` for newly discovered pitfalls;
- `architecture.md` for new upstream adaptation patterns;
- `test-matrix.md` for new regression scenarios.

## Common Mistakes

- Copying old file-level diffs before understanding new hook points.
- Letting UI local state become authoritative.
- Building tree separately in core, app-server, and UI.
- Testing only native TUI while app-server TUI is the default entrypoint.
- Treating model-generated summary as required for correctness.
- Capturing parent snapshots after child-turn context updates.
- Forgetting interrupted turns.
- Forgetting resume/replay.
- Hiding stale current-node errors behind fallback behavior.
