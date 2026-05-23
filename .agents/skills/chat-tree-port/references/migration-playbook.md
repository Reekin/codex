# Migration Playbook

Follow this when porting chat tree to a new upstream release.

## 1. Prepare Branches

1. Fetch upstream and tags.
2. Identify the latest stable upstream tag.
3. Identify the direct previous chat-tree release branch.
4. Create the new migration branch from that previous chat-tree branch.
5. Rebase the previous branch's downstream commits onto the new upstream stable tag.
6. Keep older chat tree branches available for reference after the direct predecessor has been inspected.
7. Start SmartTakeover for continuous acceptance and provide the context template from `acceptance-checklist.md`.
8. Do not start by applying old patches to a clean upstream tag unless the rebase has already proven architectural.

Default command shape:

```bash
git fetch upstream --tags
git fetch origin
git switch <previous-chat-tree-release-branch>
git switch -c <new-chat-tree-release-branch>
git rebase --onto <new-upstream-stable-tag> <previous-upstream-stable-tag> <new-chat-tree-release-branch>
```

If the rebase conflicts become architectural, stop before switching strategies. Write an inventory from the direct previous branch covering:

- downstream commits and their intent;
- skill and reference docs;
- release workflow files;
- domain/protocol types;
- rollout/persistence events and replay;
- core session/history adapters;
- completed-turn LLM summary implementation;
- app-server protocol, schema, fixtures, and compatibility docs;
- TUI or other supported UI adapters;
- tests and validation commands.

Only then create a clean port from the new stable tag. The clean port must account for every inventory item, either by carrying it forward, replacing it with a documented equivalent, or explicitly classifying it as no longer applicable.

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

Also decide which UI entrypoints are supported on this upstream base. A UI path is supported only if it can open the tree, set current, refresh the selected-branch transcript, and prevent stale projections from rolling back local state. Disable, redirect, or document any path that cannot meet that bar.

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
2. required async LLM summary for completed normal turns with non-empty assistant output;
3. late summary update notification;
4. summary persistence/replay.

Fallback-only labels are an intermediate implementation state, not a completed migration. Preserve the previous release's LLM summary behavior unless the product contract is explicitly changed.

## 9. Validate

Use `acceptance-checklist.md` as the single completion checklist. Use `test-matrix.md` for detailed scenario coverage and command ideas.

## 10. Prepare Release Handoff

Use `release-automation.md`.

Before pushing a completed migration branch:

- confirm the branch contains `.github/workflows/windows-codex-branch-release.yml`;
- keep this downstream workflow separate from upstream official release workflows;
- update only the workflow's build/staging adapter if upstream changed the build layout;
- run a lightweight workflow sanity check.

If a clean upstream tag does not contain the workflow, copy it from the current chat-tree maintenance branch before the final push.

## 11. Triage Review Feedback

Before starting another refactor round, classify review items using `acceptance-checklist.md`.

Do not keep reshaping a working implementation unless the proposed change is tied to a contract rule, a test-matrix gap, or a specific future-portability risk.

## 12. Update Skill References

Add reusable findings to:

- `implementation-guide.md` for end-to-end implementation process and environment pitfalls;
- `v116-review.md` for newly discovered pitfalls;
- `architecture.md` for new upstream adaptation patterns;
- `test-matrix.md` for new regression scenarios.
- `release-automation.md` for reusable release handoff lessons.

## Common Mistakes

- Copying old file-level diffs before understanding new hook points.
- Letting UI local state become authoritative.
- Building tree separately in core, app-server, and UI.
- Testing only native TUI while app-server TUI is the default entrypoint.
- Treating model-generated summary as optional after branch correctness passes.
- Checking only an older reference branch and missing behavior that already existed in the direct previous release branch.
- Capturing parent snapshots after child-turn context updates.
- Forgetting interrupted turns.
- Forgetting resume/replay.
- Hiding stale current-node errors behind fallback behavior.
- Letting review loops continue without classifying findings as P0/P1/P2.
- Claiming native/app-server TUI parity when one path refreshes model context but not visible transcript.
- Treating antivirus, symlink, or lockfile noise as feature bugs before checking the local Windows environment.
- Finishing and pushing a migration branch without the downstream branch-release workflow, leaving users with no fresh downloadable binary.
