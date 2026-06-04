# Test Matrix

Use this matrix for detailed test coverage. Use `feature-chat-tree-acceptance.md` as the final migration acceptance checklist.

## Golden Scenario Corpus

Build at least one reusable scenario with these steps:

1. Send prompt A; node A is created and completed.
2. Send prompt B from A; node B is child of A.
3. Send prompt C from B; node C is child of B.
4. Open chat tree and switch current node back to A.
5. Send prompt D; node D is child of A, sibling of B.
6. Switch current node to B.
7. Send prompt E and interrupt it; node E is child of B and aborted.
8. Receive late summary update for B.
9. Restart/resume the thread.
10. Verify current node, tree, summaries, and visible transcript.

Expected properties:

- full tree contains A, B, C, D, E;
- B parent is A;
- C parent is B;
- D parent is A;
- E parent is B;
- current node after replay matches last durable current-node change or last created node if no switch occurred;
- visible turns for current D are A, D;
- visible turns for current C are A, B, C;
- visible turns for current E are A, B, E;
- late summary for B changes only B's label;
- no sibling branch leaks into model-visible context.

## Core Unit Tests

- Creating first node produces no external parent.
- Creating child captures current node as parent.
- Switching current restores the selected branch.
- Unknown current-node switch returns a user-facing error.
- Completed turn finalizes node.
- Interrupted/replaced/review-ended turns finalize nodes.
- Late summary update does not mutate node context.
- Parent snapshot is not polluted by child-turn context updates.
- Subagent/side-thread turns are excluded unless product contract changes.

## Replay Tests

- Replay reconstructs full tree from durable facts.
- Replay restores current node changes.
- Replay handles late summary after turn completion.
- Replay handles interrupted turn node.
- Replay handles compacted history on one branch.
- Replay handles standalone compaction after a tree exists and keeps the current node context cache consistent.
- Replay handles rollback by pruning removed visible nodes or reports that rollback is unsupported before mutating history.
- Replay handles forked thread initial context without polluting old branch snapshots.
- Replay handles stale/missing current node with deterministic fallback or diagnostic.
- Replay rejects or diagnoses non-increasing durable revisions.
- Replay rejects or diagnoses duplicate node start, missing parent, and unknown current-node facts.
- Mixed legacy linear history has explicit compatibility behavior.

## Projection Tests

- Current node A shows A only.
- Current node B shows A, B.
- Current node D after branching from A shows A, D and not B/C.
- Unknown current node does not silently expose all branch turns in chat-tree-enabled history.
- Unmapped turns in chat-tree-enabled history are either legacy-visible by explicit rule or treated as corruption.

## App-Server Tests

- `chatTree/read` returns full tree, current node, revision, and visible projection.
- `chatTree/setCurrent` changes current node and returns the updated projection.
- After `chatTree/setCurrent`, `thread/read` with turns returns the same selected branch as `chatTree.visibleTurnIds`.
- After `chatTree/setCurrent`, `thread/turns/list` returns the same selected branch as `chatTree.visibleTurnIds`.
- Set-current persists and survives restart.
- Set-current emits `chatTree/updated` with `change.type = currentNodeChanged`.
- Stale or same-revision `chatTree/updated` payloads do not rewind a newer client projection.
- Node summary update emits `chatTree/updated` with `change.type = nodeSummaryUpdated`.
- `thread/read` with turns returns selected branch projection.
- Loaded live thread current state and rollout current state do not diverge after flush/restart.
- App-server can serve tree for resumed thread.
- Stable `chatTree/read` request/response JSON fixtures match `feature-chat-tree-app-server.md`.
- Stable `chatTree/setCurrent` request/response JSON fixtures match `feature-chat-tree-app-server.md`.
- Stable `chatTree/updated` notification JSON fixtures match `feature-chat-tree-app-server.md`.
- Error responses include stable `data.kind` values when supported.
- Existing non-chat-tree app-server methods keep their schemas and behavior.

## TUI Tests

- `/chattree` opens overlay when tree is non-empty.
- Empty tree shows info message.
- Overlay selects current node by default.
- Space selects node and submits set-current operation.
- Esc/q/Ctrl-C closes without selection.
- Arrow keys and k/j move selection.
- Long labels wrap.
- Selected row scrolls into view.
- Current marker updates after set-current.
- Switching is blocked while a task is running.
- Set-current refreshes the visible transcript to the selected branch before the next user turn.
- Stale chat-tree notifications do not overwrite a newer overlay/current marker.

## Cross-UI Tests

- Native TUI and app-server TUI use the same overlay entries for the same tree.
- Native TUI and app-server TUI both block unsafe switch during running task.
- App-server TUI refreshes transcript after current-node change.
- Native TUI refreshes transcript after current-node change, or native `/chattree` is explicitly unsupported/redirected for that upstream base.
- App-server TUI updates pending summary labels without changing current transcript unless needed.

## Summary Tests

- Deterministic summary fallback always returns a bounded single-line label.
- Completed normal turns with non-empty assistant output spawn a separate async LLM summary request and persist `NodeSummaryUpdated`.
- Completed turns without assistant output do not spawn a model summary request.
- Model summary failure does not fail turn completion.
- Model summary cancellation does not fail turn completion.
- Empty model summary falls back or remains pending without breaking tree.
- Late summary update persists and replays.

## Validation Commands

Adjust crate names to the current upstream layout.

- `cargo test -p codex-core chat_tree`
- `cargo test -p codex-app-server-protocol chat_tree`
- `cargo test -p codex-app-server thread_chat_tree`
- `cargo test -p codex-tui chattree`
- `cargo test -p codex-tui-app-server chat_tree`

Run `just fmt` after Rust changes. Run schema generation after app-server protocol changes.
