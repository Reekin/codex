# Chat Tree Contract

This file defines the behavior that must remain stable across upstream Codex architecture changes.

## Product Behavior

- Conversation history is represented as a tree of user turns.
- Each normal user prompt creates a new node under the current node.
- A node contains the user prompt for that turn and all assistant/tool/context changes produced until that turn completes or aborts.
- The current node determines the context for future turns.
- The user can open `/chattree`, inspect all nodes, and choose a different current node.
- Choosing a different current node does not delete or mutate other nodes.

## Node Fields

- `node_id`: stable string ID, usually the turn/submission ID.
- `parent_node_id`: parent node ID, or `None` for root-level turns.
- `turn_id`: associated turn ID when different from `node_id` or when API consumers need turn mapping.
- `summary`: optional display label.
- `status`: pending, completed, interrupted, replaced, or review-ended when the architecture supports it.
- `order`: chronological creation order.

Use an internal root sentinel only inside the domain layer. Do not expose a fake root node unless the UI contract intentionally changes.

## Turn Lifecycle

- Before a new user turn mutates model-visible history, capture the current node as the parent.
- Create the child node for that user turn.
- Apply the user turn's context/user/tool/model changes to the child node path.
- On successful completion, finalize the node.
- On interruption, replacement, or review end, finalize the node with aborted status.
- Do not represent subagent/side-thread turns as main chat tree nodes unless the product contract is explicitly changed.

Important invariant: parent node context must not contain child-turn-only context updates.

## Context Projection

- The visible branch for a current node is the path from root to current node.
- Model requests must be assembled from only that visible branch plus the active new turn.
- `thread/read` with turns should show the same selected branch projection.
- `chatTree/read` should return the whole tree and current projection.
- Unknown or unmapped turns in a chat-tree-enabled thread are suspicious. Avoid default-visible behavior except for explicit legacy compatibility.

## Persistence

The persistent fact source must recover:

- all node IDs;
- parent relationships;
- current node changes;
- turn-to-node mapping;
- node status;
- summaries;
- chronological order.

Runtime memory, UI caches, and context snapshots are not authoritative. They must be rebuildable from durable data.

## Current Node Semantics

- Current node changes must be durable.
- Current node changes must be visible to API consumers and UI clients.
- Switching current node must update future model context.
- Switching current node must not cancel late summary jobs for completed nodes.
- If a durable current node points to a missing node, prefer a safe fallback and surface a diagnostic instead of silently corrupting the projection.

## Summary Semantics

See `feature-chat-tree-summary.md` for the required LLM summary flow.

- Summary is a display enhancement, not a context primitive.
- During a node lifecycle, summary can be missing, pending, deterministic, or model-generated.
- Completed normal turns with non-empty assistant output must schedule an asynchronous LLM summary request after turn completion.
- The model-generated result must be normalized to a bounded single-line label and persisted as a summary update.
- Deterministic labels are required fallback behavior for pending summaries, aborted turns, empty assistant output, empty model output, cancellation, or model failure.
- Summary generation failure must not fail the turn.
- Late summary updates must not alter node context snapshots or branch projection.
- Aborted turns should have deterministic fallback labels such as `turn interrupted`, `turn replaced`, or `turn review ended`.

## UI Behavior

- Slash command: `/chattree`.
- Empty tree message: tell the user to send a prompt first.
- Overlay shows all nodes in chronological sibling order.
- Overlay uses indentation for depth.
- Current node is marked as `[*]`; other nodes use `[ ]`.
- Selected row is indicated with `>`.
- `↑`/`↓` and preferably `k`/`j` move selection.
- Space sets current node and closes the overlay.
- `q`, Esc, and Ctrl-C close without changing current node.
- The current node is selected when opening; otherwise select the newest node.
- Long summaries wrap.
- The selected row must scroll into view.
- Switching current node while a task is running should be blocked unless the architecture explicitly supports safe mid-turn switching.

## API Behavior

Use these app-server methods when app-server exists:

- `chatTree/read`: returns full tree, current node, revision, and current projection.
- `chatTree/setCurrent`: sets current node and returns the updated projection.

Use this app-server notification when app-server supports notifications:

- `chatTree/updated`: emits the updated full projection after node/current/summary changes.

The API should not require UI-specific state to answer correctly.
