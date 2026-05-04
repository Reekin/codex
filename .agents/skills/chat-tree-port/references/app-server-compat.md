# Stable App-Server Contract

This file defines the downstream chat tree app-server API that external clients and scripts should target after the next redesign.

After this contract is implemented once, future upstream migrations must preserve it. Breaking changes require a new versioned method family.

## Design Goals

- Keep chat tree API separate from `thread/read` schema churn.
- Make external clients able to read the full tree and the active projection in one call.
- Make set-current return the updated projection without requiring a second read.
- Use one notification shape for all chat tree changes.
- Include enough fields that clients do not need to reverse-engineer rollout events.
- Keep future extension possible through optional additive fields.

## Stable Methods

Use these method names:

- `chatTree/read`
- `chatTree/setCurrent`

Do not rename these methods after adoption. If an incompatible redesign is unavoidable, add a new method family such as `chatTree.v2/read` and keep this contract available until external clients migrate.

## Stable Notification

Use one notification method:

- `chatTree/updated`

This notification carries the new full projection after any durable chat tree change. It replaces separate node/current notifications.

Live notifications should build the projection from authoritative in-memory chat tree state. If a future architecture can only rebuild from rollout files, wait until the projection contains the reported `change` without blocking the active turn loop.

## Core Types

### `ChatTreeNode`

```json
{
  "nodeId": "node-b",
  "parentNodeId": "node-a",
  "turnId": "node-b",
  "order": 1,
  "status": "completed",
  "summary": "Implement parser and tests"
}
```

Fields:

- `nodeId`: string, required. Stable node identifier.
- `parentNodeId`: string or null, required. Null means root-level node.
- `turnId`: string or null, required. Associated turn identifier.
- `order`: integer, required. Chronological node creation order, starting at 0.
- `status`: string, required. One of `pending`, `completed`, `interrupted`, `replaced`, `reviewEnded`.
- `summary`: string or null, required. Display label. Null means unavailable or pending.

### `ChatTreeProjection`

```json
{
  "version": 1,
  "revision": 7,
  "currentNodeId": "node-b",
  "visibleNodeIds": ["node-a", "node-b"],
  "visibleTurnIds": ["node-a", "node-b"],
  "nodes": []
}
```

Fields:

- `version`: integer, required. Must be `1` for this contract.
- `revision`: integer, required. Monotonically increases for each durable chat tree change in the thread.
- `currentNodeId`: string or null, required. Current selected node.
- `visibleNodeIds`: string array, required. Root-to-current path.
- `visibleTurnIds`: string array, required. Turn IDs projected into current transcript/model context.
- `nodes`: `ChatTreeNode[]`, required. Full tree in chronological node creation order.

`visibleNodeIds` and `visibleTurnIds` must be empty when there is no current node.

## `chatTree/read`

Request params:

```json
{
  "threadId": "THREAD_ID"
}
```

Response result:

```json
{
  "threadId": "THREAD_ID",
  "chatTree": {
    "version": 1,
    "revision": 3,
    "currentNodeId": "node-b",
    "visibleNodeIds": ["node-a", "node-b"],
    "visibleTurnIds": ["node-a", "node-b"],
    "nodes": [
      {
        "nodeId": "node-a",
        "parentNodeId": null,
        "turnId": "node-a",
        "order": 0,
        "status": "completed",
        "summary": "Inspect project"
      },
      {
        "nodeId": "node-b",
        "parentNodeId": "node-a",
        "turnId": "node-b",
        "order": 1,
        "status": "completed",
        "summary": "Implement parser"
      }
    ]
  }
}
```

Behavior:

- Returns the authoritative durable chat tree projection for the requested thread.
- Does not depend on TUI-local state.
- Works for loaded and resumable threads when their persisted data exists.
- Empty tree returns `chatTree.nodes: []`, `currentNodeId: null`, and empty visible arrays.

## `chatTree/setCurrent`

Request params:

```json
{
  "threadId": "THREAD_ID",
  "nodeId": "node-a",
  "expectedRevision": 3
}
```

Fields:

- `threadId`: string, required.
- `nodeId`: string, required.
- `expectedRevision`: integer or null, optional. When provided, the server should reject the request if the current chat tree revision differs.

Response result:

```json
{
  "threadId": "THREAD_ID",
  "chatTree": {
    "version": 1,
    "revision": 4,
    "currentNodeId": "node-a",
    "visibleNodeIds": ["node-a"],
    "visibleTurnIds": ["node-a"],
    "nodes": []
  }
}
```

Behavior:

- Sets the current node for future turns.
- Persists the current-node change before returning success.
- Returns the updated projection.
- Emits `chatTree/updated` after the durable change is visible to app-server clients.
- Must not allow switching to an unknown node.
- Should reject switching while a task is running unless the architecture supports safe mid-turn switching.

## `chatTree/updated`

Notification params:

```json
{
  "threadId": "THREAD_ID",
  "change": {
    "type": "currentNodeChanged",
    "nodeId": "node-a"
  },
  "chatTree": {
    "version": 1,
    "revision": 4,
    "currentNodeId": "node-a",
    "visibleNodeIds": ["node-a"],
    "visibleTurnIds": ["node-a"],
    "nodes": []
  }
}
```

`change.type` values:

- `nodeStarted`
- `nodeFinalized`
- `nodeSummaryUpdated`
- `currentNodeChanged`
- `treeRebuilt`

`change.nodeId` is string or null. It should be set when the change is about one node.

Notification behavior:

- Send after durable chat tree state changes.
- Include the full updated projection so external clients do not need immediate follow-up reads.
- Ensure the included projection already reflects the included `change`.
- Do not emit replayed historical changes as live notifications unless the app-server has a replay marker that external clients can distinguish.
- When a client connects and needs current state, it should call `chatTree/read`; do not rely on notification replay for initial state.

## Relationship to `thread/read`

`thread/read` remains the thread/transcript API.

Rules:

- Do not change existing `thread/read` field names or field types because of chat tree.
- When chat tree is enabled and turns are included, `thread/read` should return the current branch projection.
- External clients that need the full tree must call `chatTree/read`.
- External clients that need to change current branch must call `chatTree/setCurrent`.

## Error Contract

Use the app-server's existing JSON-RPC error code conventions, but include stable `data.kind` values when possible.

Recommended `data.kind` values:

- `invalidThreadId`
- `threadNotLoaded`
- `unknownNode`
- `revisionConflict`
- `taskRunning`
- `chatTreeUnavailable`
- `internal`

Example:

```json
{
  "code": -32600,
  "message": "unknown chat tree node: node-x",
  "data": {
    "kind": "unknownNode",
    "threadId": "THREAD_ID",
    "nodeId": "node-x"
  }
}
```

## Compatibility Rules After Adoption

Allowed:

- adding optional request fields;
- adding optional response fields;
- adding optional notification fields;
- adding new `change.type` values if old clients can safely ignore them;
- adding new methods while keeping this method family.

Not allowed:

- renaming stable methods;
- removing required fields;
- changing required field types;
- changing enum string values already listed here;
- changing `nodes` from array to map;
- changing `order` semantics;
- changing `revision` monotonicity;
- making `summary` absent instead of null;
- making `parentNodeId` absent instead of null;
- requiring external clients to consume UI-local state.

## Required Fixtures

Each future port should keep wire-level fixtures or tests for:

- empty tree read;
- populated tree read;
- set-current success;
- set-current unknown node error;
- set-current revision conflict;
- node summary update notification;
- current-node changed notification;
- non-chat-tree `thread/read` schema stability.

Fixtures should be plain JSON-RPC request/result/notification examples so external client implementations can test against them.

## Review Checklist

Before accepting an app-server chat tree port:

- Verify generated JSON schema and TypeScript schema match this contract.
- Verify `chatTree/read`, `chatTree/setCurrent`, and `chatTree/updated` exist.
- Verify all required fields are always serialized, with nulls instead of omission where specified.
- Verify `revision` increments on durable changes.
- Verify `chatTree/setCurrent` response includes the updated projection.
- Verify app-server clients can validate behavior without using TUI.
- Verify non-chat-tree app-server workflows still pass focused tests.
- Update this file only for additive extensions or explicitly versioned replacements.
