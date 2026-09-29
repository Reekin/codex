# Stream Retry Timeout

## Goal

Keep a slow response eligible to finish while a concurrent replacement request gets a chance to
return a usable result sooner.

## Stable Contract

- **REQ-1**: The default initial waiting interval is five minutes. An explicit provider stream
  timeout supplies the initial interval instead. Each subsequent attempt adds five minutes.
- **REQ-2**: If no complete tool call or answer arrives within that interval and retry budget
  remains, start one concurrent replacement with the same prompt while retaining the original.
  Both remain eligible for the replacement's waiting interval. At most two requests run at once;
  the replacement consumes one retry. If both fail or the selection window expires, retry using
  the remaining budget. A new logical request resets the interval and budget.
- **REQ-3**: Adopt the first complete tool call, final answer, or compaction result. Reasoning,
  commentary, and partial output do not select a winner. A terminal response is also delivered so
  existing empty-answer handling remains effective. Once selected, the winner is fixed and the
  other request is canceled. If one candidate fails, keep waiting for the other.
- **REQ-4**: Candidate events are buffered until selection, so initial text and reasoning become
  visible together with the selected complete result. Only that stream enters history and tool
  execution; subsequent events resume normal streaming. Canceling the turn cancels both candidates.
- **REQ-5**: Sampling and streaming remote compaction share this policy over HTTP SSE and
  WebSockets. Transport fallback preserves cumulative waiting-interval growth. Upload and
  connection retries retain their existing ownership.

## Non-Goals

Changing upload deadlines, connection deadlines, unary compaction retry behavior, or model catalog
fields is outside this feature. Canceling a request does not undo work already done by the provider.

## Portability Constraints

- Retry growth MUST belong to the logical request's retry state, separate from transport-specific
  retry counters that may reset during fallback.
- Each candidate MUST use an independent connection and WebSocket continuation state. Only the
  selected candidate may supply the reusable connection and previous-response state.
- Selection MUST happen before events reach conversation history or client-side tool execution.
- The original transport MUST stay alive beyond the replacement trigger. Bound the entire
  selection window even when a provider sends only heartbeats or reasoning.
- Timeout arithmetic MUST saturate rather than wrap.

## Adapter Seams

- Provider default timeout resolution.
- Sampling and streaming remote-compaction retry loops.
- HTTP SSE setup and reused WebSocket request setup.

## P0 Acceptance

1. The original finishes after a replacement starts, and the replacement finishes first in a
   second scenario. Through the production model client, verify only the winning tool call appears
   in the continuation request and executes (REQ-1 through REQ-4).
2. A fast response starts no replacement; a failed replacement leaves the original eligible; turn
   cancellation closes both candidates. Exercise deterministic stream selection (REQ-2 to REQ-4).
3. Verify cumulative increases, retry-budget consumption, and transport-counter reset; inspect
   reused WebSocket ownership and remote-compaction wiring (REQ-1, REQ-2 and REQ-5).

## Integration Contract

This policy composes with empty-final-answer retry and compaction routing without changing their
retry ownership. Upload retries stay owned by the HTTP transport. Release packaging remains
integration-owned.
