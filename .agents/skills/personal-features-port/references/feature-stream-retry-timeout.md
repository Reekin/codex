# Stream Retry Timeout

## Goal

Give slow model responses more time on each retry so an unchanged idle timeout does not repeatedly
discard a response before it can finish.

## Stable Contract

- **REQ-1**: The default response-stream idle timeout is ten minutes. An explicit provider timeout
  supplies the initial timeout instead.
- **REQ-2**: Each failed attempt in one logical Responses request adds five minutes to the next
  attempt's idle timeout. This applies to sampling and streaming remote compaction, over HTTP SSE
  and WebSockets, including transport fallback.
- **REQ-3**: A new logical request starts from the provider's initial timeout. Successful tool-call
  continuations and later user turns do not inherit an earlier request's increases.
- **REQ-4**: Existing retry budgets, retry delays, cancellation, and error reporting remain in force.
  The increment extends the idle wait, not the delay before retrying.

## Non-Goals

Changing upload deadlines, connection deadlines, unary compaction retry behavior, or model catalog
fields is outside this feature.

## Portability Constraints

- Retry growth MUST belong to the logical request's retry state, separate from transport-specific
  retry counters that may reset during fallback.
- Every attempt MUST pass its effective timeout to the transport, including reused WebSockets.
- Timeout arithmetic MUST saturate rather than wrap.

## Adapter Seams

- Provider default timeout resolution.
- Sampling and streaming remote-compaction retry loops.
- HTTP SSE setup and reused WebSocket request setup.

## P0 Acceptance

1. A stream fails and its retry completes after a pause longer than the configured initial timeout.
   Verify through the production model client with a controlled streaming server. This covers
   REQ-1 and REQ-2 with an explicit provider timeout.
2. After the recovered response, a new request again retries when the same initial timeout is
   exceeded. Verify the request count and successful completion through the model client (REQ-3).
3. Exercise cumulative increases and transport-counter reset independently; inspect both streaming
   transports and remote-compaction wiring. Retain existing retry-budget and cancellation tests
   (REQ-2 and REQ-4).

## Integration Contract

This policy composes with empty-final-answer retry and compaction routing without changing their
retry ownership. Upload retries stay owned by the HTTP transport. Release packaging remains
integration-owned.
