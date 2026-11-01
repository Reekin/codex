# Large Request Upload Resilience

## Goal

Prevent a large model request from remaining indefinitely on a degraded HTTP connection when the
same destination is reachable at normal speed through a fresh connection.

## Non-Goals

- Choosing request compression for custom model providers.
- Changing provider retry counts or retrying completed model requests.
- Treating model generation latency or response streaming latency as upload failure.
- Adding operating-system-specific TCP diagnostics to the normal request path.

## Stable Contract

- **REQ-1**: Codex observes eligible HTTP request bodies and detects an abnormally long projected
  remaining upload time before the body has been fully submitted.
- **REQ-2**: A detected slow upload cancels the incomplete attempt so its HTTP/1.1 connection cannot
  return to the reusable pool, then retries the request on a fresh connection. A request allows at
  most three such retries (four attempts in total). Every attempt within that budget is monitored
  with the same stall rule, because a fresh connection may land on a slow network path again.
- **REQ-2a**: Once the retry budget is exhausted, the final attempt runs without slow-upload
  cancellation until it completes, so the request does not fail because of upload speed.
- **REQ-3**: Codex never triggers the upload retry after the complete request body has been handed
  to the transport, even while it is still waiting for response headers.
- **REQ-4**: Requests expected to complete promptly and requests that maintain acceptable upload
  progress retain the existing request and retry behavior.
- **REQ-5**: Diagnostics identify the total body size, submitted bytes, elapsed upload time, observed
  rate, and the stalled attempt number out of the maximum attempts for every fresh-connection
  retry, without logging request contents.

## Portability Constraints

- The policy MUST live at the HTTP request-body submission seam, not in turn orchestration or model
  response handling.
- The retry budget MUST be a request-scoped attempt counter shared by clones created for transport
  retries.
- Detection MUST use monotonic time and transport backpressure; wall-clock jumps must not affect it.
- Detection MUST combine observed throughput with projected remaining duration. A byte-size
  threshold alone is insufficient.
- The incomplete attempt MUST be dropped before the retry begins.
- The attempt after the retry budget is exhausted MUST be allowed to complete without slow-upload
  cancellation so genuinely slow user connections remain supported.
- The implementation MUST remain portable across Linux, macOS, and Windows.

## Adapter Seams

- Prepared encoded JSON request body construction.
- Reqwest body submission and response-header wait.
- Existing HTTP retry classification and attempt loop.
- HTTP transport diagnostics.

## P0 Acceptance

1. A large request whose first connection consumes body chunks below the permitted progress floor
   is canceled before completion. The next attempt uses a new connection and succeeds. Evidence:
   an HTTP transport integration test that observes two server connections and one completed body.
2. A large request whose connections keep stalling is canceled on every monitored attempt, uses a
   new connection each time, and its final attempt after three retries completes despite slow
   progress. Evidence: an HTTP transport integration test that stalls three connections and
   observes the fourth completing after a delay that would trip the stall rule.
3. A large request whose body advances normally completes on its first connection. Evidence: an
   HTTP transport integration test with exactly one accepted connection.
4. A request that has fully submitted its body but receives delayed response headers is not retried.
   Evidence: an HTTP transport integration test that delays headers after reading the complete body.
5. A small request is not subject to the slow-upload policy. Evidence: a focused transport test.

## Integration Contract

- No ordering dependency on other personal features.
- The integration smoke path must send a normal Responses request through the production HTTP
  transport.
- Packaging and release automation remain integration-owned.

## Maintenance Rules

- Keep thresholds and retry ownership in the HTTP client layer.
- Do not move upload policy into provider-specific executors unless a provider has a distinct wire
  contract.
- Changes to compression policy require a separate explicit contract decision.
