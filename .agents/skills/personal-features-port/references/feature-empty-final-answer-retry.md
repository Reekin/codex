# Empty Final-Answer Retry

## Feature Metadata

- **Feature name**: empty final-answer retry.
- **Feature branch pattern**: `ft/retry-empty-final-answer-<upstream-version>`.
- **Target integration branch pattern**: `integration-<upstream-version>`.
- **Primary owner / purpose**: prevent a regular Codex turn from completing silently when the model emits a blank final answer.
- **Current upstream base**: `rust-v0.147.0`.
- **Current feature branch**: `ft/retry-empty-final-answer-0.147.0`.

## 1. Requirement Goal

When a regular turn receives a completed assistant message in the final-answer phase but the visible text is empty or whitespace, Codex should treat that as a recoverable semantic failure and ask the model once more for a non-empty final answer in the same turn.

The user-visible result is that a transient empty model final does not produce an empty final answer or `TurnComplete.last_agent_message = None` when a retry can recover it.

Non-goals:

- Do not retry forever.
- Do not retry normal stream or transport errors; existing retry/backoff owns those failures.
- Do not retry commentary-phase messages that are intentionally interim text.
- Do not add a new public configuration surface unless the behavior later proves unsafe.

Compatibility requirements:

- Preserve rollout/history visibility for the recovery prompt so request traces explain the extra model call.
- Preserve normal tool follow-up, pending user input, stop-hook, and after-agent-hook behavior.
- Keep turns that truly have no assistant message distinct from turns that have an empty final-answer assistant message.

## 2. Technical Plan

Stable behavior:

- A final assistant message whose combined visible text trims to empty is an empty final-answer event.
- A commentary assistant message with empty text is not an empty final answer.
- A sampling request retries only if it ended without a non-empty `last_agent_message`.
- Retry budget is one internal continuation per regular turn.

Adapter placement:

| Layer | Owns | Must not own |
| --- | --- | --- |
| Stable core | identifying empty final-answer facts and the one-retry rule | UI rendering, transport retry policy |
| Runtime adapter | current `run_turn` sampling loop and session-history continuation prompt | persistent feature-specific state |
| Persistence adapter | normal conversation item recording and rollout capture | a separate retry log format |
| API/tool adapter | no new public API | protocol/schema changes |
| UI adapter | normal final recovered assistant message display | special empty-final UI banners |

Rejected designs:

- Retrying whenever `last_agent_message` is `None` is too broad because some tests and flows complete without any assistant message.
- Treating the empty final as a transport error is inaccurate and would mix model semantic recovery with HTTP/SSE retry policy.
- Silently rerunning without recording the recovery prompt makes request history harder to audit.

## 3. Implementation Details And References

Expected code shape on current upstream:

- `codex-rs/core/src/stream_events_utils.rs`
  - Add an `empty_final_answer` fact beside `last_agent_message`.
  - Set it only for assistant `TurnItem::AgentMessage` values where `phase` is not `MessagePhase::Commentary` and combined text trims empty.
  - Pass the fact through `OutputItemResult`.
- `codex-rs/core/src/session/turn.rs`
  - Pass `empty_final_answer` through `SamplingRequestResult`.
  - In the regular sampling loop, before stop hooks and after confirming no model/tool/user follow-up is pending, record a short internal user message and continue if the retry budget is available.
  - Use a fixed one-retry budget.
- `codex-rs/core/tests/suite/client.rs`
  - Cover a first response with `phase = "final_answer"` and empty text.
  - Cover the second response returning a normal assistant message.
  - Assert `TurnComplete.last_agent_message` is the recovered message and the captured second request contains the recovery prompt.

Reference source priority:

1. This document defines the downstream behavior.
2. The current feature branch implementation is the first code reference for future ports.
3. Upstream retry/backoff code remains transport-owned and should not be copied into this feature.
4. Current upstream session/task architecture decides the exact hook names.

## 4. Operating Rules

- Keep this feature on a focused feature branch before merging to the versioned integration branch.
- Before editing a future port, inspect how the current upstream records assistant message facts and how `TurnComplete.last_agent_message` is produced.
- Do not add a retry based only on `last_agent_message = None`; preserve the explicit empty-final fact.
- Do not hide the continuation prompt from history unless upstream gains a first-class internal continuation item.
- Do not move this into TUI/app-server code; the behavior belongs in core turn execution.

## 5. Lessons And Pitfalls

- Semantic-empty responses are different from stream failures. Retry policy should preserve that distinction.
- Absence of an assistant message is not the same as an empty assistant final answer. Porting must keep those states separate.
- A recovery request must be visible in captured requests or rollout data, otherwise the extra model call becomes difficult to diagnose.
- Stop hooks and after-agent hooks should see the recovered final answer, not the empty placeholder.

## 6. Acceptance Standard

P0 gates:

- Empty final-answer assistant message triggers exactly one internal retry when no other follow-up is pending.
- A recovered second response becomes `TurnComplete.last_agent_message`.
- The retry request contains an auditable internal prompt explaining the empty final answer.
- Normal non-empty assistant messages do not retry.
- Turns with no assistant message are not automatically reclassified as empty final answers.

Evidence table:

| Gate | Evidence type | Command / file | Result | Notes |
| --- | --- | --- | --- | --- |
| empty final retry | test | `cargo test -p codex-core empty_final_answer_retries_once_before_turn_complete` | pending | verifies recovered completion and captured retry request |
| formatting | command | `just fmt` in `codex-rs` | pending | required after Rust edits |
| crate regression | command | `cargo test -p codex-core` | pending | run for core behavior changes |

## 7. Migration Work Package

For future upstream ports, fill:

| Section | Notes |
| --- | --- |
| Context | upstream version, feature branch, target integration branch |
| Hook map | current assistant item finalization path, sampling-result type, regular turn loop |
| Carry-forward inventory | empty-final fact, one-retry guard, recovery prompt, client test |
| Validation commands | focused test, `cargo test -p codex-core`, `just fmt` |
| Residual risks | any upstream changes to assistant message phases or hidden/internal prompt support |

## 8. Integration Handoff

Merge the focused feature branch into the versioned integration branch after the P0 gates pass. Integration should only verify this feature's interaction with other downstream turn-loop features; packaging and release automation remain integration-only work.
