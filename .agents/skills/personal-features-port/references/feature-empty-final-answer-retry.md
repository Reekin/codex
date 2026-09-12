# Missing or Empty Final-Answer Retry

## Goal

Prevent a regular turn from completing silently when the model emits output without a visible final
answer, or when a final-answer assistant message contains only empty or whitespace text.

## Non-Goals

- Do not retry transport or stream failures.
- Do not classify commentary messages as final answers.
- Do not retry a completed sampling response that contains no output items.
- Do not change plan-mode completion semantics.
- Do not retry indefinitely.
- Do not add a public configuration surface unless product requirements change.

## Stable Contract

- **REQ-1 Classification**: Preserve three states for each completed sampling response: no final
  assistant message, only empty final assistant messages, and at least one non-empty final
  assistant message. A completed assistant message whose phase is final or absent under upstream
  semantics participates in this classification; commentary does not. Separately track whether the
  sampling response emitted any output item.
- **REQ-2 Budget**: When no model, tool, or pending-user follow-up remains, continue the same regular
  turn at most once if the final answer was empty, or if output was emitted without any final
  assistant message.
- **REQ-3 Auditability**: Record a short model-visible recovery prompt through the normal
  conversation-history and rollout path.
- **REQ-4 Completion**: A recovered response becomes the normal final assistant message and
  `TurnComplete.last_agent_message`.
- **REQ-5 Ordering**: Stop hooks, after-agent hooks, and normal turn completion observe the
  recovered result rather than the missing or empty result.
- **REQ-6 Exhaustion**: If the one retry also completes without a non-empty final answer, finish
  without a third request. A retry response with no output items does not request another recovery.

## Portability Constraints

- **MUST** preserve three distinct states: no final assistant message, empty final assistant
  message, and non-empty final assistant message.
- **MUST** derive the empty-final fact where assistant items and phases are aggregated.
- **MUST** keep the response-output fact local to one sampling response; it must not carry across
  tool follow-ups, compaction, or another sampling iteration.
- **MUST** apply the retry only after ordinary tool/model/user follow-up decisions are settled and
  before terminal hooks and completion.
- **MUST** use one per-turn retry budget that cannot reset during compaction or another sampling
  iteration.
- **MUST** represent the recovery prompt through the repository's normal bounded contextual-user
  fragment mechanism.
- **MUST** keep plan mode outside this regular-turn recovery policy.
- **PREFERRED** keep the implementation at the existing sampling loop seam instead of introducing a
  separate retry subsystem.

## Adapter Seams

Rediscover these semantics on every upstream base:

- assistant item phase and visible-text aggregation;
- sampling result and final-message state;
- regular turn loop follow-up decision;
- conversation-history recording;
- compaction and retry-loop state;
- stop/after-agent hooks and `TurnComplete`.

## P0 Acceptance

### Recovery

Cover both recovery inputs:

- Return an empty final answer, then a non-empty final answer.
- After a tool follow-up, return reasoning without an assistant message, then a non-empty final
  answer.

Prove the recovery request contains the recorded prompt and completion exposes the recovered
message.

### Retry Exhaustion

Use reasoning-only and empty final responses across the initial request and retry. Prove they share
one budget, no third recovery request occurs, and the turn terminates normally.

### Classification

Prove non-empty final output does not trigger the retry. Prove commentary does not count as a final
answer, a regular response containing only commentary is recoverable as missing a final answer, a
response with no output items does not trigger recovery, and plan mode does not use this recovery.
Prove an absent phase follows the same final-message semantics as the surrounding upstream turn
pipeline.

### Follow-Up And Ordering

Prove pending tool/model/user follow-up takes precedence. Prove terminal hooks and completion run
once with the recovered result.

## Integration Contract

- Existing transport retry, compaction, tool follow-up, stop-hook, and after-agent-hook behavior
  remains authoritative outside this one semantic recovery.
- Keep packaging and release automation out of the feature branch.
