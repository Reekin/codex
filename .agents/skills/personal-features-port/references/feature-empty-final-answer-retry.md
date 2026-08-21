# Empty Final-Answer Retry

## Goal

Prevent a regular turn from completing silently when the model emits a final-answer assistant
message whose visible text is empty or whitespace.

## Non-Goals

- Do not retry transport or stream failures.
- Do not retry commentary messages.
- Do not treat the complete absence of an assistant message as an empty final answer.
- Do not retry indefinitely.
- Do not add a public configuration surface unless product requirements change.

## Stable Contract

- **REQ-1 Classification**: Classify a completed assistant message whose phase is final or absent
  under upstream semantics and whose visible text trims empty as an empty final answer. Exclude
  commentary.
- **REQ-2 Budget**: When no model, tool, or pending-user follow-up remains, continue the same regular
  turn at most once.
- **REQ-3 Auditability**: Record a short model-visible recovery prompt through the normal
  conversation-history and rollout path.
- **REQ-4 Completion**: A recovered response becomes the normal final assistant message and
  `TurnComplete.last_agent_message`.
- **REQ-5 Ordering**: Stop hooks, after-agent hooks, and normal turn completion observe the
  recovered result rather than the empty placeholder.
- **REQ-6 Exhaustion**: If the one retry also completes empty, finish without a third request.

## Portability Constraints

- **MUST** preserve three distinct states: no final assistant message, empty final assistant
  message, and non-empty final assistant message.
- **MUST** derive the empty-final fact where assistant items and phases are aggregated.
- **MUST** apply the retry only after ordinary tool/model/user follow-up decisions are settled and
  before terminal hooks and completion.
- **MUST** use one per-turn retry budget that cannot reset during compaction or another sampling
  iteration.
- **MUST** represent the recovery prompt through the repository's normal bounded contextual-user
  fragment mechanism.
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

Return an empty final answer, then a non-empty final answer. Prove exactly two model requests occur,
the second request contains the recorded recovery prompt, and completion exposes the recovered
message.

### Retry Exhaustion

Return empty final answers twice. Prove no third request occurs and the turn terminates normally.

### Classification

Prove non-empty final output, empty commentary, and a turn with no assistant message do not trigger
the retry. Prove an absent phase follows the same final-message semantics as the surrounding
upstream turn pipeline.

### Follow-Up And Ordering

Prove pending tool/model/user follow-up takes precedence. Prove terminal hooks and completion run
once with the recovered result.

## Integration Contract

- Chat-tree completion and summary capture must observe the recovered final message.
- Existing transport retry, compaction, tool follow-up, stop-hook, and after-agent-hook behavior
  remains authoritative outside this one semantic recovery.
- Keep packaging and release automation out of the feature branch.
