# Local Compaction Handoff

## Goal

Make Codex local compaction produce a high-fidelity handoff that preserves the distinctions,
evidence, and active work needed to continue a long engineering task without silently turning
tentative ideas into accepted decisions. Keep the full rollout discoverable when exact
pre-compaction details are needed.

## Non-Goals

- Changing remote compaction protocols or their output.
- Replacing model-generated summaries with deterministic transcript compression.
- Preserving arbitrary pre-compaction assistant messages or tool results in active context.
- Guaranteeing that every provider returns a particular summary length.
- Adding a second model pass that reviews or repairs the generated summary.

## Stable Contract

- **REQ-1**: The default local compaction prompt must tell the model that its handoff becomes the
  primary surviving record of prior assistant work and tool evidence, so completeness and factual
  accuracy take priority over brevity.
- **REQ-2**: The default handoff must explicitly separate confirmed decisions, tentative
  proposals, rejected options, unresolved questions, completed work, verification evidence,
  repository state, and pending next actions.
- **REQ-3**: The handoff must preserve consequential user corrections and exact critical details
  such as paths, symbols, commands, identifiers, errors, measurements, and validation status. It
  must not promote an unverified claim, estimate, or proposal into an accepted fact.
- **REQ-4**: When the session has a local rollout, the installed local compaction summary must
  identify that rollout as the recovery source for exact code, tool output, errors, or decisions
  omitted from active context.
- **REQ-5**: A configured custom compaction prompt remains a complete replacement for the default
  prompt. The rollout recovery reference is independent of prompt selection and is still installed
  with a successful local compaction summary.
- **REQ-6**: Manual, pre-turn automatic, and mid-turn automatic local compaction must share the
  same handoff and recovery-reference behavior.
- **REQ-7**: Remote compaction requests and remote replacement-history semantics must remain
  unchanged.

## Portability Constraints

- The policy MUST remain at the local compaction lifecycle seam used by both manual and automatic
  entrypoints.
- The default prompt MUST remain owned by the prompts crate rather than duplicated at dispatch
  call sites.
- The recovery reference MUST use the session's canonical rollout path and MUST degrade cleanly
  when the session has no local rollout.
- Custom prompt precedence MUST remain unchanged.
- Remote compaction MUST NOT inherit local handoff text or local rollout recovery instructions.
- Persisted replacement history and the live post-compaction history MUST remain identical.

## Adapter Seams

- Default local summarization prompt resolution.
- Local compaction model request construction.
- Successful local summary installation into replacement history.
- Session rollout-path lookup and rollout persistence.
- Manual and automatic compaction dispatch.
- Resume reconstruction from persisted compacted replacement history.

## P0 Acceptance

1. Run manual local compaction through the core integration-test model client. Observe the full
   default handoff prompt in the ordinary model request and no remote compaction trigger.
2. Complete local compaction in a persisted session and submit a follow-up turn. Observe one
   installed compact summary containing the canonical rollout recovery reference, and verify the
   persisted replacement history matches the follow-up request.
3. Configure a custom compaction prompt and run local compaction. Observe the custom prompt instead
   of the default while the installed summary still carries the rollout recovery reference.
4. Run remote compaction through a remote-capable model. Observe the native remote request and
   confirm no local handoff prompt or rollout recovery text is injected.

Evidence must use captured production request shapes through the core integration-test client and
persisted rollout reconstruction where applicable.

## Integration Contract

- Model-aware compaction decides whether this feature is active: only models routed to local
  compaction receive the high-fidelity handoff behavior.
- Integration smoke must cover one remote-capable model and one local-only model through the same
  provider.
- No packaging or schema work is required beyond carrying the prompt template and feature contract.

## Maintenance Rules

- Keep this file current-state only.
- Change the stable contract only with an intentional requirement decision.
- Record prompt wording refinements here only when they change a stable behavioral requirement.
- Record routine adapter movement, current commands, results, and residual risks in the temporary
  port brief.
