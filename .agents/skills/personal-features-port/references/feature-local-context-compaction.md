# Local Context Compaction

## Goal

Keep dialogue intact by preparing tool-result reductions in the background and applying them when
their savings justify a history rewrite. Bound older history at ordinary compaction boundaries and
keep original local records queryable without a hosted notes service.

## Non-Goals

- Recovering output discarded before the conversation recorder received it.
- Preserving unlimited verbatim history or fitting instructions larger than the model window.
- Changing native remote compaction payloads.

## Stable Contract

- **REQ-1**: A separate background request uses the current step's model and inference settings to
  classify completed, unmarked tool results as keep, shorten, or drop. Requests contain current
  conversation and explicit model-readable source IDs. Shorten includes concise evidence. Batches
  start after configurable accumulation: 32 records or 5% of the usable window by default. Only one
  marking request is in flight per session; normal model work does not wait for it.
- **REQ-2**: Results are stored by stable record ID and original content. Appended messages, new user
  input, and disjoint tool cleanup do not invalidate a batch. Only unchanged, still-present records
  can receive a result. Invalid JSON, duplicate/foreign IDs, and oversized replacements cannot
  modify the live view. Failed marking is nonfatal and cannot cause a tight retry loop.
- **REQ-3**: At safe request boundaries, apply completed marks when they release at least the
  configured share of the whole usable window (default 30 percentage points). Thus 50% to 30% is
  insufficient and 70% to 40% qualifies. No total-occupancy target or 50% trigger governs tool cleanup.
  Dialogue, call/result pairing, and unmarked output stay intact. Replacements expose original IDs.
- **REQ-4**: Before launching a batch, bound achievable savings by known pending savings, potentially
  removable unmarked results, and remaining growth before the ordinary hard limit. If even that
  optimistic bound is below the cleanup requirement, skip marking. Count currently protected output
  that can become eligible later; exclude known keeps and mandatory call/reference content. Budget
  or full-window changes require a fresh calculation.
- **REQ-5**: Full compaction runs only through original automatic-compaction triggers (including
  existing configured limit/scope and model transitions) or an explicit manual request. It cancels
  marking for the old full window and ignores late results. Tool-only cleanup does not cancel a
  disjoint marking batch. Background response usage is charged without replacing foreground
  occupancy, completion events, or current inference state.
- **REQ-6**: Full compaction assigns bounded budgets to recent verbatim dialogue, condensed older
  dialogue, an oldest-range overview, and active constraints/decisions. Older material is repeatedly
  eligible for promotion; no completed turn or previous window is permanently verbatim. An arbitrarily
  long active user turn is not an indivisible protected segment. Preserve canonical instructions and
  pending call groups. Prefer total occupancy of 30% after full compaction, but accept a larger valid
  result that safely permits continued work. Missing an ideal ratio is not itself an error.
  Old history length alone must not exhaust compression.
- **REQ-7**: Classifier and summarizer output stays private until validated installation. Persist
  originals and a consistent replacement checkpoint before updating the active view. Failed analysis
  cannot partially install changes. Real analysis/storage failure or irreducible active input can
  still be reported; no old handoff-summary fallback exists.
- **REQ-8**: Recall tools and CLI share bounded local listing/search, turn detail selection, and
  original-item pagination. Installed views and source references survive resume; inherited history
  respects fork boundaries. Output and every injected fragment have hard size limits below 10k tokens.
- **REQ-9**: Provider/model capability routing remains the default. A local preference can force
  this pipeline and bypass hosted-notes rollover. Configuration separately controls marking batch
  size, reclaim percentage, and preferred full-compaction occupancy. Obsolete occupancy-trigger and
  minimum-savings options have no fallback path. Custom guidance supplements the structured protocol.

## Portability Constraints

- Keep budgeting, per-record decisions, and tier planning separate from runtime orchestration.
- Preserve original archive records and source IDs through successful cleanup, resume, and fork.
- Use a full-window generation to reject cancelled background results, not every history rewrite.
- Apply tool changes at safe sampling boundaries; unfinished batches cannot block ordinary requests.
- Keep background billing separate from foreground context usage and response-completed events.
- Replace obsolete local rules; retain replay helpers only where archived checkpoints need them.

## Adapter Seams

- Accumulation checks and completed-batch consumption before ordinary inference.
- Session-owned background task lifetime, hard-window reset, and shutdown.
- Model transport and separate usage accounting.
- Original automatic/manual compaction dispatch and persisted history installation.
- Local archive tools, CLI, configuration, and resume.

## P0 Acceptance

1. Hold a marking response open while ordinary model work completes. Verify current model settings,
   visible IDs, one in-flight batch, and unchanged live dialogue (REQ-1,2,5).
2. Accumulate validated savings: 50 to 30 must not rewrite; 70 to 40 may rewrite. Newer unmarked
   results and concurrent batch targets survive; originals remain queryable (REQ-2,3,7,8).
3. Exercise feasible/impossible bounds, future eligibility of protected results, and a changed
   hard limit. Capture absence of needless marking requests (REQ-4,9).
4. Trigger the original hard limit and manual compaction. Verify cancellation of old marking,
   ignored late responses, bounded repeated tiers, preserved constraints, valid pairs, and success
   above preferred occupancy when still safe (REQ-5–7).
5. Return invalid data, fail/delay marking, append user input, and rewrite a window. Verify no partial
   rewrite, normal progress, independent usage, and no stale result install (REQ-1,2,5,7).
6. Resume/fork after cleanup and read originals with model tools and CLI. Compare forced-local and
   native-remote routes, and round-trip configuration (REQ-8,9).

## Integration Contract

Model-aware compaction supplies default routing; context usage describes only the active view.
CLI and app-server share local recall. Packaging carries feature crates/prompts and the executable
acceptance path. Validate composed local/remote and background/foreground behavior.

## Maintenance Rules

Keep this contract current-state only. Concrete branches, commands, evidence, and residual risks
belong in the temporary port record. Intentional policy changes update this contract.
