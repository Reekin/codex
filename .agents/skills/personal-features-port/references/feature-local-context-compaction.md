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
  classify completed, unmarked tool results, including results with images, as keep, shorten, or
  drop. It repeats the next ordinary
  request's instructions, tools and history unchanged and only appends the question, so the shared
  prefix is served from the prompt cache. The question names each candidate by its result ID and the
  model-visible call ID. A result's size includes its paired call when that call carries large
  arguments (1,000 bytes or more); such candidates also ask for a one-line call summary. Each batch
  takes the largest unmarked results first and adds results while their worst-case replacements
  (never larger than the original) fit half the response limit and the candidate list fits its
  request cap, so many small results share one batch. Shorten includes concise evidence. Because
  every batch resends the whole context, a batch starts only once unmarked tool volume reaches a
  configurable share of the usable window (5% by default); record counts never start one. Only
  one marking request is in flight per session; normal model work does not wait for it.
- **REQ-2**: Results are stored by stable record ID and original content. Appended messages, new user
  input, and disjoint tool cleanup do not invalidate a batch. Only unchanged, still-present records
  can receive a result. Invalid JSON, duplicate/foreign IDs, and oversized replacements cannot
  modify the live view. Failed marking is nonfatal and cannot cause a tight retry loop. Validated
  results are persisted immediately, before any cleanup; resume and fork restore those whose records
  are still present and unchanged, so they are neither lost nor sent for marking again. A fork keeps
  its own copy of inherited marks and never writes to its parent's.
- **REQ-3**: At safe request boundaries, apply completed marks when they release at least the
  configured share of the whole usable window (default 30 percentage points). Thus 50% to 30% is
  insufficient and 70% to 40% qualifies. No total-occupancy target or 50% trigger governs tool cleanup.
  Dialogue, call/result pairing, and unmarked output stay intact. Replacements expose original IDs.
  Shorten and drop remove a result's images and say how many were removed. A large paired call keeps
  its record type, name and call ID; only its arguments become the summary (a JSON object for
  function calls, plain text for free-form calls) with the original ID.
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
  Original images appear in item text as numbered placeholders. Reading an item attaches up to four
  original images as real images on its first page, with an image cursor for the rest; the CLI can
  include them as data URLs on request.
- **REQ-9**: Provider/model capability routing remains the default. A local preference can force
  this pipeline and bypass hosted-notes rollover. Configuration separately controls the marking
  volume threshold, reclaim percentage, and preferred full-compaction occupancy. Obsolete
  occupancy-trigger, minimum-savings and record-count options have no fallback path. Custom
  guidance supplements the structured protocol.
- **REQ-10**: App-server clients can read, for a loaded thread, whether marking runs on its current
  route, whether a batch is in flight, the estimated release of all current validated marks, and the
  automatic cleanup requirement. Clients can request immediate tool cleanup that applies every
  validated mark regardless of that requirement. It does not await in-flight marking, send model
  requests, or start full compaction, and it reports the estimated release (zero when nothing
  changes) together with the refreshed status.

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
- Durable mark storage and the app-server status/cleanup methods.

## P0 Acceptance

1. Hold a marking response open while ordinary model work completes. Verify current model settings,
   an unchanged ordinary-request prefix, largest-first candidates with visible call IDs, one
   in-flight batch, and unchanged live dialogue (REQ-1,2,5).
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
7. Validate marks below the savings requirement, restart and fork, and verify identical status
   without new marking requests. Apply manually through the app-server methods, including on the
   packaged binary; verify the release, the next request's view, and a no-op second apply (REQ-2,10).
8. Mark a screenshot result and a large free-form call. Verify the summary request flag, image
   removal with a count, the call keeping its name and ID with the summary as arguments, and recall
   returning the original image to the model (REQ-1,3,8).

## Integration Contract

Model-aware compaction supplies default routing; context usage describes only the active view.
CLI and app-server share local recall. Packaging carries feature crates/prompts and the executable
acceptance path. Validate composed local/remote and background/foreground behavior.

## Maintenance Rules

Keep this contract current-state only. Concrete branches, commands, evidence, and residual risks
belong in the temporary port record. Intentional policy changes update this contract.
