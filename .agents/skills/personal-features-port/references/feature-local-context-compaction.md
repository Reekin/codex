# Local Context Compaction

## Goal

Preserve high-quality dialogue verbatim for longer by selectively cleaning tool results before
compressing older dialogue. Keep original history locally queryable with stable references,
independent of hosted notes, encrypted compaction protocols, and model vendor.

## Non-Goals

- Capturing tool output discarded before it reached the conversation recorder.
- Guaranteeing lossless summaries or unlimited verbatim dialogue in a bounded window.
- Changing native remote compaction payloads.

## Stable Contract

- **REQ-1**: Local compaction first asks the conversation model to classify unprocessed, completed
  tool results as keep, shorten, or drop. The request includes current context and stable source
  IDs. Shorten includes replacement text preserving relevant evidence. Recent or unresolved tool
  work is protected. Unknown or duplicate IDs and oversized replacements cannot mutate history.
- **REQ-2**: Valid decisions are staged in memory against their source snapshot. Until cleanup is
  applied, active history is unchanged. New user direction invalidates staged decisions; newer
  appended items survive. Kept items can be reassessed when cleanup is insufficient.
- **REQ-3**: Automatic cleanup has configurable trigger and target occupancy, defaulting to 50%
  and 30%. It batches meaningful savings rather than rewriting every request. Budget calculations
  include fixed context. This soft trigger permits tool-only cleanup. If it cannot reach the target,
  keep the active view unchanged and suspend further soft attempts for that history generation and
  model, including after new user input. A rewritten window or model change permits soft attempts
  again. The ordinary Codex automatic-compaction limit and limit scope remain the hard trigger;
  manual compaction also permits full compression.
- **REQ-4**: Tool-only cleanup preserves dialogue verbatim and in order, and retains valid tool
  call/result pairing. Shortened and removed results carry stable original-content references.
  Decisions and analysis output are not ordinary dialogue.
- **REQ-5**: Only an ordinary hard compaction trigger or manual compaction may promote history into
  bounded chronological tiers when tool cleanup cannot provide sufficient room. Soft cleanup never
  invokes the tier summarizer. The tiers retain recent
  original history, dialogue with concise evidence, older condensed dialogue, and a bounded oldest
  range overview. Active constraints and decisions survive in a bounded ledger. Older ranges merge
  rather than accumulating one permanent entry per turn. Summaries preserve uncertainty,
  corrections, pending work, and verification status. Injected items have hard size limits below
  ten thousand tokens.
- **REQ-6**: Successful cleanup is installed and persisted as one consistent replacement view.
  Originals remain in the local archive. Failed analysis, invalid output, and cancellation cannot
  partially install changes or fall back to the removed handoff-summary policy.
- **REQ-7**: Local recall supports listing/searching history, reading turns at different detail
  levels, and reading original items in bounded pages. Model tools and CLI share data and retrieval
  semantics, expose pagination, and do not require a remote notes service.
- **REQ-8**: Provider/model capability routing remains the default. A local preference can force
  this pipeline for remote-capable models and bypass notes-driven window rollover. Native remote
  compaction remains independently available where supported.
- **REQ-9**: Resume preserves installed tiers, source IDs, and replacement content. Existing
  summarized conversations can participate without retaining the old summary generator. Custom
  compaction guidance can supplement the required structured-output protocol.

## Portability Constraints

- Budget selection, validation, and durable tier facts MUST be separate from request orchestration.
- Original archive records MUST remain recoverable after replacement and resume.
- Classifier/summarizer output MUST stay private until successful replacement installation.
- Manual, pre-turn, and mid-turn local paths MUST share the same pipeline.
- Call/result groups MUST be handled atomically when moving into history summaries.
- Old user-only retention and monolithic local handoff generation MUST be removed without fallback.
- Remote-only reconstruction helpers may remain where required by native remote behavior.

## Adapter Seams

- Provider/model routing and explicit local preference.
- Safe sampling boundaries after tool completion and before inference.
- Isolated model requests, cancellation, and usage accounting.
- History envelopes, checkpoints, and resume.
- Local rollout reading, tool registration, and CLI dispatch.

## P0 Acceptance

1. Return keep/shorten/drop JSON through the production model client; verify staged decisions stay
   private, dialogue survives cleanup, IDs retrieve originals, and subsequent requests contain the
   installed view. Cover manual and automatic entrypoints (REQ-1–4,7).
2. Repeatedly promote long dialogue/tool history; verify bounds, protected recent work, persistent
   constraints, chronological source ranges, and no orphan calls (REQ-5). Before the original hard
   limit, unsuccessful soft cleanup must leave the view intact, send no tier request, and suppress
   repeated soft requests despite further user input. At the original configured hard trigger,
   observe full compaction; also cover explicit manual compaction (REQ-3,5).
3. Return malformed/oversized/foreign-ID output and interrupt analysis; compare history before and
   after. Append newer items and invalidate on new user direction (REQ-1,2,6).
4. Resume after cleanup and read originals with model tools and CLI; verify replacement identity,
   pagination, and another cleanup of resumed history (REQ-6,7,9).
5. Use a remote-capable model with default routing and forced local preference, and a local-only
   model; capture request shapes and absence of hosted-notes dependencies (REQ-8,9).

## Integration Contract

Model-aware compaction selects the route. Context usage describes the installed view. CLI and
app-server expose recall for local compaction. Packaging carries workspace crates and prompts.
Capture composed local/remote smoke against the production client or packaged binary.

## Maintenance Rules

Keep this contract current-state only. Record concrete branches, commands, evidence, and residual
risks in the temporary port record. Intentional policy changes update this contract.
