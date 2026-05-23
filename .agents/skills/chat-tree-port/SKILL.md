---
name: chat-tree-port
description: Use when porting, reimplementing, reviewing, or validating the downstream Codex chat tree feature across upstream versions or architecture changes. Covers branch strategy, feature contract, required LLM summary generation, migration workflow, stable architecture, reference implementation notes, and validation for preserving chat tree behavior while tracking upstream stable releases.
---

# Chat Tree Port

Use this skill to keep the downstream chat tree feature working while rebasing onto newer upstream Codex releases.

The goal is not to manually re-invent chat tree on every upstream release. The goal is to preserve the stable domain/API/UI contract while adapting only the upstream-specific hook layer.

When upstream changes are small enough, prefer a normal rebase and keep the reusable domain, wire contract, projection, overlay, and tests intact. When upstream architecture changes make the old patch impossible to apply cleanly, start from the new stable base and reintroduce the reusable pieces first, then rewrite only the adapters.

## Start Here

Read only the references needed for the current phase:

1. For any migration, read `references/migration-playbook.md` and `references/contract.md`.
2. Before implementation, write the brief described in `references/implementation-guide.md`.
3. Before wiring summaries, read `references/summary-implementation.md`.
4. Before app-server API/schema work, read `references/app-server-compat.md`.
5. Before final review or SmartTakeover, use `references/acceptance-checklist.md` as the single acceptance source.
6. Before pushing a completed branch, read `references/release-automation.md`.

## Branch Strategy

- Treat OpenAI's repo as `upstream`.
- Treat this repo/fork as the downstream maintenance line.
- Keep the skill on the durable chat tree maintenance branch, currently `codex/chat-tree`.
- For each upstream stable release, create the new migration branch from the direct previous chat-tree release branch, then rebase that branch's downstream commits onto the new upstream stable tag.
- Prefer normal release-to-release rebase when conflicts stay localized to adapter code and the reusable domain/API/test contract remains intact.
- Do not blindly hard-merge old chat tree branches into a new upstream release when conflicts cross session, rollout, app-server, and TUI architecture boundaries.
- Use the direct previous release branch as the first behavior reference, then use older chat tree branches as regression oracles. Do not treat an older "complete reference" as a substitute for checking the immediately preceding port.
- If rebase becomes architectural and a clean port is necessary, first produce an inventory of every downstream commit/file category from the direct previous branch and account for each item in the clean port.
- Reuse copyable files or modules from the previous implementation for domain, projection, wire schema, fixtures, and tests.
- Rewrite upstream-specific adapters when their hook points have moved or their semantics changed.
- After a migration branch is stable, update the user-facing downstream branch to point at it or merge it according to the repository owner's release flow.

## Core Contract Snapshot

- Every normal user turn creates one chat tree node.
- The new node's parent is the current node at the moment the user turn starts.
- The new node becomes current before the next user turn can branch.
- Model-visible history for a turn is only the selected node's ancestor path plus the new turn.
- Switching current node changes future context but does not delete existing branches.
- Completed, interrupted, replaced, and review-ended turns are all represented as nodes.
- Current node changes, node metadata, and summaries must be recoverable after restart.
- Completed normal turns with non-empty assistant output must spawn an asynchronous LLM summary request and persist a bounded summary update. Deterministic labels are fallback behavior, not complete feature parity.
- Summary generation must never block turn completion or change model-visible history.

For the full acceptance standard, use `references/acceptance-checklist.md`.

## Implementation Workflow

1. **Identify upstream architecture**
   - Locate session state, turn lifecycle, history recording, rollout/replay, app-server protocol, and active TUI entrypoints.
   - Note whether native TUI, app-server TUI, or another UI is the default path.

2. **Install the domain layer first**
   - Prefer a pure chat tree module with state, events, projection, and overlay-entry flattening.
   - Keep this module independent from TUI, app-server, and model client APIs.

3. **Attach adapters**
   - Core adapter: creates/finalizes nodes and projects model-visible history.
   - Persistence adapter: records and replays chat tree events.
   - API adapter: exposes read/set-current operations and notifications.
   - UI adapter: opens `/chattree`, renders entries, and sends set-current requests.

4. **Validate behavior**
   - Run focused tests first.
   - Add or update tests from `references/test-matrix.md`.
   - Verify resume/replay and app-server paths, not just live native TUI behavior.

5. **Prepare release automation**
   - Ensure the downstream branch contains the branch-push Windows release workflow described in `references/release-automation.md`.
   - A completed migration branch should produce a GitHub prerelease automatically after it is pushed.

6. **Run continuous acceptance**
   - Start SmartTakeover for every release migration before claiming completion.
   - Give SmartTakeover the handoff context from `references/acceptance-checklist.md`.
   - Treat `references/acceptance-checklist.md` as the primary review checklist; use `test-matrix.md` only for detailed test ideas.

7. **Update this skill**
   - Add new architecture notes or new pitfalls to the reference docs when migration reveals a reusable lesson.

8. **Stop review loops deliberately**
   - Use `references/acceptance-checklist.md` for P0/P1/P2 definitions, stop conditions, and acceptance status.
   - Record non-blocking follow-up work in references or follow-up tasks instead of repeatedly reshaping working code.
   - Do not continue iterating on architecture ideals after the acceptance checklist passes.

## Preferred Technical Shape

- `ChatTreeEngine`: pure state reducer and invariant checker.
- `ChatTreeEvent`: durable event types such as node started/finalized, summary updated, current changed.
- `ChatTreeState`: nodes, current node, status, order, and metadata.
- `ChatTreeProjection`: visible ancestor path, visible turn IDs, full tree, overlay entries.
- `SummaryProvider`: deterministic fallback plus required async LLM summary for completed normal turns with assistant output.
- Adapters around upstream-specific session/history/app-server/TUI structures.

Do not duplicate tree construction, path projection, or overlay flattening in multiple UI/API modules.

## Reference Branches

- `codex/chat-tree`: original v0.114-based concept branch.
- `rebase/v0.116-chat-tree`: fuller v0.116-based reference implementation with many bug fixes.
- For each new port, inspect the direct previous release branch first, for example `rebase/v0.128.0-chat-tree` before porting to v0.133. Use `rebase/v0.116-chat-tree` as an older complete behavior reference only after checking the direct predecessor.
- Prefer behavior and tests from the reference branch over copying file-level diffs.

## Validation Commands

Adapt commands to the changed crates and current upstream layout.

- Format Rust changes with `just fmt` in `codex-rs`.
- Run crate-specific tests for changed crates, for example `cargo test -p codex-core`, `cargo test -p codex-app-server-protocol`, `cargo test -p codex-tui`, and app-server tests if touched.
- Regenerate app-server schema after protocol changes with `just write-app-server-schema`.
- If changing common/core/protocol behavior, ask before running the complete workspace test suite.
- For UI-visible changes, update and review insta snapshots.
- For app-server changes, validate the wire-level fixtures and compatibility rules in `references/app-server-compat.md`.
- Before pushing a completed migration branch, verify the branch contains the automatic dev release workflow from `references/release-automation.md`.

## Acceptance

Use `references/acceptance-checklist.md` for final acceptance and SmartTakeover. It owns the P0 gates, validation order, stop conditions, and required handoff context. Do not invent a second checklist from the other reference files.
