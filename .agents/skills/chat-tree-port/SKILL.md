---
name: chat-tree-port
description: Use when porting, reimplementing, reviewing, or validating the downstream Codex chat tree feature across upstream versions or architecture changes. Covers branch strategy, feature contract, migration workflow, stable architecture, reference implementation notes, and required validation for preserving chat tree behavior while tracking upstream stable releases.
---

# Chat Tree Port

Use this skill to keep the downstream chat tree feature working while rebasing onto newer upstream Codex releases.

The goal is not to manually re-invent chat tree on every upstream release. The goal is to preserve the stable domain/API/UI contract while adapting only the upstream-specific hook layer.

When upstream changes are small enough, prefer a normal rebase and keep the reusable domain, wire contract, projection, overlay, and tests intact. When upstream architecture changes make the old patch impossible to apply cleanly, start from the new stable base and reintroduce the reusable pieces first, then rewrite only the adapters.

## Start Here

1. Read this file first.
2. Read `references/contract.md` before changing code.
3. Read `references/migration-playbook.md` before choosing hook points.
4. Read `references/implementation-guide.md` before implementing, triaging review feedback, resuming a long migration, or deciding whether another refactor is worth doing.
5. Read `references/test-matrix.md` before writing or accepting tests.
6. Read `references/app-server-compat.md` before changing app-server API, schema, notifications, or external-client behavior.
7. Read `references/release-automation.md` before pushing a completed migration branch.
8. Read `references/architecture.md` when deciding how much to refactor or when upstream architecture has changed.
9. Read `references/v116-review.md` when comparing against the v116 reference branch or investigating regressions.

## Branch Strategy

- Treat OpenAI's repo as `upstream`.
- Treat this repo/fork as the downstream maintenance line.
- Keep the skill on the durable chat tree maintenance branch, currently `codex/chat-tree`.
- For each upstream stable release, create a clean implementation branch from the stable tag, for example `rebase/v0.128-chat-tree`.
- Prefer rebasing the previous chat-tree branch when conflicts stay localized to adapter code and the reusable domain/API/test contract remains intact.
- Do not blindly hard-merge old chat tree branches into a new upstream release when conflicts cross session, rollout, app-server, and TUI architecture boundaries.
- Use older chat tree branches as behavior references and regression oracles, not as patch sources.
- Reuse copyable files or modules from the previous implementation for domain, projection, wire schema, fixtures, and tests.
- Rewrite upstream-specific adapters when their hook points have moved or their semantics changed.
- After a migration branch is stable, update the user-facing downstream branch to point at it or merge it according to the repository owner's release flow.

## Non-Negotiable Feature Rules

- Every normal user turn creates one chat tree node.
- The new node's parent is the current node at the moment the user turn starts.
- The new node becomes current before the next user turn can branch.
- Model-visible history for a turn is only the selected node's ancestor path plus the new turn.
- Switching current node changes future context but does not delete existing branches.
- Completed, interrupted, replaced, and review-ended turns are all represented as nodes.
- Current node changes, node metadata, and summaries must be recoverable after restart.
- Summary generation must never block turn completion or change model-visible history.
- UI state is never the source of truth.
- Runtime snapshots or caches are optimizations, not the persistent fact source.
- After the new stable app-server contract is adopted, future ports must preserve it without breaking external clients.

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

6. **Update this skill**
   - Add new architecture notes or new pitfalls to the reference docs when migration reveals a reusable lesson.

7. **Stop review loops deliberately**
   - Classify every review item as P0 contract break, P1 portability/test hardening, or P2 long-term debt.
   - Fix all P0 items before claiming the migration is complete.
   - Fix P1 items when they are small or when they protect future migrations.
   - Record P2 items in references or follow-up tasks instead of repeatedly reshaping working code.
   - Stop when the feature contract and golden tests pass, not when every possible architecture ideal has been exhausted.

## Preferred Technical Shape

- `ChatTreeEngine`: pure state reducer and invariant checker.
- `ChatTreeEvent`: durable event types such as node started/finalized, summary updated, current changed.
- `ChatTreeState`: nodes, current node, status, order, and metadata.
- `ChatTreeProjection`: visible ancestor path, visible turn IDs, full tree, overlay entries.
- `SummaryProvider`: deterministic fallback plus optional async model summary.
- Adapters around upstream-specific session/history/app-server/TUI structures.

Do not duplicate tree construction, path projection, or overlay flattening in multiple UI/API modules.

## Reference Branches

- `codex/chat-tree`: original v0.114-based concept branch.
- `rebase/v0.116-chat-tree`: fuller v0.116-based reference implementation with many bug fixes.
- Use `rebase/v0.116-chat-tree` as the most complete behavior reference.
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

## Stop Conditions

Stop and report before finalizing if:

- Current-node switching works live but does not survive restart.
- `thread/read` and model-visible context disagree about the selected branch.
- Native TUI and app-server TUI expose different tree/current behavior.
- A supported UI switches future model context but keeps showing the old branch transcript.
- Summary updates can alter a node's stored context.
- The implementation requires copying large old patches into unrelated new architecture.
- Validation depends on manual UI interaction without an equivalent CLI/RPC/test path.
- External app-server clients or scripts would need schema/method changes without an explicit compatibility plan.
- Review feedback contains untriaged P0/P1/P2 items, making it unclear whether more refactoring is required for correctness or only for long-term polish.
- A completed migration branch has no branch-push release workflow, because downstream users would not get a fresh executable after push.
