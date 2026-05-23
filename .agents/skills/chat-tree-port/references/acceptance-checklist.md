# Acceptance Checklist

Use this file as the single primary checklist for final chat tree migration review and SmartTakeover acceptance.

Supporting references:

- `contract.md`: behavior definitions when an item below is ambiguous.
- `test-matrix.md`: detailed test ideas and scenario variants.
- `summary-implementation.md`: required completed-turn LLM summary behavior.
- `release-automation.md`: branch-push release handoff.

## Required Evidence

Give the reviewer these facts before asking for acceptance:

- upstream stable tag/commit and the migration branch name;
- direct previous chat-tree release branch that was compared;
- older reference branches consulted, if any;
- implementation brief: supported UI paths, persistence source, app-server contract, summary implementation, rollback/compaction strategy, runtime context strategy;
- multi-agent/subagent identity regression status: whether the upstream base already exposes unambiguous current-agent identity to spawned subagents and guards ambiguous waits, or what downstream fix was carried forward;
- list of changed files grouped by domain/core/persistence/app-server/UI/tests/release workflow;
- focused validation commands and pass/fail results;
- known residual risks classified as P0, P1, or P2.

## P0 Completion Gates

All P0 gates must pass before the migration can be called complete.

- Normal user turns create durable chat tree nodes.
- New nodes use the current node at turn start as parent.
- New nodes become current before the next branchable user turn.
- Completed, interrupted, replaced, and review-ended turns are represented.
- Model-visible history for a new turn is only the selected ancestor path plus the new turn.
- `thread/read` and any selected transcript API agree with the real model-visible branch.
- Current-node changes persist and survive restart/resume.
- Replay restores nodes, parents, statuses, summaries, current node, and selected branch.
- Late summary updates never mutate node context, current node, or visible turn IDs.
- Completed normal turns with non-empty assistant output spawn a separate async LLM summary request and persist a bounded summary update.
- Summary request failure, cancellation, or empty output does not fail turn completion.
- Subagent or side-thread turns are excluded unless the product contract is explicitly changed.
- Spawned subagents keep unambiguous self identity after migration. A subagent that calls `list_agents` must be able to distinguish itself from `/root`, siblings, and children. A subagent that calls `wait_agent` must be told which agent mailbox is being observed, and legacy target-based waits must reject waiting for the current agent. If upstream already solves this, record the evidence; otherwise preserve or reimplement the identity hint, tool-output marker, and wait-identity reminder approach.
- App-server `chatTree/read`, `chatTree/setCurrent`, and `chatTree/updated` preserve the stable wire contract when app-server exists.
- App-server chat-tree errors expose stable `data.kind` values when supported by the current API layer.
- Every supported UI path can open the tree, set current, block unsafe switching while a task runs, and refresh the visible transcript to the selected branch.
- The completed migration branch contains the downstream branch-push Windows release workflow.

## Focused Validation Order

Run focused checks in this order where the current upstream layout supports them:

1. Domain reducer/projection tests.
2. Core golden context test: A -> B -> C, switch A, send D, and assert the next model request includes A/D but excludes B/C.
3. Replay tests: restart, current-node changes, late summaries, compaction/rollback behavior, missing parent/current diagnostics.
4. Completed-turn LLM summary tests: separate model request, persistence, replay, failure, cancellation, and no request when assistant output is empty.
5. Multi-agent identity static audit: inspect the subagent-visible context construction, inter-agent task rendering, `list_agents` schema/output code, and `wait_agent` result/error text to verify the active subagent can identify its own canonical path instead of assuming it is `/root`.
6. App-server read/set-current tests and wire fixtures.
7. `thread/read` and `thread/turns/list` selected-branch tests.
8. TUI overlay and transcript-refresh tests for each supported UI path.
9. Schema generation, snapshots, formatting, scoped lint/fix, and release workflow sanity checks as applicable.

## Stop Conditions

Stop and return to implementation if any item is true:

- Live current-node switching works but does not survive restart.
- Projection shown to UI/API differs from the real next-turn model context.
- Sibling branch turns can leak into a selected branch's next model request.
- A UI path switches future model context while still showing the old branch transcript.
- Completed-turn LLM summary generation is missing, disabled, or only documented as future work.
- Summary updates can alter stored context or branch projection.
- A spawned subagent can see sibling agents but cannot tell which `list_agents` entry is itself, receives only a raw inter-agent envelope without model-visible self identity, or can perform an ambiguous wait without being reminded that it is observing its own mailbox.
- Validation depends only on manual UI interaction without an equivalent CLI/RPC/test path.
- External app-server clients would need a breaking schema or method change without an explicit compatibility plan.
- Review feedback contains untriaged P0/P1/P2 items.
- The branch cannot produce the expected downstream release artifact after push.

## SmartTakeover Handoff

Every release migration must start SmartTakeover for continuous acceptance before claiming completion.

Provide this context to SmartTakeover:

```text
Use chat-tree-port for acceptance.

Primary checklist:
- .agents/skills/chat-tree-port/references/acceptance-checklist.md

Supporting references:
- .agents/skills/chat-tree-port/references/contract.md
- .agents/skills/chat-tree-port/references/test-matrix.md
- .agents/skills/chat-tree-port/references/summary-implementation.md
- .agents/skills/chat-tree-port/references/app-server-compat.md
- .agents/skills/chat-tree-port/references/release-automation.md

Migration context:
- upstream stable tag/commit:
- current migration branch:
- direct previous chat-tree release branch compared:
- older reference branches consulted:
- supported UI paths:
- persistence source:
- summary implementation hook and tests:
- app-server methods/fixtures touched:
- release workflow status:
- multi-agent/subagent identity regression status:

Changed files by area:
- domain:
- core:
- persistence/replay:
- app-server:
- UI:
- tests:
- workflow/docs:

Validation results:
- command:
- result:
- notes:

Known residual risks:
- P0:
- P1:
- P2:

Review instructions:
- Classify findings as P0/P1/P2.
- P0 means the migration cannot be accepted.
- P1 should be fixed when small or when it protects future migrations.
- P2 is follow-up debt and should not block acceptance.
- Check actual next-turn model request bodies for context correctness, not only UI/API projections.
- Check completed-turn LLM summary behavior against summary-implementation.md.
- Statically audit spawned subagent identity behavior when the port touches session context, multi-agent tools, rollout replay, or inter-agent communication. This bug is model-behavior-sensitive and may not reproduce reliably in unit tests. If upstream solved it, cite the upstream code path; otherwise require explicit subagent identity context, a `list_agents` current-agent marker, V2 `wait_agent` mailbox-identity reminder, and V1 current-agent wait rejection.
```
