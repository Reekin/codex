# Acceptance Checklist

Use this file as the single primary checklist for final chat tree feature-branch review and SmartTakeover acceptance.

Supporting references:

- `feature-chat-tree-contract.md`: behavior definitions when an item below is ambiguous.
- `feature-chat-tree-test-matrix.md`: detailed test ideas and scenario variants.
- `feature-chat-tree-summary.md`: required completed-turn LLM summary behavior.
- `integration-release-automation.md`: integration branch release handoff after this feature is merged.

## Required Evidence

Give the reviewer these facts before asking for acceptance:

- upstream stable tag/commit and the migration branch name;
- direct previous chat-tree release branch that was compared;
- older reference branches consulted, if any;
- implementation brief: supported UI paths, persistence source, app-server contract, summary implementation, rollback/compaction strategy, runtime context strategy;
- list of changed files grouped by domain/core/persistence/app-server/UI/tests;
- integration target branch and release workflow status, if known;
- focused validation commands and pass/fail results;
- known residual risks classified as P0, P1, or P2.
- filled P0 evidence table from `feature-chat-tree-migration-template.md`.

## P0 Evidence Table

Before claiming acceptance, every P0 gate below must map to evidence in the migration work package.

Evidence can be:

- focused automated test;
- captured model request body;
- app-server schema or fixture;
- TUI snapshot or integration test;
- CLI/RPC smoke command;
- explicit unsupported-path decision with user-visible behavior.

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
- App-server `chatTree/read`, `chatTree/setCurrent`, and `chatTree/updated` preserve the stable wire contract when app-server exists.
- App-server chat-tree errors expose stable `data.kind` values when supported by the current API layer.
- Every supported UI path can open the tree, set current, block unsafe switching while a task runs, and refresh the visible transcript to the selected branch.
- The feature branch is ready to merge or replay into the versioned integration branch.

## Focused Validation Order

Run focused checks in this order where the current upstream layout supports them:

1. Domain reducer/projection tests.
2. Core golden context test: A -> B -> C, switch A, send D, and assert the next model request includes A/D but excludes B/C.
3. Replay tests: restart, current-node changes, late summaries, compaction/rollback behavior, missing parent/current diagnostics.
4. Completed-turn LLM summary tests: separate model request, persistence, replay, failure, cancellation, and no request when assistant output is empty.
5. App-server read/set-current tests and wire fixtures.
6. `thread/read` and `thread/turns/list` selected-branch tests.
7. TUI overlay and transcript-refresh tests for each supported UI path.
8. Schema generation, snapshots, formatting, and scoped lint/fix as applicable.

## Stop Conditions

Stop and return to implementation if any item is true:

- Live current-node switching works but does not survive restart.
- Projection shown to UI/API differs from the real next-turn model context.
- Sibling branch turns can leak into a selected branch's next model request.
- A UI path switches future model context while still showing the old branch transcript.
- Completed-turn LLM summary generation is missing, disabled, or only documented as future work.
- Summary updates can alter stored context or branch projection.
- Validation depends only on manual UI interaction without an equivalent CLI/RPC/test path.
- External app-server clients would need a breaking schema or method change without an explicit compatibility plan.
- Review feedback contains untriaged P0/P1/P2 items.
- The feature has no path to be composed into the versioned integration branch.

## SmartTakeover Handoff

Every chat tree feature migration should start SmartTakeover for continuous acceptance before claiming feature-branch completion.

Provide this context to SmartTakeover:

```text
Use personal-features-port for acceptance.

Primary checklist:
- .agents/skills/personal-features-port/references/feature-chat-tree-acceptance.md

Supporting references:
- .agents/skills/personal-features-port/references/feature-chat-tree-contract.md
- .agents/skills/personal-features-port/references/feature-chat-tree-test-matrix.md
- .agents/skills/personal-features-port/references/feature-chat-tree-summary.md
- .agents/skills/personal-features-port/references/feature-chat-tree-app-server.md
- .agents/skills/personal-features-port/references/integration-release-automation.md

Migration context:
- upstream stable tag/commit:
- current migration branch:
- direct previous chat-tree release branch compared:
- older reference branches consulted:
- supported UI paths:
- persistence source:
- summary implementation hook and tests:
- app-server methods/fixtures touched:
- target integration branch:
- integration release workflow status:

Changed files by area:
- domain:
- core:
- persistence/replay:
- app-server:
- UI:
- tests:
- docs:

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
- Check completed-turn LLM summary behavior against feature-chat-tree-summary.md.
```
