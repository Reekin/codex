# Chat Tree Migration Work Package

Copy or fill this template for every non-trivial chat tree port before making implementation edits.

Preferred handoff location:

- Keep the filled work package in the PR body, SmartTakeover prompt, or a temporary working note.
- If the migration needs multi-agent handoff across sessions, store it under `.agents/migration-notes/chat-tree/<upstream-version>/implementation-brief.md` and remove or archive it before final cleanup if it is not meant to ship.

Do not start code edits until sections 1 through 5 have enough detail to guide adapter work.

## 1. Context

- upstream stable tag/commit:
- previous upstream stable tag:
- previous chat-tree feature branch:
- new chat-tree migration branch:
- target integration branch:
- reference branches consulted:
- current default UI path:
- app-server present: yes/no
- schema generation path:

## 2. Reference Source Priority

Use this default priority unless the user gives a different one:

1. `feature-chat-tree-contract.md` and `feature-chat-tree-acceptance.md` define required behavior.
2. The direct previous chat-tree feature branch is the primary code carry-forward reference.
3. `feature-chat-tree-summary.md` defines completed-turn LLM summary behavior; use the latest branch that implemented it correctly as a code reference.
4. `feature-chat-tree-v116-lessons.md` is historical lessons and pitfalls, not a patch template.
5. Current upstream architecture decides adapter placement.

If a reference branch conflicts with the contract, follow the contract and record the deviation here:

| Conflict | Decision | Reason | Evidence |
| --- | --- | --- | --- |

## 3. Branch Strategy

- strategy: rebase / clean port with reused modules
- reason:
- new feature branch uses `ft/chat-tree-<upstream-version>` naming: yes/no
- old branch inventory completed: yes/no
- integration-only changes excluded from feature branch: yes/no
- release workflow left for integration branch: yes/no
- expected merge/replay path into integration:

## 4. Carry-Forward Inventory

| Area | Previous branch path | New target path | Action | Reason | Test evidence |
| --- | --- | --- | --- | --- | --- |
| domain | | | copy / adapt / replace / omit | | |
| reducer tests | | | copy / adapt / replace / omit | | |
| persistence/replay | | | copy / adapt / replace / omit | | |
| app-server protocol | | | copy / adapt / replace / omit | | |
| app-server fixtures/schema | | | copy / adapt / replace / omit | | |
| TUI overlay | | | copy / adapt / replace / omit | | |
| summary provider | | | copy / adapt / replace / omit | | |
| docs/skill references | | | copy / adapt / replace / omit | | |
| integration-only release workflow | | integration branch | omit from feature branch | belongs to integration | |

## 5. Upstream Hook Map

Every hook entry must identify the file/function and why the timing satisfies the contract.

### Turn Start

- file/function:
- called for normal user turns:
- excludes subagent/side-thread:
- runs before child-turn context mutation:
- captures parent from current node:
- test proving timing:

### Turn Finalize / Abort

- completed hook:
- interrupted hook:
- replaced hook:
- review-ended hook:
- deterministic fallback label path:
- durable flush point:
- test evidence:

### Model-Visible History

- file/function:
- selected branch source:
- sibling exclusion assertion:
- captured request-body test:

### Persistence And Replay

- durable append path:
- flush guarantee:
- replay source:
- current-node replay path:
- late-summary replay path:
- compaction/rollback path:
- test evidence:

### App-Server

- read method registration:
- set-current method registration:
- update notification routing:
- selected transcript read path:
- schema fixture paths:
- test evidence:

### UI

- slash command entrypoint:
- overlay data source:
- set-current command path:
- transcript refresh path:
- running-task behavior:
- stale notification handling:
- test/snapshot evidence:

## 6. Architecture Decisions

### Persistence Decision

- chosen source: rollout / sidecar / hybrid
- app-server unloaded read path:
- append/flush guarantee:
- replay source:
- schema impact:
- archive/export behavior:
- rollback behavior:
- source of truth for revision:
- source of truth for current node:
- source of truth for node facts:
- source of truth for summaries:

For hybrid persistence, every source-of-truth row must name exactly one authority.

### UI Path Classification

| UI path | Status | Read source | Set-current path | Transcript refresh path | Running-task behavior | Evidence |
| --- | --- | --- | --- | --- | --- | --- |
| native TUI | supported / redirected / unsupported | | | | block / safe switch / unsupported | |
| app-server TUI | supported / redirected / unsupported | | | | block / safe switch / unsupported | |

### Runtime Context Strategy

- selected strategy: full snapshots / ancestor rebuild / deltas / shared-prefix cache
- why:
- compaction behavior:
- rollback behavior:
- memory/performance risk:
- test evidence:

### Summary Provider

- deterministic fallback path:
- async LLM summary scheduling point:
- cancellation behavior:
- failure behavior:
- persistence event:
- tests:

## 7. Implementation Phase Checklist

| Phase | Required output | Status | Notes |
| --- | --- | --- | --- |
| Domain | reducer/projection/invariant tests | pending | |
| Core adapter | lifecycle, context projection, branch request body test | pending | |
| Persistence | append, flush, replay, current-node restore | pending | |
| App-server | `chatTree/read`, `chatTree/setCurrent`, `chatTree/updated`, selected transcript reads | pending | |
| UI | overlay, set-current, transcript refresh, running-task block | pending | |
| Summary | async LLM summary, fallback, cancellation/failure tests | pending | |
| Docs | updated feature references for reusable lessons only | pending | |

## 8. Current Validation Commands

| Layer | Command | Required before acceptance | Result | Notes |
| --- | --- | --- | --- | --- |
| domain | | yes | pending | |
| core golden context | | yes | pending | must capture request body |
| replay | | yes | pending | restart/resume/current/summary |
| app-server protocol | | if touched | pending | include schema fixtures |
| TUI snapshots | | if UI touched | pending | include insta review |
| summary | | yes | pending | async LLM request and failure cases |
| formatting/lint | `cd codex-rs && just fmt` | yes for Rust changes | pending | |

## 9. P0 Evidence Table

| P0 gate | Evidence type | Command / file | Result | Notes |
| --- | --- | --- | --- | --- |
| Normal user turns create durable nodes | test | | pending | |
| New nodes use current node at turn start as parent | test / captured event | | pending | |
| New nodes become current before next branchable turn | test | | pending | |
| Completed/interrupted/replaced/review-ended turns are represented | tests | | pending | |
| Next model request uses selected ancestor path only | captured request body | | pending | must exclude sibling turns |
| `thread/read` agrees with selected branch | app-server test | | pending | |
| Current-node changes survive restart/resume | replay test | | pending | |
| Replay restores nodes, parents, statuses, summaries, current node, selected branch | replay test | | pending | |
| Late summary updates do not mutate context/current/visible turn IDs | test | | pending | |
| Completed turns with assistant output spawn async LLM summary | test | | pending | |
| Summary failure/cancellation does not fail turn completion | test | | pending | |
| Subagent/side-thread turns are excluded | test | | pending | |
| App-server read/setCurrent/updated preserve wire contract | fixture/schema test | | pending | |
| Supported UI paths refresh transcript after set-current | snapshot / integration test | | pending | |
| Feature branch is ready for integration branch composition | diff/review | | pending | |

## 10. Residual Risks

Severity rubric:

- **P0**: breaks model-visible context correctness, durable replay, required summary behavior, app-server contract, or supported UI transcript consistency.
- **P1**: does not break core correctness but weakens future port stability, diagnostics, coverage, performance bounds, or integration safety.
- **P2**: naming, local cleanup, ergonomics, or extra non-blocking coverage.

| Severity | Risk | Impact | Owner/action |
| --- | --- | --- | --- |

## 11. SmartTakeover Handoff

```text
Use personal-features-port for acceptance.

Primary checklist:
- .agents/skills/personal-features-port/references/feature-chat-tree-acceptance.md

Supporting references:
- .agents/skills/personal-features-port/references/feature-chat-tree-contract.md
- .agents/skills/personal-features-port/references/feature-chat-tree-test-matrix.md
- .agents/skills/personal-features-port/references/feature-chat-tree-summary.md
- .agents/skills/personal-features-port/references/feature-chat-tree-app-server.md
- .agents/skills/personal-features-port/references/feature-chat-tree-migration-template.md

Filled work package:
- location or pasted content:

Validation results:
- command:
- result:
- notes:

Known residual risks:
- P0:
- P1:
- P2:

Review instructions:
- Check actual next-turn model request bodies, not only UI/API projections.
- Check completed-turn LLM summary behavior against feature-chat-tree-summary.md.
- Check every P0 gate has evidence in the work package.
```
