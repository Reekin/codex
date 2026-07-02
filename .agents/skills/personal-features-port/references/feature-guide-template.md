# Feature Guide Template

Use this structure for every personal feature maintained by `personal-features-port`.

Each feature reference should answer the same questions in the same order. Large features may split these sections across several `feature-<name>-*.md` files. Small features can keep them in one file.

This framework is distilled from the chat tree migration docs. Chat tree is the detailed example; subagent identity is the compact example.

## Feature Metadata

- **Feature name**:
- **Feature branch pattern**:
- **Target integration branch pattern**:
- **Primary owner / purpose**:
- **Current upstream base**:
- **Direct previous feature branch**:

## 1. Requirement Goal

State the product or agent-behavior goal in stable terms.

Required content:

- user-visible or agent-visible behavior;
- non-goals;
- compatibility requirements;
- data or state that must survive restart/rebase;
- paths that must be command-line, RPC, or test accessible.

Avoid implementation names unless they are part of the public contract.

Reusable pattern from chat tree:

- define the feature in terms of user-visible or agent-visible behavior;
- list state transitions that must be durable;
- list paths that must agree with the same source of truth;
- explicitly exclude side paths such as subagents, internal threads, or unsupported UI entrypoints when they are not part of the feature.

## 2. Technical Plan

Define the stable core and upstream adapter boundary.

Required content:

- stable domain/core concepts;
- durable facts or persisted state;
- public API/tool/schema/UI contract;
- adapter surfaces that may change with upstream;
- rejected designs and why;
- feature interactions that must be tested in integration.

The technical plan should make future ports mostly adjust adapters, not rediscover feature behavior.

Every feature plan should include this boundary:

| Layer | Owns | Must not own |
| --- | --- | --- |
| Stable core | domain rules, durable facts, projections, contract invariants | upstream orchestration objects, UI state, file IO, model clients |
| Runtime adapter | current upstream lifecycle hooks and runtime caches | durable feature semantics |
| Persistence adapter | append/flush/replay plumbing | duplicate source-of-truth decisions |
| API/tool adapter | wire/tool/schema conversion | independent reconstruction of feature truth |
| UI adapter | rendering and commands | authoritative feature state |

For small tool-only features, these layers may collapse into fewer modules, but the ownership rule still applies.

## 3. Implementation Details And References

Record how to implement the feature on a new upstream base.

Required content:

- copyable modules or tests from the direct previous feature branch;
- reference branch priority;
- expected file/module shape;
- upstream hook map requirements;
- current known crate or package boundaries;
- schema/snapshot generation requirements;
- examples of important tests or fixtures.

When a detail is version-specific, say how to rediscover it instead of hardcoding the old path as permanent truth.

Reference source priority should always be explicit:

1. feature requirement and acceptance docs define behavior;
2. direct previous feature branch is the first code carry-forward reference;
3. older branches are historical lessons or regression oracles, not patch templates;
4. current upstream architecture decides adapter placement;
5. when code references conflict with the contract, follow the contract and record the deviation.

Implementation references should distinguish:

- **copyable logic**: stable data model, durable event vocabulary, projection/reducer algorithms, fallback behavior, wire contract, fixtures, golden tests;
- **adapter-specific logic**: lifecycle hook placement, history/context assembly, persistence append/replay plumbing, app-server registration, tool schema generation, UI entrypoints, build-layout adapters.

This distinction is the main portability mechanism.

## 4. Operating Rules

State rules agents must follow while porting or integrating the feature.

Required content:

- branch hygiene;
- what belongs on feature branches vs integration branches;
- what must be done before editing;
- what must be done before claiming completion;
- commands or tools that are preferred;
- what not to preserve as fallback when a design is replaced.

Required pre-edit artifacts for non-trivial features:

- branch plan;
- carry-forward inventory;
- upstream hook map;
- persistence/source-of-truth decision;
- UI/tool/API path classification;
- validation command mapping.

If the feature has no filled work package or equivalent handoff artifact, do not start broad implementation edits.

## 5. Lessons And Pitfalls

Capture reusable lessons only.

Required content:

- common wrong assumptions;
- known upstream coupling traps;
- historical bugs that recur across versions;
- environment issues that can mimic feature failures;
- old implementation details that should not be copied.

Do not store one-off status reports here.

Good lessons are phrased as portable rules:

- "UI/API projections consume stable core output; they do not rebuild facts independently."
- "Runtime caches are rebuildable acceleration, not durable truth."
- "Hybrid persistence must name exactly one authority for each fact."
- "Manual UI validation never replaces a CLI/RPC/test path."

Avoid lessons that only say what changed in one release.

## 6. Acceptance Standard

Define how to prove the feature migrated correctly.

Required content:

- P0 gates that block acceptance;
- P1 risks that usually should be fixed;
- P2 follow-up examples;
- required evidence table;
- focused validation commands;
- UI/RPC/CLI smoke paths;
- integration smoke expectations.

Every P0 gate should map to a command, fixture, captured request, snapshot, or explicit unsupported-path decision.

Use this severity rubric:

- **P0**: breaks core behavior, model/tool-visible correctness, durable replay, required background work, public API/tool contract, or a supported UI path.
- **P1**: does not break core behavior but weakens future portability, diagnostics, test coverage, performance bounds, or integration safety.
- **P2**: naming, local cleanup, ergonomics, or extra non-blocking coverage.

The acceptance section should contain an evidence table:

| Gate | Evidence type | Command / file | Result | Notes |
| --- | --- | --- | --- | --- |
| stable behavior gate | test / fixture / captured request / snapshot / explicit unsupported decision | | pending | |

## 7. Migration Work Package

Every non-trivial feature should have a migration work package template that agents fill before or during a port.

The work package should include:

- context and branch plan;
- reference source priority;
- carry-forward inventory;
- upstream hook map;
- architecture decisions;
- implementation phase checklist;
- validation command mapping;
- P0 evidence table;
- residual risks;
- review or SmartTakeover handoff block.

Minimum work package shape:

| Section | Purpose |
| --- | --- |
| Context | upstream tag, previous branch, new feature branch, target integration branch |
| Reference source priority | prevents old branches from becoming accidental contracts |
| Carry-forward inventory | accounts for copied/adapted/replaced/omitted feature parts |
| Hook map | proves adapter placement and timing |
| Architecture decisions | records persistence, runtime cache, API/tool/UI support |
| Phase checklist | standardizes implementation order |
| Validation commands | maps examples to current crate/test names |
| P0 evidence table | makes acceptance review mechanical |
| Residual risks | separates blockers from follow-up |

## 8. Integration Handoff

Every feature guide should say what the integration branch must do with the feature.

Required content:

- expected merge or replay path;
- feature interactions that need integration smoke tests;
- release/build/package work that must remain integration-only;
- known conflicts with other personal features;
- minimal integration evidence before the feature is considered shipped.
