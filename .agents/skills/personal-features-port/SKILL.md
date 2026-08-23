---
name: personal-features-port
description: Port, reimplement, review, integrate, or validate downstream personal Codex features across upstream releases, and define new downstream features that must remain maintainable in future ports. Use for versioned feature branches, integration branches, migration planning, contract preservation, adapter remapping, acceptance, and downstream release handoff.
---

# Personal Features Port

Preserve downstream feature behavior while replacing only the wiring forced by upstream changes.

## Source Of Truth

Use this priority when sources disagree:

1. An explicit requirement change approved for the current task.
2. The feature contract in this skill.
3. The direct previous accepted feature branch or tag and its executable tests.
4. Current upstream architecture for hook and adapter placement.

Treat old code as evidence, not as the contract. Do not weaken a contract to match an incomplete port.

The latest immutable `personal/integration/<upstream-version>` acceptance tag is the canonical
launch ref. It must point to an accepted `integration-<upstream-version>` commit containing this
complete skill, every feature contract, and integration-owned release knowledge. For another
accepted revision on the same upstream version, create
`personal/integration/<upstream-version>-r<N>`; never move an existing acceptance tag.

If no personal acceptance tags exist yet, bootstrap them before starting another migration:

1. Verify the latest accepted integration commit.
2. Resolve and verify the direct accepted tip for every feature in the registry.
3. Tag every feature and the integration using the conventions below.

Old squashed integration history does not prove feature ancestry. During bootstrap, compare each
feature diff with the accepted integration behavior and evidence before tagging it. Prefer the
highest accepted upstream version and then the highest revision, not the most recent commit
timestamp.

## Feature Registry

Read only the target feature reference unless integration work needs more.

| Feature | Branch pattern | Portability profile | Reference |
| --- | --- | --- | --- |
| Chat tree | `ft/chat-tree-<upstream-version>` | stateful core | `references/feature-chat-tree.md` |
| Argv-native exec | `ft/exec-argv-<upstream-version>` | cross-cutting contract | `references/feature-exec-argv.md` |
| Empty final-answer retry | `ft/retry-empty-final-answer-<upstream-version>` | lifecycle policy | `references/feature-empty-final-answer-retry.md` |
| Subagent identity | `ft/subagent-identity-labels-<upstream-version>` | typed projection | `references/feature-subagent-identity.md` |
| Model-aware compaction | `ft/model-aware-compaction-<upstream-version>` | typed projection | `references/feature-model-aware-compaction.md` |
| Local compaction handoff | `ft/local-compaction-handoff-<upstream-version>` | lifecycle policy | `references/feature-local-compaction-handoff.md` |

For integration packaging and release work, also read
`references/integration-release-automation.md`.

## Start A Migration

Before editing code:

1. Inspect worktrees, local branches, remote branches, and tags. Confirm `HEAD` matches the latest
   accepted integration tag; do not trust the current directory name alone.
2. Read the repository `AGENTS.md`, this file, and the target feature reference.
3. Resolve the target upstream stable tag and the direct previous immutable feature acceptance tag.
4. Create or reuse `integration-<target-version>` from the target upstream tag.
5. Before creating feature branches, carry this complete skill and integration-owned release
   infrastructure from the launch integration into the target integration as a knowledge-baseline
   commit.
6. Create each `ft/<feature>-<target-version>` branch and worktree from that knowledge baseline.

This ordering ensures every migration agent starts with all contracts even when only one feature
branch is checked out.

Record a port brief in the task plan, PR body, or other temporary handoff. Do not add it to the
skill:

- launch integration and target upstream tag;
- launch integration acceptance tag;
- previous and new feature refs;
- target integration branch;
- selected feature set and whether the target integration is a staging branch or complete release;
- contract changes, normally none;
- carry-forward inventory: stable, adapt, replace, omit, or integration-only;
- integration-owned release file inventory from the previous integration;
- current upstream hook map, including timing, source of truth, failure behavior, and evidence;
- validation mapping for every P0 scenario;
- residual risks.

Do not begin broad implementation edits until the previous implementation and new hook points are
accounted for.

## Choose The Port Shape

Prefer a rebase or replay when conflicts are localized to adapter locations, imports, registration,
or small API changes.

Start a clean port from the target knowledge baseline when upstream changed the lifecycle, storage,
protocol, UI, or execution model enough that conflict resolution would preserve obsolete
assumptions.

In either mode:

- preserve the feature contract, stable data semantics, public wire behavior, and feature-owned
  tests;
- reuse stable implementation units when they still fit;
- rewrite adapters against current upstream instead of recreating old upstream architecture;
- do not use a port to perform speculative architecture cleanup; a `PREFERRED` shape never
  justifies a clean port or refactor by itself;
- keep packaging and cross-feature conflict fixes out of feature branches;
- remove replaced approaches rather than retaining the old path as fallback.

## Apply The Portability Profile

Use the profile from the feature registry. Do not force every feature into the same architecture.
Choose the profile for the feature's dominant migration risk. A small supporting durable field does
not by itself make a lifecycle policy or typed projection into a stateful core.

### Stateful Core

Keep domain state, durable facts, invariants, and projections independent of upstream orchestration.
Put lifecycle, persistence, API, and UI coupling behind adapters. Prefer no semantic diff in the
stable core; explain any change as a contract change or deliberate core improvement.

### Cross-Cutting Contract

Carry one typed source of truth through every subsystem that observes the feature. Avoid parallel
string/boolean fields and implicit sentinels that can diverge across hooks, approvals, sessions, or
outputs. Preserve a proven carrier when it remains coherent; introduce a replacement only when the
current upstream cannot express the contract safely.

### Lifecycle Policy

Keep the rule at the real upstream lifecycle seam and test ordering end to end. Do not create a
fake domain layer when the stable behavior is a small, timing-sensitive policy.

### Typed Projection

Derive every model-visible, API-visible, and UI-visible representation from one small set of stable
facts. Rendering and protocol adapters may change; identity or state facts must not be reconstructed
independently by each surface.

## Implement And Validate

Implement in dependency order:

1. stable data or behavior;
2. durable or model-visible contract;
3. upstream runtime adapters;
4. external API/tool/UI adapters;
5. focused tests and fixtures;
6. real supported user path.

For each feature, prove:

- contract delta is zero unless an intentional change was approved;
- stable units stayed unchanged or have a documented reason to change;
- central upstream modules contain only necessary hooks, not duplicated feature rules;
- every P0 scenario has automated or captured evidence;
- persistence, replay, model/tool requests, and public wire behavior are checked where applicable;
- supported UI paths are exercised as users operate them;
- a command-line, RPC, or test path exists for every critical behavior.

A real path exercises the production binary, RPC, model/tool client, or UI route. A deterministic
test server is acceptable when the production runtime reaches it end to end; direct calls to
implementation helpers are not sufficient by themselves.

Follow the current repository `AGENTS.md` for formatting, tests, schema generation, snapshots, and
full-suite approval. Do not copy version-sensitive commands into this skill.

After implementation and real-path acceptance, use a fresh independent reviewer. Give it the
feature contract, direct previous ref, migration diff, port brief, and evidence. Ask only for
requirement drift, adapter-boundary violations, and credible migration regressions. Do not start an
open-ended general hardening loop.

For a brand-new feature with no previous ref, use the feature branch base, current upstream behavior,
and the new contract as the review baseline.

## Integrate

Build the target integration branch from the knowledge baseline, then:

1. Merge each accepted feature branch separately and preserve its commit identity. Do not squash
   all features into one integration commit.
2. Resolve only cross-feature conflicts in integration commits.
3. Carry forward integration-owned packaging and release automation.
4. Run composed smoke paths for feature interactions.
5. Update feature references only when stable behavior or portability constraints changed.
6. Tag every accepted feature as `personal/<feature>/<upstream-version>`. Use `-r<N>` for another
   accepted revision on the same upstream version and never move an existing tag.
7. After every feature declared in the port brief is accepted and merged, create the immutable
   `personal/integration/<upstream-version>` acceptance tag, or the next `-r<N>` revision.

The target integration is complete only after every feature declared in the port brief is accepted
and merged. That completed integration becomes the canonical launch branch for the next migration.

## Add A New Personal Feature

Read `references/feature-guide-template.md`, then:

1. For a feature targeting the current accepted integration, create its feature branch from that
   integration. During an active release migration, branch from the target knowledge baseline.
2. Add the registry row and `references/feature-<name>.md` as the first feature-owned commit before
   broad implementation.
3. Choose the smallest correct portability profile.
4. Define stable behavior, non-goals, portability constraints, and P0 scenarios.
5. Create feature-owned tests and at least one real CLI, RPC, model/tool, or UI acceptance path.
6. State cross-feature interactions and integration-only work.

Default to one reference file per feature. Split out another file only for a large, independently
stable public contract that agents can load conditionally, and link it directly from this file.

## Maintain This Skill

- Put common workflow changes on the current integration branch.
- Put an intentional feature contract change on that feature branch so code and contract are
  reviewed together; merging it updates the next integration's canonical knowledge.
- Do not edit a feature contract for routine adapter movement.
- Do not store current versions, concrete branch refs, source paths, test filters, command results,
  `pending` tables, migration status, or historical changelogs in long-term references.
- Keep per-port inventories, hook maps, evidence, and risks in temporary work records.
- After accepting common knowledge or feature changes on an already released integration version,
  create a new integration acceptance-tag revision.
- Replace obsolete guidance instead of adding another overlapping reference.
