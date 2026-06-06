---
name: personal-features-port
description: Use when porting, reviewing, integrating, or validating downstream personal Codex features that are not expected to be accepted upstream. Covers per-feature migration branches, integration branches such as integration-0.133, adapter-first architecture, feature-specific references such as chat tree, and downstream packaging/release automation that should live on integration branches.
---

# Personal Features Port

Use this skill to keep multiple downstream personal features alive while tracking newer upstream Codex releases.

The maintenance model is:

1. Port each feature on its own feature branch.
2. Keep each feature branch focused on that feature's stable behavior and tests.
3. Merge or replay completed feature branches into one integration branch for the upstream version, for example `integration-0.133`.
4. Keep cross-cutting downstream infrastructure, packaging, and release workflow changes on the integration branch and carry them forward from integration branch to integration branch.

Do not treat the integration branch as the place to solve every feature's migration. A feature should first prove it can run against the new upstream base with a clear adapter boundary, then integration resolves feature-to-feature conflicts and downstream distribution.

## Start Here

Read only the references needed for the current phase:

- For the standard feature guide structure, read `references/feature-guide-template.md`.
- For any feature migration, read `references/common-migration-playbook.md`.
- For chat tree work, read `references/feature-chat-tree-contract.md` first, then the other `feature-chat-tree-*` files needed by the touched surface.
- For chat tree migration execution, fill `references/feature-chat-tree-migration-template.md` before implementation edits.
- For subagent identity work, read `references/feature-subagent-identity.md`.
- For integration branch work, read `references/integration-release-automation.md`.
- For final chat tree acceptance, read `references/feature-chat-tree-acceptance.md`.

## Branch Strategy

- Treat OpenAI's repo as `upstream`.
- Treat this repo/fork as the downstream personal-feature maintenance line.
- Use upstream stable tags as the base for each release train.
- For each feature, create a focused migration branch from the relevant upstream stable tag or from the previous release's feature branch, depending on which gives the cleanest isolated diff.
- Include the target upstream version in every feature branch name, for example `ft/<feature>-0.133`.
- After a feature branch passes its own acceptance checks, merge or replay it into the integration branch for that upstream version.
- Name integration branches by upstream version, for example `integration-0.133`.
- The integration branch carries:
  - the selected set of personal features for that upstream version;
  - cross-feature conflict resolution;
  - downstream build/package/release automation;
  - integration-only smoke tests and handoff notes.

Avoid putting feature-specific runtime behavior directly into the integration branch without a corresponding feature branch. The integration branch should compose proven feature branches, not hide feature work in release plumbing.

## Adapter-First Principle

Every personal feature should be organized around a stable feature core and a small upstream adapter.

Common rule for every feature branch:

- Keep product behavior, domain invariants, durable events, projection rules, and feature-owned tests as stable as possible across upstream versions.
- Put upstream-specific session, protocol, app-server, TUI, storage, or build-layout coupling behind adapter modules.
- On each upstream port, first ask whether the stable core can remain unchanged.
- If upstream changed underneath the feature, adjust the adapter-to-upstream implementation before redesigning the feature core.
- Do not let feature cores import large upstream orchestration types when a small adapter-owned data shape would preserve the boundary.
- Do not duplicate stable projection or reducer logic in UI/API/build glue.

The desired outcome is that future ports mostly answer: "where did upstream move the hook?" rather than "what did this feature mean?"

## Per-Feature Migration Workflow

1. Identify the upstream stable tag and the direct previous feature branch.
2. Inventory the feature's stable core, adapter surfaces, tests, and docs.
3. Try a normal release-to-release rebase when conflicts are localized to adapters.
4. If conflicts become architectural, start from the new upstream base and reinstall:
   - stable feature core;
   - feature contract/API/schema;
   - durable persistence/replay facts;
   - tests and fixtures;
   - thin adapters for the new upstream hook points.
5. Keep unrelated integration concerns out of the feature branch.
6. Run focused feature validation before merging into integration.
7. Update the feature-specific reference files with reusable lessons.

## Integration Workflow

Use an integration branch after the per-feature branches are individually coherent.

1. Create or update the versioned integration branch, for example `integration-0.133`.
2. Bring in each completed feature branch one at a time.
3. Resolve cross-feature conflicts in integration commits.
4. Carry forward downstream packaging and branch-release automation from the previous integration branch.
5. Keep release workflow changes in integration unless a feature branch truly cannot build or test without a small local adapter.
6. Run integration-level smoke tests that exercise the composed feature set.
7. Push the integration branch only after it can build and produce the expected downstream artifacts.

Packaging and branch release work is not part of any one feature's product contract. It should flow from integration branch to integration branch.

## Feature References

Current feature-specific references:

- `feature-guide-template.md`: common structure for every personal feature guide.
- `feature-exec-argv.md`: argv-native unified exec tool contract, adapter map, porting checklist, and acceptance matrix.
- `feature-subagent-identity.md`: compact guide for the subagent identity clarification feature.
- `feature-chat-tree-contract.md`: stable chat tree product/API behavior.
- `feature-chat-tree-architecture.md`: preferred stable core and adapter shape for chat tree.
- `feature-chat-tree-implementation.md`: chat tree implementation path and pitfalls.
- `feature-chat-tree-migration-template.md`: required work package template for non-trivial chat tree ports.
- `feature-chat-tree-summary.md`: required completed-turn LLM summary behavior.
- `feature-chat-tree-app-server.md`: app-server compatibility and schema guidance.
- `feature-chat-tree-test-matrix.md`: detailed chat tree regression scenarios.
- `feature-chat-tree-acceptance.md`: final chat tree acceptance and SmartTakeover handoff.
- `feature-chat-tree-v116-lessons.md`: older chat tree implementation lessons and pitfalls.

When adding another personal feature, add new `feature-<name>-*.md` references rather than expanding the skill body.

## Validation Commands

Adapt commands to the changed crates and current upstream layout.

- Format Rust changes with `just fmt` in `codex-rs`.
- Run crate-specific tests for changed crates, for example `cargo test -p codex-core`, `cargo test -p codex-app-server-protocol`, `cargo test -p codex-tui`, and app-server tests if touched.
- Regenerate app-server schema after protocol changes with `just write-app-server-schema`.
- If changing common/core/protocol behavior, ask before running the complete workspace test suite.
- For UI-visible changes, update and review insta snapshots.
- For app-server changes, validate wire-level fixtures and compatibility rules from the relevant feature reference.
- Before pushing an integration branch, verify the downstream release workflow using `references/integration-release-automation.md`.

## Acceptance

Feature branches are accepted by their feature-specific references. Integration branches are accepted when:

- every included feature has passed its own P0 acceptance gates;
- cross-feature conflicts are resolved in integration commits;
- integration-only release/build changes are present and current;
- the branch has a command-line validation path and does not depend only on manual UI checks;
- the downstream package/release workflow is ready to produce a fresh artifact after push.
