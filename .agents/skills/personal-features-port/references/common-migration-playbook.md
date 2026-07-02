# Common Migration Playbook

Use this playbook for any downstream personal feature that must keep following upstream Codex releases.

The default shape is per-feature first, integration second:

1. Port each personal feature on its own branch.
2. Validate that feature against its own contract.
3. Bring the validated feature branches into the versioned integration branch.
4. Keep downstream packaging and release automation on the integration branch.

Every feature should be documented using `feature-guide-template.md`. Large features should also provide a migration work package template so agents have a fixed execution artifact instead of inventing one during the port.

## 1. Prepare Branches

1. Fetch upstream and downstream refs.
2. Identify the target upstream stable tag.
3. Identify the direct previous branch for the feature being ported.
4. Identify the previous integration branch for downstream packaging/release automation.
5. Create a focused feature migration branch.
6. Create or update the versioned integration branch only after at least one feature branch is ready.

Default command shape for a feature branch:

```bash
git fetch upstream --tags
git fetch origin
git switch <previous-feature-branch>
git switch -c <new-feature-branch>
git rebase --onto <new-upstream-stable-tag> <previous-upstream-stable-tag> <new-feature-branch>
```

Default command shape for an integration branch:

```bash
git switch -c <integration-branch> <new-upstream-stable-tag>
git merge --no-ff <ported-feature-branch>
git merge --no-ff <next-ported-feature-branch>
```

Use cherry-pick or replay instead of merge when the repository owner wants a linear integration history.

## 2. Inventory the Feature

Before porting, classify the previous feature branch into:

- stable feature core;
- feature-owned contract/API/schema;
- durable persistence or replay facts;
- UI behavior and projections;
- tests and fixtures;
- upstream adapters;
- feature docs and skill references;
- integration-only build/package/release changes.

The last category belongs on integration branches unless the feature cannot be built or tested without a tiny local build adapter.

Record the inventory in the feature's migration work package. At minimum, account for each item as:

- copy unchanged;
- adapt to new upstream;
- replace with an equivalent;
- omit because it is no longer applicable;
- move to integration because it is build/package/release plumbing.

## 3. Choose Rebase or Clean Port

Prefer a normal rebase when:

- stable core files apply with localized conflicts;
- tests and fixtures are still meaningful;
- conflicts are mostly imports, module paths, registration sites, or small API call changes;
- the upstream lifecycle model is recognizably the same.

Prefer a clean port with reused modules when:

- turn/session/storage/UI architecture moved to different abstractions;
- resolving conflicts would mix old adapter assumptions into new upstream code;
- the old branch is too coupled to upstream internals to review safely.

Before clean-porting, account for every inventory item. Carry it forward, replace it with a documented equivalent, or explicitly classify it as no longer applicable.

## 4. Preserve the Stable Core

For every personal feature, keep the stable core separate from upstream adapters.

Stable core usually includes:

- product/domain invariants;
- pure reducer or projection logic;
- durable event vocabulary;
- schema or command contract;
- deterministic fallback behavior;
- feature-owned tests.

Adapter code usually includes:

- session and turn lifecycle hooks;
- history/context assembly;
- persistence plumbing;
- app-server registration;
- TUI rendering entrypoints;
- build-layout specifics.

Porting should usually rewrite adapters, not redefine the feature.

Use this source priority when references disagree:

1. the feature requirement and acceptance docs;
2. the direct previous feature branch;
3. feature-specific implementation references;
4. older branches as lessons or regression oracles;
5. current upstream code for adapter placement.

When a previous implementation conflicts with the feature contract, follow the contract and record the deviation.

## 5. Map the New Upstream Architecture

Find the current upstream equivalents for the feature's adapter points before editing.

Common adapter points:

- session state;
- turn lifecycle start/complete/abort;
- model-visible history assembly;
- history recording and compaction;
- durable persistence and replay;
- app-server RPC registry;
- app-server notification routing;
- native TUI or app-server UI entrypoints;
- workflow/build layout if the feature branch truly needs it.

Write down hook points before editing. This implementation brief becomes part of review handoff.

For each hook, record more than a file path:

- timing guarantee;
- excluded sources or modes;
- source of truth used;
- failure behavior;
- test or fixture proving the hook is correct.

## 6. Validate the Feature Branch

Run feature-focused tests first. Do not wait for the integration branch to discover that a feature branch does not satisfy its own contract.

For chat tree, use:

- `feature-chat-tree-contract.md`;
- `feature-chat-tree-test-matrix.md`;
- `feature-chat-tree-acceptance.md`.

For new features, add `feature-<name>-contract.md`, `feature-<name>-test-matrix.md`, and `feature-<name>-acceptance.md` when the feature is large enough to need them.

Every P0 acceptance gate should have evidence before the feature branch is called complete. Evidence should be one of:

- focused automated test;
- captured model/tool request body;
- schema or fixture diff;
- snapshot review;
- CLI/RPC smoke command;
- explicit unsupported-path decision with user-visible behavior.

## 7. Build the Integration Branch

After feature branches pass their own P0 checks:

1. Start from the upstream stable tag or an existing integration branch rebase.
2. Merge or replay completed feature branches one at a time.
3. Resolve cross-feature conflicts in integration commits.
4. Carry forward downstream release workflow and packaging changes from the previous integration branch.
5. Keep packaging fixes on integration unless they are required for a feature branch to compile or test.
6. Run integration smoke tests and release workflow sanity checks.

Integration should not reinterpret feature contracts. It composes feature branches, resolves cross-feature conflicts, and carries build/package/release automation forward.

## 8. Update References

Add reusable findings to:

- `common-migration-playbook.md` for shared process lessons;
- `integration-release-automation.md` for integration packaging/release lessons;
- feature-specific references for product, adapter, test, or acceptance details.

Do not add one-off status notes or "previous version changed X" prose. References should describe the current reusable process.

## Common Mistakes

- Porting directly on an integration branch and losing feature boundaries.
- Carrying release workflow edits on a feature branch when they belong to integration.
- Copying old file-level diffs before understanding new upstream hook points.
- Letting UI or API adapters own feature truth instead of consuming stable projections.
- Duplicating reducers/projections across feature, UI, and API modules.
- Treating a deterministic fallback as complete parity when the feature contract requires model output or background work.
- Testing only manual UI paths without a CLI/RPC/test equivalent.
- Forgetting to compare against the direct previous feature branch before using older references.
- Rewriting the stable core to match upstream internals instead of adapting upstream internals to the stable core.
