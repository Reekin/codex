# Personal Feature Port 0.142.2

## Objective

Update the downstream personal Codex feature set from the previous 0.133 integration baseline to the latest stable upstream release, `rust-v0.142.2`, and re-port each feature branch independently before composing the new integration branch.

## Source Baselines

- Upstream stable baseline: `rust-v0.142.2`
- Previous downstream integration baseline: `integration-0.133`
- New feature branch pattern: `ft/<feature>-0.142.2`
- New integration branch: `integration-0.142.2`

## Feature Set

- Chat tree
- Argv-native unified exec tool
- Empty final-answer retry
- Subagent identity labels and forked-history identity hints

## Rationale

The personal feature set is intentionally downstream-only. Each feature is migrated on its own branch against the upstream stable tag so conflicts, test failures, and future rebases stay attributable to one feature at a time. Integration-only release automation and local workflow documentation belong on the integration branch rather than being mixed into clean feature branches.

## Approach

Each feature branch is created from `rust-v0.142.2` or rebased onto it, then receives only the commits needed for that feature. Conflicts are resolved by preserving upstream 0.142.2 architecture and moving downstream behavior to the current module boundaries. After the feature branches are clean, `integration-0.142.2` is created from `rust-v0.142.2` and merges the feature branches. Integration-only automation and local porting documentation are restored from the previous integration branch.

## Acceptance Criteria

- Every migrated feature has a `ft/*-0.142.2` branch based on `rust-v0.142.2`.
- The new `integration-0.142.2` branch merges the migrated feature branches.
- Downstream-only release automation and local migration documentation are present only on the integration branch.
- Rust formatting is applied after code changes.
- Changed crates receive targeted tests.
- App-server protocol schema artifacts are regenerated when protocol shape changes.
- Any validation that cannot be completed locally is explicitly reported.

## Stable References

- Porting workflow: `.agents/skills/personal-features-port/SKILL.md`
- Common migration playbook: `.agents/skills/personal-features-port/references/common-migration-playbook.md`
- Integration release automation notes: `.agents/skills/personal-features-port/references/integration-release-automation.md`
