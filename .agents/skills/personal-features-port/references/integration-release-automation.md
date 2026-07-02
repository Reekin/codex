# Integration Release Automation

Use this page for downstream packaging and release behavior on versioned integration branches such as `integration-0.133`.

The workflow file owns automation details. Feature branches should usually not carry packaging changes unless they cannot compile or run focused tests without a small build-layout adapter.

## Ownership Rule

Downstream package/release changes belong to integration branches.

Carry these changes forward from one integration branch to the next:

- branch-push Windows release workflow;
- portable binary and archive asset naming;
- checksum generation;
- latest-release publishing policy;
- branch-specific build/staging adapters;
- integration-only release notes or handoff wording.

Do not make each feature branch rediscover or duplicate this release plumbing. A feature branch may mention that integration must include the workflow, but the workflow itself should keep flowing through integration.

## Agent Does Not Need To Manage

When the integration branch contains the downstream release workflow, these happen automatically after push:

- the push event starts the GitHub Actions build;
- a new pushed branch is treated the same as any other branch push;
- the Windows binary is built and staged;
- GitHub release assets are uploaded;
- checksum files are produced;
- the pushed build becomes the repository latest release.

Do not duplicate these mechanics in feature migration notes or feature branch commits. The agent only needs to preserve the integration workflow and check that it is still connected to the current upstream build layout.

## Agent Must Do

Before pushing a completed integration branch:

1. Confirm `.github/workflows/windows-codex-branch-release.yml` exists on the integration branch.
2. If the integration branch was created from a clean upstream tag and the workflow is missing, copy it from the previous integration branch.
3. Keep this workflow separate from upstream official release workflows.
4. Do not repurpose official release tags, signing, npm publishing, DotSlash, or WinGet automation for downstream branch builds.
5. If upstream changed the Rust crate layout, binary name, target path, or toolchain setup, update only the workflow's build/staging adapter section.
6. Keep the uploaded portable assets clearly named as standalone or portable artifacts.
7. Run a lightweight workflow sanity check before push: at minimum `git diff --check` and a visual check for YAML indentation or shell continuation mistakes.
8. Confirm every included feature branch has already passed its own focused P0 checks.

After pushing a completed integration branch:

1. Check that the branch-push release workflow started for the pushed commit.
2. If it fails before Rust compilation, fix the workflow adapter on the integration branch and push again.
3. If Rust compilation fails, treat it as an integration/build issue unless logs clearly show transient infrastructure failure.
4. Confirm the pushed commit produced the repository latest release before telling the user the packaged build is available.

## Decision Rule

The release workflow is integration infrastructure adapter code. Preserve its downstream behavior across upstream ports, and only change the parts forced by the new upstream build layout.

If the workflow is missing or broken, the integration branch is not ready to hand off even if every individual feature branch passed.
