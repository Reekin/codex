# Release Handoff Rules

Use this page only for agent behavior at the end of a chat tree port. The workflow file owns the automation details.

## Agent Does Not Need To Manage

When the branch contains the downstream release workflow, these happen automatically after push:

- the push event starts the GitHub Actions build;
- a new pushed branch is treated the same as any other branch push;
- the Windows binary is built and staged;
- GitHub release assets are uploaded;
- checksum files are produced;
- the release is kept separate from official upstream releases.

Do not duplicate these mechanics in migration notes, manual commands, or the skill body. The agent only needs to preserve the workflow and check that it is still connected to the current build layout.

## Agent Must Do

Before pushing a completed migration branch:

1. Confirm `.github/workflows/windows-codex-branch-release.yml` exists on the branch.
2. If the branch was created from a clean upstream tag and the workflow is missing, copy it from the current chat-tree maintenance branch.
3. Keep this workflow separate from upstream official release workflows.
4. Do not repurpose official release tags, signing, npm publishing, DotSlash, WinGet, or latest-release automation for downstream branch builds.
5. If upstream changed the Rust crate layout, binary name, target path, or toolchain setup, update only the workflow's build/staging adapter section.
6. Run a lightweight workflow sanity check before push: at minimum `git diff --check` and a visual check for YAML indentation or shell continuation mistakes.

After pushing a completed migration branch:

1. Check that the branch-push release workflow started for the pushed commit.
2. If it fails before Rust compilation, fix the workflow adapter and push again.
3. If Rust compilation fails, treat it as a migration/build issue unless logs clearly show transient infrastructure failure.
4. Confirm a new prerelease exists for the pushed commit before telling the user the packaged build is available.

## Decision Rule

The release workflow is infrastructure adapter code. Preserve its downstream behavior across upstream ports, and only change the parts that are forced by the new upstream build layout.

If the workflow is missing or broken, the port is not ready to hand off even if chat tree tests pass, because users will not receive a fresh executable after push.
