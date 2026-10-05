# Integration Release Automation

## Ownership

Keep downstream packaging and branch-release automation on versioned integration branches. Feature
branches may carry only the smallest build adapter required for their own focused validation.

Carry forward the previous accepted integration's:

- branch-push release workflow;
- required runtime binary set;
- portable archive and checksum behavior;
- build/staging adapters;
- downstream artifact naming and publishing policy.

Treat the workflow itself as the source of automation details. Do not duplicate them in feature
contracts.

## Migration

When creating the target integration knowledge baseline:

1. Identify the release-owned files in the direct previous integration branch.
2. Carry them forward separately from feature code.
3. Compare the target upstream's workspace layout, binary names, build commands, toolchain, and
   runtime dependencies.
4. Change only the build/staging adapter forced by upstream.
5. Keep downstream branch releases separate from upstream official signing, package-manager, tag,
   and distribution automation.

Before push, verify:

- every included feature passed static validation (see `SKILL.md`); real-path P0 acceptance and
  integration smoke paths run against the packaged binary after the remote build;
- all required binaries and runtime assets are staged;
- workflow syntax and shell continuations are valid;
- artifact names remain clearly downstream and portable;
- the integration diff contains no accidental feature work hidden in release plumbing.

## Push Scope

Push the integration acceptance tag in the same push as the integration commit it names; the
branch release reads it to title the build with its revision, so a tag pushed later leaves the
release titled with the bare version.

After push:

- If the task only asks to push or trigger the remote build, confirm the expected workflow started
  for the pushed commit and stop.
- Monitor CI, retry failures, or confirm final artifacts only when the user explicitly asks to wait,
  babysit, or verify release completion.
- Never claim a downloadable artifact is available without confirming the workflow completed for
  the exact integration commit.

Fix release-layout failures on the integration branch. Return feature-specific compile or behavior
failures to the owning feature branch when practical.
