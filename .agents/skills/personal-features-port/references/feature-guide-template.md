# Feature Guide Template

Use one file with this structure for each personal feature.

# `<Feature Name>`

## Goal

State the original user or agent problem in stable terms. Explain the observable outcome, not the
current implementation.

## Non-Goals

List nearby behavior this feature must not absorb. Use this section to prevent feature branches
from accumulating unrelated fixes.

## Stable Contract

Give each requirement a stable ID:

- **REQ-1**:
- **REQ-2**:

Cover only applicable surfaces:

- user-visible or model-visible behavior;
- durable state and replay;
- public API, tool, or schema behavior;
- supported and explicitly unsupported entrypoints;
- ordering or failure semantics that affect correctness;
- compatibility requirements.

Avoid current source paths, type names, branch names, and implementation history.

## Portability Constraints

State `MUST` constraints that prevent future migration drift.

Choose the profile in `SKILL.md`, record it only in the feature registry, and write constraints
appropriate to that profile.

Use `PREFERRED` only for a proven low-risk implementation shape that may be replaced when upstream
offers a better native mechanism. Do not describe an unimplemented preferred design as current
fact.

## Adapter Seams

Name semantic hooks to rediscover on each upstream base, for example:

- turn accepted before history mutation;
- completion or abort finalization;
- persistence append and replay;
- tool registration, approval, and resumed completion;
- API projection and notification;
- UI command and transcript refresh.

Do not freeze old file paths. The per-port hook map records current locations.

## P0 Acceptance

Define the smallest scenario corpus that proves every stable requirement. Prefer a few end-to-end
scenarios over repeated field-by-field checks.

For each scenario, state:

- setup and action;
- observable result;
- required evidence type;
- real supported path.

Every REQ must map to at least one P0 scenario. Tests, captured requests, wire fixtures, snapshots,
CLI/RPC smoke, or explicit unsupported-path behavior are valid evidence.

## Integration Contract

State:

- interactions with other personal features;
- smoke paths required after composition;
- build, packaging, or release work that belongs only on integration;
- any ordering constraint between feature merges.

## Maintenance Rules

- Keep this file current-state only.
- Change the stable contract only with an intentional requirement decision.
- Record routine adapter movement, current commands, results, and residual risks in the temporary
  port brief.
- Default to one feature file. Split only a large independently stable public contract, and link it
  directly from `SKILL.md`.
