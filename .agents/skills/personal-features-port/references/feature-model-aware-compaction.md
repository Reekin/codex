# Model-Aware Compaction

## Goal

Allow one model provider to serve models with different compaction capabilities. Models that
support the native Responses compaction protocol use remote compaction, while models that do not
support it use Codex local summarization without requiring a provider switch.

## Non-Goals

- Emulating the Responses compaction protocol in a downstream proxy.
- Inferring compaction support from model names or provider-specific aliases.
- Changing compaction prompts, token thresholds, or history reconstruction semantics.
- Changing which providers are eligible for remote compaction.

## Stable Contract

- **REQ-1**: Every model exposes whether it supports remote compaction. The capability defaults to
  enabled when omitted so existing model catalogs retain their behavior.
- **REQ-2**: Remote compaction is selected only when both the provider and the active model support
  it.
- **REQ-3**: A model that disables remote compaction uses the existing local summarization path for
  both manual and automatic compaction.
- **REQ-4**: Switching models changes the compaction path immediately according to the newly active
  model without requiring a provider or session switch.
- **REQ-5**: The remote compaction v2 feature flag selects the remote protocol version only after
  remote compaction is allowed by both provider and model.

## Portability Constraints

- The model capability MUST remain typed model metadata and MUST NOT be reconstructed from model
  names, display names, aliases, or provider IDs.
- The omitted capability MUST continue to mean enabled for backward compatibility with existing
  model catalogs.
- Manual and automatic compaction MUST use the same effective capability rule.
- Provider eligibility MUST remain an independent requirement; a model capability cannot enable
  remote compaction for a provider that does not support it.
- Local compaction MUST reuse the upstream summarization and history replacement implementation.

## Adapter Seams

- Model catalog deserialization and model metadata overrides.
- Provider and model capability resolution before compaction dispatch.
- Manual compaction task dispatch.
- Automatic pre-turn and mid-turn compaction dispatch.
- Model switching and resumed-session model metadata resolution.

## P0 Acceptance

1. Configure a remote-capable provider with a model whose capability is omitted. Run manual and
   automatic compaction and observe the configured remote protocol.
2. Configure the same provider with a model whose capability is disabled. Run manual and automatic
   compaction and observe an ordinary summarization request followed by locally rebuilt history.
3. Switch from a remote-enabled model to a remote-disabled model in one session. Trigger compaction
   and observe the local path without changing provider.
4. Configure a provider that does not support remote compaction with a model whose capability is
   enabled. Trigger compaction and observe the local path.

Evidence must include captured production request shapes through the core integration-test client,
plus focused model metadata serialization coverage.

## Integration Contract

- Model catalog overrides may disable remote compaction for downstream models such as Claude
  Fable while existing GPT entries omit the field and retain remote compaction.
- Integration smoke must exercise one remote-enabled GPT model and one remote-disabled downstream
  model through the same configured provider.
- Packaging must carry the updated model catalog schema and personal model override file where the
  downstream models are defined.

## Maintenance Rules

- Keep this file current-state only.
- Change the stable contract only with an intentional requirement decision.
- Record routine adapter movement, current commands, results, and residual risks in the temporary
  port brief.
