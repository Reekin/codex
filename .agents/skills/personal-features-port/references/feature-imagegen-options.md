# Image-Generation Options

## Goal

Let agents specify the image model, output size, and quality for each built-in image-generation
or editing call, including calls made through code mode.

## Non-Goals

Changing tool eligibility, authentication, provider routing, or adding persistent image defaults.
Backend model availability and supported size/quality combinations remain provider decisions.

## Stable Contract

- **REQ-1**: The installed imagegen tool accepts optional `model`, `size`, and `quality` fields.
  Model identifiers and size strings (`auto` or `WIDTHxHEIGHT`) are sent unchanged to the backend.
  Quality accepts `low`, `medium`, `high`, `xhigh`, `max`, and `auto`.
- **REQ-2**: Omitted or null options resolve to `gpt-image-2`, `auto` size, and `auto` quality.
  Explicit options apply equally to generation and editing, with local or conversation references.
- **REQ-3**: The advertised tool schema and description expose these options in direct and code-mode
  calls. Request serialization and response parsing support all advertised quality values.
- **REQ-4**: Preserve service errors for unsupported options. Do not retry with substituted models,
  sizes, or quality settings. Existing credentials, image selection, and output handling remain intact.

## Portability Constraints

- MUST derive tool quality choices and wire serialization from the same typed definition.
- MUST apply the options at the production image request construction seam for both operations.
- MUST preserve explicit model and size strings rather than maintain a local model catalog or
  impose model-specific resolution limits that can diverge from the provider.

## Adapter Seams

- Installed image tool argument schema and description.
- Generation and editing request construction.
- Image API request serialization and response quality deserialization.

## P0 Acceptance

- **Defaults (REQ-2)**: Exercise omitted and null options and compare complete generation/editing
  requests with the documented defaults.
- **Explicit options (REQ-1, REQ-2, REQ-3)**: Exercise production app-server/model tool calls against
  a deterministic image backend. Capture explicit model, size, and extended quality in generation
  and editing requests, and verify successful results with extended response quality values.
- **Schema and failures (REQ-3, REQ-4)**: Inspect the advertised schema for optional fields and all
  quality values; exercise an unsupported-option service response and preserve its failure.

## Integration Contract

Compose with auth-independent tool exposure by checking explicit image options through a custom
provider. No feature merge ordering is required. Build and release packaging belong to integration.

## Maintenance Rules

Keep behavior here and derive implementation adapters from the current upstream. Keep concrete
source locations, branch refs, verification results, and migration status in the task handoff.
