# Auth-Independent Tool Exposure

## Goal

Expose installed tools to the model according to their capabilities and configuration when
the user connects with an API key or a ChatGPT login.

## Non-Goals

Changing service credentials, authentication, authorization, subscription entitlements, provider
endpoints, or implementing tools that are not installed. Backend-only services that require a
Codex service session, including history notes, retain their service eligibility checks.

## Stable Contract

- **REQ-1**: An installed image-generation tool is exposed with API-key authentication when
  its feature is enabled and the provider and model support it. Login method alone must not
  suppress its model-visible specification.
- **REQ-2**: Feature disablement, provider image/namespace capability limits, model image modality
  requirements, and free-plan entitlement restrictions continue to control exposure.
- **REQ-3**: Requests preserve the user's actual credentials and identity. Tool execution uses
  the existing service authentication and authorization path and preserves service failures.

## Portability Constraints

- MUST separate model-visible tool eligibility from credential selection and service execution.
- MUST use installed production tool contributors; do not synthesize a replacement tool.
- MUST retain capability and entitlement checks independent of the removed login-method check.

## Adapter Seams

- Extension installation and per-thread contribution.
- Per-turn tool eligibility and model request serialization.
- Provider capability projection and authenticated service execution.

## P0 Acceptance

- **API exposure (REQ-1, REQ-3)**: Start a supported thread with API-key authentication and the
  installed image-generation extension; submit a turn. Capture the production model request
  and verify the image-generation namespace and function are advertised and the API key is used.
- **Eligibility (REQ-2)**: Exercise the tool planner with the feature disabled, a text-only model,
  an unsupported provider, and a free-plan account. Each must omit image generation; a supported
  ChatGPT account must retain it. Require executable planner or model-request evidence.
- **Service boundary (REQ-3)**: Execute the installed tool through its normal RPC/model path
  against a deterministic service denial. Verify the failure remains observable. Credential
  selection and backend authorization code must remain unchanged.

## Integration Contract

No merge ordering with other personal features is required. Composed acceptance must verify
API-auth tool exposure through the shipped host's installed extensions. Packaging and release
builds belong to integration.

## Maintenance Rules

Keep this contract current-state only. Change requirements only through an intentional decision.
Keep current source locations, commands, evidence, and remaining validation in the port handoff.
