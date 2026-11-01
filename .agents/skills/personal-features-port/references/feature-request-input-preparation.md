# Request Input Preparation

## Goal

Prepare images and bound outbound request bytes on the client using explicit per-model limits, reducing gateway work while preserving session affinity and stable prompt prefixes.

## Non-Goals

Do not move gateway account selection, authentication, provider protocol translation, cache-marker placement, or signing into the client. Exact byte equality with the gateway's previous image transformation is not required. Do not delete the gateway's final input validation or change unconfigured models.

## Stable Contract

- **REQ-1**: Model overrides can explicitly specify image dimension and outbound JSON byte limits. These are independent of token/context-window budgets and are not inferred from model-name substrings.
- **REQ-2**: Configured images are prepared deterministically and reused across retries and subsequent requests. An opaque prepared image is sent as JPEG when that is smaller; images with transparency keep their format. Preparing one model's outbound view does not overwrite the durable original conversation with that model's reduced images.
- **REQ-3**: Before transmission, remove historical image payloads oldest first when the configured byte budget is exceeded, preserving text, tool-call/result relationships, and relative ordering. Each crossing removes images until the request is back down to 70% of the budget, so the prefix changes once per several new images rather than on every one; the newest image is removed only when it alone does not fit. If even required non-image content exceeds the budget, return an explicit local error; do not silently remove text.
- **REQ-4**: Repeated preparation of the same logical input yields identical output. An image a session removed stays removed in its later requests, so appending turns or retrying never restores it and later requests keep the cached prefix. Preserve session identity and routing/cache identity inputs independently of projected image bytes. Necessary prefix changes at an image-budget boundary are allowed; arbitrary per-request re-encoding or selection churn is not.
- **REQ-5**: HTTP, WebSocket full/delta requests, model switching, and supported resumed sessions use the same preparation semantics. Local transcript and original assets remain available. Byte accounting covers the emitted request including tools/instructions, not just image data.

## Portability Constraints

MUST keep preparation rules in a feature-owned module with small upstream hooks. MUST derive all transports from the same prepared projection. MUST keep byte limits distinct from context token estimates. MUST retain gateway final checks because translation/injection can change the final provider request size.

## Adapter Seams

Model metadata override loading; request construction before serialization and WebSocket delta comparison; image encoding/cache; session identity propagation; durable history/replay.

## Configuration

The model catalog entry accepts `input_limits.max_image_dimension` (positive pixels) and
`input_limits.max_request_bytes` (positive uncompressed JSON bytes). Both are optional.
For a gateway with a 32,000,000-byte provider limit, a client budget of 30,000,000 bytes
reserves translation headroom; the gateway still validates its final provider request.

```json
"input_limits": {"max_image_dimension": 2000, "max_request_bytes": 30000000}
```

Inline message images, tool-result images and generated-image results participate. Remote
image URLs are not downloaded; their URL JSON contributes to the byte budget and the gateway
validates the referenced image. Budget removal replaces message/tool images with a short
omission note and clears generated image payloads while retaining their call record.
Remote compaction uses its own complete JSON envelope with the same preparation rules.

The configured limit is not a promise of zero cache misses: crossing the budget can remove
an older image and change that prefix. Stable session identity, unchanged retained prefixes,
and deterministic image encoding avoid unrelated churn.

## P0 Acceptance

1. Override configured and unconfigured models through the real model client: inspect captured HTTP bodies for dimensions and byte limits, unchanged unconfigured behavior, preserved text/tool pairing (REQ-1/2/3/5).
2. Multi-turn image history crosses its budget, then retries and appends another turn: compare retained images, stable encoded bytes, session headers and routing/cache keys. Resume and switch models without corrupting local history (REQ-2/4/5).
3. Exercise full and incremental WebSocket requests and HTTP fallback with identical logical inputs. Oversized text-only input errors locally; limits account for tools and instructions (REQ-1/3/5).
4. Route prepared requests through the gateway to a deterministic upstream: verify stable account/session selection and cache markers for unchanged retained prefixes. Verify valid provider size/dimensions; do not claim prompt cache hits solely from JSON equality (REQ-4/5).

## Integration Contract

Compose with model-aware compaction, context usage breakdown and large-request upload resilience. Preparation must precede upload retries and incremental request comparison. Packaging and remote validation workflow ownership remain with integration.

## Maintenance Rules

Keep this contract current-state only. Record concrete refs, paths, validation commands and measurements in the temporary port brief. Acceptance requires independent review and the feature's production request path, not helper tests alone.
