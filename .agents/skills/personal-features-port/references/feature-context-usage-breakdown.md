# Context Usage Breakdown

## Goal

Clients need one trustworthy number for how full the model context is, where auto-compaction will
fire, and what the context is made of. The number shown to users must be the same number that
drives automatic compaction, and every node of a thread history must be able to show the usage it
ended with, including threads reopened after a restart.

## Non-Goals

- Tokenizer-accurate per-category counts. Categories are estimates scaled to the provider-reported
  total.
- Changing model-visible context, compaction prompts, or compaction thresholds.
- Per-content-item or per-file attribution below the stable category list.
- Cost, billing, or rate-limit reporting.

## Stable Contract

- **REQ-1**: Active context usage is the provider-reported token total of the most recent model
  response plus a local estimate of history items appended after the last model-generated item.
  Auto-compaction, the reported context usage, and the category breakdown all use this same value.
  Reasoning from earlier turns is never re-estimated on top of the provider total.
- **REQ-2**: Every token usage snapshot that has provider usage carries a context usage record with
  the active context tokens, the active-context token count at which automatic compaction will
  trigger (the smaller of the configured auto-compaction trigger and the model's usable context
  window, or none when neither applies), and a category breakdown.
- **REQ-3**: The breakdown uses exactly these categories: base instructions, developer
  instructions, AGENTS.md, skills and plugins, tool definitions, environment and runtime rules,
  user messages, agent messages, tool calls (calls and outputs together), reasoning, compaction,
  and other. Categories are non-negative and sum exactly to the active context tokens.
- **REQ-4**: Classification is derived from harness-owned content classifications, response item
  types, and the request's base instructions and tool definitions. Unknown or future
  classifications fall into `other`; they never fail the snapshot.
- **REQ-4a**: Each model declares in its catalog entry whether reasoning from earlier turns stays in
  its input; omitted means it does. For models that drop it, the breakdown counts only reasoning
  recorded since the latest user turn, so dropped reasoning does not take share from other
  categories.
- **REQ-5**: The context usage record is persisted with the token usage snapshot, replayed when a
  client attaches to an existing thread, and included in live token usage notifications.
- **REQ-6**: A thread token usage read returns, for every turn of a thread that recorded provider
  usage, the last token usage snapshot recorded during that turn, keyed by turn id. It works for
  loaded and unloaded threads, and for forks the inherited history is attributed to the inherited
  turn ids.
- **REQ-7**: Snapshots persisted before this feature, which lack a context usage record, still
  load; the record is simply absent.

## Portability Constraints

- Active context usage MUST have one implementation shared by auto-compaction, notifications, and
  the breakdown; no surface may recompute it independently.
- Category mapping MUST live in one place and map harness content classifications by their stable
  feature prefix; it MUST NOT parse rendered text or tags.
- The breakdown MUST be scaled to the active context tokens with an exact-sum remainder rule.
- The context usage record MUST be carried by the same token usage snapshot type used for
  persistence, replay, and notifications so the three cannot diverge.
- The per-turn read MUST be derived from persisted token usage snapshots and the upstream history
  turn attribution; it MUST NOT introduce a separate store.

## Adapter Seams

- token usage recorded from a completed model response;
- token count emission and persistence;
- auto-compaction status computation;
- prompt construction (base instructions and tool definitions for the sampling request);
- session reconstruction of the latest token usage on resume;
- app-server token usage notification, attach replay, and history loading for loaded and unloaded
  threads.

## P0 Acceptance

1. **Unified accounting**: a thread whose history holds encrypted reasoning from an earlier turn
   starts a new turn. Evidence: automated test showing auto-compaction and the reported context
   tokens equal the provider total plus pending local items, with no reasoning surcharge.
2. **Breakdown on a real turn**: run a turn through the app-server against a mock Responses server
   with developer instructions, AGENTS.md, a tool call, and a final message. Evidence: the live
   `thread/tokenUsage/updated` notification has a context usage record whose categories sum to its
   tokens, with non-zero developer instructions, AGENTS.md, tools, user, tool calls, and agent
   messages, and a compaction trigger derived from the model configuration.
   Reasoning attribution for models that drop prior reasoning is covered by a breakdown unit test
   comparing the dropped estimate with a history that contains only the current turn's reasoning.
3. **Persistence and replay**: resume the thread from rollout in a new app-server. Evidence: the
   attach replay notification carries the same context usage record.
4. **Per-turn read**: after two turns, `thread/tokenUsage/read` returns one entry per turn with each
   turn's last snapshot; the same call works when the thread is unloaded. Evidence: app-server
   integration test.
5. **Legacy snapshot**: a rollout whose token counts lack context usage loads and is returned by
   the per-turn read with the record absent. Evidence: automated test.

## Integration Contract

- Model-aware compaction and local compaction handoff change how compaction runs, not how active
  context usage is counted; their compaction triggers must read the shared active usage.
- Integration smoke: run a real turn with the packaged binary and confirm the live notification
  carries a context usage record, then call `thread/tokenUsage/read` on the same thread.

## Maintenance Rules

- Keep this file current-state only.
- Change the stable contract only with an intentional requirement decision.
- When upstream adds content classifications, map them in the single category mapping; routine
  additions are adapter work, not contract changes.
