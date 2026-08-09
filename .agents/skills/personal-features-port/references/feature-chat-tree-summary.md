# Summary Implementation

Use this as the required completed-turn LLM summary pattern. It is derived from the `rebase/v0.128.0-chat-tree` implementation and should be adapted to the current upstream hook names rather than copied blindly.

## Required Behavior

- Completed normal turns with non-empty assistant output schedule a background LLM summary request.
- The summary request runs after the chat tree node has been finalized and the main turn completion has been emitted.
- Summary generation never blocks turn completion.
- Summary generation never changes model-visible history, node context snapshots, current node, or visible turn IDs.
- Aborted turns and turns without assistant text keep deterministic fallback labels.
- Subagent turns do not create main-thread summary jobs unless the product contract changes.
- Summary output is persisted as a late summary update event, for example `NodeSummaryUpdated`.
- Restart/replay restores late summary updates.

## Prompt Shape

Use a small no-tools request.

System instructions:

```text
You generate a concise summary label for one completed assistant turn.
```

User request template:

```text
Summarize this turn for a chat tree node.
Requirements:
- single line
- at most 96 characters
- no markdown
- no surrounding quotes
- describe user intent + assistant outcome

User message:
{user_message}

Assistant message:
{assistant_message}
```

Use the same model/session configuration as the completed turn where practical, but disable reasoning summaries for the summary request. Do not attach tools or parallel tool calls.

## Runtime Flow

1. Capture the user message for summary near user-turn start, before later context rewrites obscure the prompt that created the node.
2. On turn completion, keep the final assistant message only if it is non-empty after trimming.
3. Finalize the chat tree node as completed and persist/flush that durable fact.
4. Emit the normal turn completion event.
5. If the turn is a normal main-thread turn and assistant text exists, register a per-node summary job with a cancellation token.
6. Spawn the async summary job.
7. In the job, create a fresh model client session and send the no-tools summary prompt.
8. Collect assistant text from streamed deltas and/or final assistant message items.
9. Require a completed stream before accepting the result.
10. Normalize the result.
11. If the normalized summary is non-empty and the job was not cancelled, persist `NodeSummaryUpdated`.
12. Notify app-server/UI through the normal chat tree update path.
13. Remove the job from the active summary job map and wake shutdown waiters.

## Normalization

Normalize both model output and fallback labels:

- trim leading and trailing whitespace;
- choose the first non-empty line;
- remove one pair of surrounding double quotes;
- trim again;
- cap to 96 characters;
- append `...` when truncation occurs;
- reject empty normalized model output.

Fallback labels should be deterministic, bounded, and single-line. Use status-specific labels for aborted turns such as `turn interrupted`, `turn replaced`, and `turn review ended`.

## Cancellation And Failure

- Cancelling before the model request starts exits silently.
- Cancelling while streaming exits silently.
- Model request startup failure logs and exits.
- Stream error logs and exits.
- Missing assistant text logs and exits.
- Empty normalized model output logs and exits.
- Cancellation before persistence prevents the late update.
- Shutdown or session-loop exit must cancel outstanding summary jobs and wait until they are removed.

None of these cases may fail or rewind the completed turn.

## Required Tests

- Completed normal turn with non-empty assistant output makes a second model request dedicated to summary.
- The summary request uses no tools and includes the captured user and assistant messages.
- The normalized summary is persisted as a late summary update and visible through replay.
- Completed turn with empty assistant output does not call the model for summary.
- Subagent turn does not call the main-thread summary path.
- Summary model failure does not fail turn completion.
- Summary cancellation does not fail turn completion.
- Empty model output does not corrupt the node or branch projection.
- Late summary update changes only the node label and never changes the next-turn model request context.
