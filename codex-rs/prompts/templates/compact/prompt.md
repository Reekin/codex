You are performing a CONTEXT CHECKPOINT COMPACTION.

Your output will become the primary surviving record of prior assistant messages, tool results,
technical investigation, decisions, and verification. Information omitted from the handoff may be
unavailable to the next model. Completeness and factual accuracy take priority over brevity.

Before writing, review the conversation chronologically and reconcile later corrections with
earlier statements. Pay special attention to explicit user feedback, rejected approaches, tool
evidence, and the work in progress immediately before this request.

Output only a structured handoff with these sections:

1. Current Objective
- State the user's latest request and the exact task currently in progress.
- Explain precisely where the work stopped.

2. User Requirements and Corrections
- Record active requirements, constraints, preferences, and acceptance criteria.
- Preserve critical user wording verbatim when paraphrasing could change the meaning.
- Record directions the user rejected or corrected.

3. Decision Status
- Separate confirmed decisions, tentative proposals, rejected options, and unresolved questions.
- Never turn an estimate, suggestion, or unverified hypothesis into an accepted decision.

4. Completed Work
- List concrete changes already made and why.
- Include exact file paths, symbols, APIs, data structures, commands, and important values.

5. Verification Evidence
- Record commands executed and meaningful results.
- Distinguish real end-to-end acceptance, automated tests, partial checks, and unverified work.
- Preserve important logs, errors, screenshots, seeds, identifiers, and measurements.

6. Errors and Lessons
- Record failures, root causes, attempted approaches, and successful fixes.
- Preserve user feedback that changed the implementation direction.

7. Repository State
- Record the working directory, repository, branch, relevant commits, and uncommitted changes.
- State whether committing, pushing, or destructive operations were authorized.

8. Pending Work
- Give an ordered, executable task list.
- Include the exact next action.
- Do not revive completed, rejected, or unrelated older tasks.

9. Critical References
- Include files, commands, URLs, identifiers, configuration values, and exact snippets needed to
  continue without rediscovery.
- Mark details that require consulting the original rollout rather than guessing.

Accuracy rules:
- Do not infer facts that were not established.
- Explicitly label uncertainty and conflicting evidence.
- Preserve exact numbers, names, paths, commands, errors, and status.
- Prefer detailed factual notes over narrative prose.
- Do not omit consequential information merely to keep the handoff concise.
