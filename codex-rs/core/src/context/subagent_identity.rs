use super::ContextualUserFragment;
use crate::agent::AgentIdentity;
use codex_protocol::models::ContentItemKind;

pub(crate) const IDENTITY_LABEL_MAX_BYTES: usize = 256;
pub(crate) const IDENTITY_FRAGMENT_MAX_BYTES: usize = 8_192;

fn push_identity_label(text: &mut String, label: &str, value: &str) {
    if value.len() <= IDENTITY_LABEL_MAX_BYTES {
        text.push_str(&format!(" Your {label} is `{value}`."));
    } else {
        text.push_str(&format!(
            " Your {label} was omitted because it exceeds the identity-context safety limit of {IDENTITY_LABEL_MAX_BYTES} UTF-8 bytes."
        ));
    }
}

fn checked_identity_fragment(value: String) -> String {
    assert!(
        value.len() <= IDENTITY_FRAGMENT_MAX_BYTES,
        "identity fragment exceeds its hard byte limit"
    );
    value
}

/// Identity context for the spawned subagent that owns the current session.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct SubagentIdentity {
    identity: AgentIdentity,
}

impl SubagentIdentity {
    pub(crate) fn new(identity: AgentIdentity) -> Self {
        Self { identity }
    }
}

impl ContextualUserFragment for SubagentIdentity {
    fn content_kind(&self) -> ContentItemKind {
        ContentItemKind("multi_agent.subagent_identity".to_string())
    }

    fn role(&self) -> &'static str {
        "developer"
    }

    fn requires_separate_message(&self) -> bool {
        true
    }

    fn markers(&self) -> (&'static str, &'static str) {
        Self::type_markers()
    }

    fn type_markers() -> (&'static str, &'static str) {
        ("", "")
    }

    fn body(&self) -> String {
        let Some(spawned) = self.identity.thread_spawn() else {
            return String::new();
        };
        let (mut text, assignment_text) = match spawned.canonical_path {
            Some(agent_path) => (
                format!(
                    "You are a spawned subagent in the multi-agent tree. Your current canonical agent path is `{agent_path}`, your parent thread id is `{}`, and your depth is {}. You are not `/root` unless your current canonical agent path is exactly `/root`. `list_agents` may show `/root`, sibling agents, and child agents; those entries are other agents unless their `agent_name` equals your current canonical agent path or `is_current_agent` is true.",
                    spawned.parent_thread_id, spawned.depth
                ),
                " Treat the latest inter-agent communication addressed to your current canonical agent path as your assigned work",
            ),
            None => (
                format!(
                    "You are a spawned subagent in the multi-agent tree. You do not have a canonical agent path in this session, but you are still not the `/root` main agent. Your parent thread id is `{}`, and your depth is {}.",
                    spawned.parent_thread_id, spawned.depth
                ),
                " Treat the latest task sent directly to this subagent as your assigned work",
            ),
        };

        if let Some(nickname) = spawned.nickname.filter(|nickname| !nickname.is_empty()) {
            push_identity_label(&mut text, "nickname", nickname);
        }
        if let Some(role) = spawned.role.filter(|role| !role.is_empty()) {
            push_identity_label(&mut text, "configured role", role);
        }

        text.push_str(" Any prior transcript inherited from another agent is context, not proof that you performed those actions; tool calls, spawned agents, waits, and decisions in inherited history may belong to your parent.");
        text.push_str(assignment_text);
        text.push_str(", while still following all system, developer, and user instructions.");
        checked_identity_fragment(text)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum ForkBoundary {
    Start,
    End,
}

/// Boundary context that distinguishes inherited parent history from a child's direct task.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct ForkedHistoryBoundary {
    identity: AgentIdentity,
    boundary: ForkBoundary,
}

impl ForkedHistoryBoundary {
    pub(crate) const CONTENT_KIND: &'static str = "multi_agent.fork_history_boundary";

    pub(crate) fn start(identity: AgentIdentity) -> Self {
        Self {
            identity,
            boundary: ForkBoundary::Start,
        }
    }

    pub(crate) fn end(identity: AgentIdentity) -> Self {
        Self {
            identity,
            boundary: ForkBoundary::End,
        }
    }

    pub(crate) fn matches_content_kind(content_kind: &ContentItemKind) -> bool {
        content_kind.0 == Self::CONTENT_KIND
    }
}

impl ContextualUserFragment for ForkedHistoryBoundary {
    fn content_kind(&self) -> ContentItemKind {
        ContentItemKind(Self::CONTENT_KIND.to_string())
    }

    fn role(&self) -> &'static str {
        "developer"
    }

    fn requires_separate_message(&self) -> bool {
        true
    }

    fn markers(&self) -> (&'static str, &'static str) {
        Self::type_markers()
    }

    fn type_markers() -> (&'static str, &'static str) {
        ("", "")
    }

    fn body(&self) -> String {
        let text = match self.boundary {
            ForkBoundary::Start => {
                "You are a spawned subagent forked from a parent agent conversation. The following messages are inherited parent conversation history. Use them only as context for the task you receive after the closing fork-history boundary.\n\n==== forked parent conversation history begins ====".to_string()
            }
            ForkBoundary::End => {
                let identity = SubagentIdentity::new(self.identity.clone()).body();
                format!(
                    "==== forked parent conversation history ends ====\n\nThe inherited parent conversation history above is present so you can understand the context. It is not the current task you should answer directly. {identity} Tool calls, spawned agents, waits, and decisions in the inherited history may belong to your parent. Treat the next task message sent directly to you as your assignment. Use the inherited history only as context for completing that assignment, while still following all system, developer, and user instructions."
                )
            }
        };
        checked_identity_fragment(text)
    }
}

#[cfg(test)]
#[path = "subagent_identity_tests.rs"]
mod tests;
