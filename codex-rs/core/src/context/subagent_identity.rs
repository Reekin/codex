use super::ContextualUserFragment;

/// Identity facts for the spawned subagent that owns the current session.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct SubagentIdentity {
    text: String,
}

impl SubagentIdentity {
    pub(crate) fn new(text: String) -> Self {
        Self { text }
    }
}

impl ContextualUserFragment for SubagentIdentity {
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
        self.text.clone()
    }
}
