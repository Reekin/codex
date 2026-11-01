//! Bounded recovery prompt for a regular turn that ended without a visible final answer.

use super::ContextualUserFragment;
use codex_protocol::models::ContentItem;
use codex_protocol::models::ContentItemKind;
use codex_protocol::models::input_text;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct FinalAnswerRetry;

impl ContextualUserFragment for FinalAnswerRetry {
    // Harness guidance must not count as a user turn, user message, or notify input.
    fn role(&self) -> &'static str {
        "developer"
    }

    fn content_kind(&self) -> ContentItemKind {
        ContentItemKind("generic.final_answer_retry".to_string())
    }

    fn markers(&self) -> (&'static str, &'static str) {
        Self::type_markers()
    }

    fn type_markers() -> (&'static str, &'static str) {
        ("", "")
    }

    fn body(&self) -> ContentItem {
        input_text(
            concat!(
                "The previous response did not provide a visible final answer. ",
                "Continue the same turn and provide a non-empty final answer to the user's request. ",
                "Do not call tools unless necessary."
            )
            .to_string(),
        )
    }
}
