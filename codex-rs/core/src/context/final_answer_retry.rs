use super::ContextualUserFragment;
use codex_protocol::models::ContentItemKind;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct FinalAnswerRetry;

impl ContextualUserFragment for FinalAnswerRetry {
    fn role(&self) -> &'static str {
        "user"
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

    fn body(&self) -> String {
        concat!(
            "The previous response did not provide a visible final answer. ",
            "Continue the same turn and provide a non-empty final answer to the user's request. ",
            "Do not call tools unless necessary."
        )
        .to_string()
    }
}
