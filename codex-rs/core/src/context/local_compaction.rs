use codex_history::LocalCompactionSource;
use codex_protocol::models::ContentItemKind;

use super::ContextualUserFragment;

/// Private request protocol. Each serialized instruction item has a hard byte cap.
pub(crate) struct LocalCompactionRequest(String);

impl LocalCompactionRequest {
    /// Instructions and source labels.
    pub(crate) const MAX_BYTES: usize = 9_000;
    /// A marking request also lists its candidates; still well below 10k tokens.
    pub(crate) const MAX_CLASSIFY_BYTES: usize = 24_000;

    pub(crate) fn new(
        marker: &str,
        payload: serde_json::Value,
        max_bytes: usize,
    ) -> codex_protocol::error::Result<Self> {
        let text = format!("{marker}\n{payload}");
        if text.len() > max_bytes {
            return Err(codex_protocol::error::CodexErr::InvalidRequest(
                "Local compaction guidance exceeds its request limit".to_string(),
            ));
        }
        Ok(Self(text))
    }
}

impl ContextualUserFragment for LocalCompactionRequest {
    fn content_kind(&self) -> ContentItemKind {
        ContentItemKind("compaction.request".to_string())
    }
    fn role(&self) -> &'static str {
        "user"
    }
    fn markers(&self) -> (&'static str, &'static str) {
        Self::type_markers()
    }
    fn type_markers() -> (&'static str, &'static str) {
        ("", "")
    }
    fn body(&self) -> String {
        self.0.clone()
    }
}

/// Validated, bounded semantic memory, with original content available in the local archive.
#[derive(Debug, Clone)]
pub(crate) struct LocalCompactionFragment {
    pub(crate) text: String,
    pub(crate) source: LocalCompactionSource,
}

impl ContextualUserFragment for LocalCompactionFragment {
    fn content_kind(&self) -> ContentItemKind {
        ContentItemKind("compaction.local".to_string())
    }

    fn role(&self) -> &'static str {
        "user"
    }

    fn markers(&self) -> (&'static str, &'static str) {
        Self::type_markers()
    }

    fn type_markers() -> (&'static str, &'static str) {
        ("<local_compaction>", "</local_compaction>")
    }

    fn body(&self) -> String {
        format!(
            "{:?}; original range {}..{}\n{}\nUse recall_read_item for exact originals; recall_list_turns and recall_search locate older turns.",
            self.source.kind, self.source.first_item_id, self.source.last_item_id, self.text
        )
    }
}
