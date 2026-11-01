//! Model-visible records local compaction adds: the private request question and the bounded
//! summaries that stand in for removed history.

use codex_context_fragments::ContextualUserFragment;
use codex_history::LocalCompactionKind;
use codex_history::LocalCompactionSource;
use codex_history::ResponseItemEnvelope;
use codex_protocol::models::ContentItem;
use codex_protocol::models::ContentItemKind;
use codex_protocol::models::input_text;
use codex_utils_output_truncation::TruncationPolicy;
use codex_utils_output_truncation::truncate_text;

use crate::CompactionError;
use crate::MAX_SUMMARY_BYTES;
use crate::SourceRange;

/// Private request protocol, appended to a copy of the ordinary request. Each serialized
/// question has a hard byte cap.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LocalCompactionRequest(String);

impl LocalCompactionRequest {
    /// Instructions and source labels.
    pub const MAX_BYTES: usize = 9_000;
    /// A marking request also lists its candidates; still well below 10k tokens.
    pub const MAX_CLASSIFY_BYTES: usize = 24_000;

    pub fn new(
        marker: &str,
        payload: serde_json::Value,
        max_bytes: usize,
    ) -> Result<Self, CompactionError> {
        let text = format!("{marker}\n{payload}");
        if text.len() > max_bytes {
            return Err(CompactionError::Invalid(
                "local compaction guidance exceeds its request limit",
            ));
        }
        Ok(Self(text))
    }
}

impl ContextualUserFragment for LocalCompactionRequest {
    fn role(&self) -> &'static str {
        "user"
    }

    fn content_kind(&self) -> ContentItemKind {
        ContentItemKind("compaction.request".to_string())
    }

    fn markers(&self) -> (&'static str, &'static str) {
        Self::type_markers()
    }

    fn body(&self) -> ContentItem {
        input_text(self.0.clone())
    }

    fn type_markers() -> (&'static str, &'static str) {
        ("", "")
    }
}

/// Validated, bounded semantic memory, with original content available in the local archive.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LocalCompactionFragment {
    pub text: String,
    pub source: LocalCompactionSource,
}

impl ContextualUserFragment for LocalCompactionFragment {
    fn role(&self) -> &'static str {
        "user"
    }

    fn content_kind(&self) -> ContentItemKind {
        ContentItemKind("compaction.local".to_string())
    }

    fn markers(&self) -> (&'static str, &'static str) {
        Self::type_markers()
    }

    fn body(&self) -> ContentItem {
        input_text(format!(
            "{:?}; original range {}..{}\n{}\nUse recall_read_item for exact originals; recall_list_turns and recall_search locate older turns.",
            self.source.kind, self.source.first_item_id, self.source.last_item_id, self.text
        ))
    }

    fn type_markers() -> (&'static str, &'static str) {
        ("<local_compaction>", "</local_compaction>")
    }
}

/// The handoff summary of `range`, as the history record that replaces it. An empty summary
/// is a failed analysis.
pub fn summary_record(
    text: &str,
    range: &SourceRange,
    turn_id: &str,
) -> Result<ResponseItemEnvelope, CompactionError> {
    let text = text.trim();
    if text.is_empty() {
        return Err(CompactionError::Invalid("the summary is empty"));
    }
    let source = LocalCompactionSource {
        first_item_id: range.first_item_id.clone(),
        last_item_id: range.last_item_id.clone(),
        kind: LocalCompactionKind::OldestOverview,
    };
    let mut envelope =
        ResponseItemEnvelope::new(ContextualUserFragment::into(LocalCompactionFragment {
            text: truncate_text(text, TruncationPolicy::Bytes(MAX_SUMMARY_BYTES)),
            source: source.clone(),
        }));
    envelope.metadata.get_or_insert_default().local_compaction = Some(source);
    envelope.item.set_turn_id_if_missing(turn_id);
    Ok(envelope)
}
