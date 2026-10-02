//! Bounded original-history queries shared by model tools and the CLI.

use std::collections::HashSet;
use std::io;
use std::io::BufRead;
use std::io::BufReader;
use std::io::Read;
use std::path::Path;

use codex_protocol::protocol::EventMsg;
use serde::Deserialize;
use serde::Serialize;
use serde_json::Value;
use serde_json::json;

use crate::RolloutItem;

const MAX_PAGE_BYTES: usize = 8000;

/// All offsets are zero-based; character offsets count Unicode scalar values, not bytes.
#[derive(Debug, Deserialize, Serialize)]
#[serde(tag = "action", rename_all = "snake_case", deny_unknown_fields)]
pub enum RecallQuery {
    ListTurns {
        #[serde(default)]
        offset: usize,
        #[serde(default = "default_limit")]
        limit: usize,
    },
    Search {
        query: String,
        #[serde(default)]
        turn_id: Option<String>,
        #[serde(default)]
        offset: usize,
        #[serde(default = "default_limit")]
        limit: usize,
    },
    ReadTurn {
        turn_id: String,
        #[serde(default)]
        detail: RecallDetail,
        #[serde(default)]
        offset: usize,
        #[serde(default = "default_limit")]
        limit: usize,
    },
    ReadItem {
        item_id: String,
        #[serde(default)]
        start_char: usize,
        #[serde(default = "default_chars")]
        max_chars: usize,
    },
}

/// Summary returns brief previews; full indexes every original item with bounded previews.
/// Follow each item's `next_char` with ReadItem to retrieve the remaining original JSON.
#[derive(Debug, Default, Deserialize, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum RecallDetail {
    #[default]
    Summary,
    Dialogue,
    Tools,
    Full,
}

fn default_limit() -> usize {
    10
}

fn default_chars() -> usize {
    4000
}

struct OriginalItem {
    id: String,
    turn_id: String,
    kind: String,
    dialogue: bool,
    tool: bool,
    text: String,
}

/// A read-only snapshot of the canonical response records, never replacement history.
#[derive(Default)]
pub struct RecallArchive {
    items: Vec<OriginalItem>,
    seen: HashSet<String>,
    turn_id: String,
}

impl RecallArchive {
    /// Loads the selected rollout and only the immutable ancestor prefixes it references.
    /// A fork with a history base requires its owning Codex home to resolve that lineage.
    pub async fn load(path: &Path, codex_home: Option<&Path>) -> io::Result<Self> {
        let mut segments = Vec::new();
        let mut path = path.to_path_buf();
        let mut end = None;
        let mut seen = HashSet::new();
        loop {
            if !seen.insert(crate::plain_rollout_path(&path)) {
                return Err(io::Error::other("recall archive lineage contains a cycle"));
            }
            let meta = crate::read_session_meta_line(&path).await?;
            let base = meta.meta.history_base;
            segments.push((path, end));
            let Some(base) = base else { break };
            let home = codex_home.ok_or_else(|| {
                io::Error::other("fork recall requires the owning Codex home (--codex-home)")
            })?;
            path = crate::find_rollout_path_by_rollout_id(home, base.thread_id)
                .await?
                .ok_or_else(|| io::Error::other("recall ancestor rollout is unavailable"))?;
            end = Some(base);
        }
        tokio::task::spawn_blocking(move || {
            let mut archive = Self::default();
            for (path, end) in segments.into_iter().rev() {
                let file = crate::open_rollout_seekable_reader(&path)?;
                let length = file.metadata()?.len();
                let bound = end.as_ref().map_or(length, |end| end.end_byte_offset);
                if bound > length {
                    return Err(io::Error::other("recall ancestor prefix is incomplete"));
                }
                for (index, line) in BufReader::new(file.take(bound)).lines().enumerate() {
                    let line = line?;
                    if line.trim().is_empty() {
                        continue;
                    }
                    let record = crate::parse_rollout_line(&line).map_err(io::Error::other)?;
                    if end.as_ref().is_some_and(|end| {
                        record
                            .ordinal
                            .is_some_and(|ordinal| ordinal >= end.end_ordinal_exclusive)
                    }) {
                        continue;
                    }
                    let plain_path = crate::plain_rollout_path(&path);
                    let source = plain_path.file_name().unwrap_or_default().to_string_lossy();
                    archive.record(
                        record.item,
                        &format!("{source}:{}", record.ordinal.unwrap_or(index as u64)),
                    )?;
                }
            }
            Ok(archive)
        })
        .await
        .map_err(io::Error::other)?
    }

    fn record(&mut self, item: RolloutItem, source: &str) -> io::Result<()> {
        match item {
            RolloutItem::TurnContext(context) => {
                if let Some(id) = context.turn_id {
                    self.turn_id = id;
                }
            }
            RolloutItem::EventMsg(EventMsg::TurnStarted(event)) => self.turn_id = event.turn_id,
            RolloutItem::ResponseItem(envelope) => {
                if envelope
                    .metadata
                    .as_ref()
                    .is_some_and(|metadata| metadata.local_compaction.is_some())
                {
                    return Ok(());
                }
                let item = envelope.item;
                let value = serde_json::to_value(&item).map_err(io::Error::other)?;
                let kind = value["type"].as_str().unwrap_or("unknown").to_owned();
                let id = item
                    .id()
                    .map(ToString::to_string)
                    .or_else(|| value["call_id"].as_str().map(|id| format!("{kind}:{id}")))
                    .unwrap_or_else(|| format!("record:{source}"));
                // The first archived response wins even if a later record reuses its ID.
                if !self.seen.insert(id.clone()) {
                    return Ok(());
                }
                let turn_id = item.turn_id().unwrap_or(&self.turn_id).to_owned();
                let dialogue = kind == "agent_message"
                    || (kind == "message"
                        && matches!(value["role"].as_str(), Some("user" | "assistant")));
                let tool = kind.contains("call") || kind.ends_with("_output");
                self.items.push(OriginalItem {
                    id,
                    turn_id: if turn_id.is_empty() {
                        "unattributed".into()
                    } else {
                        turn_id
                    },
                    kind,
                    dialogue,
                    tool,
                    text: serde_json::to_string(&item).map_err(io::Error::other)?,
                });
            }
            // Compacted replacements and event projections are not original response records.
            _ => {}
        }
        Ok(())
    }

    /// Returns a bounded JSON page; original content is serialized ResponseItem JSON.
    pub fn query(&self, query: RecallQuery) -> io::Result<Value> {
        match query {
            RecallQuery::ListTurns { offset, limit } => {
                let mut seen = HashSet::new();
                let turns = self.items.iter().filter(|item| seen.insert(item.turn_id.as_str()))
                    .map(|item| json!({"turn_id": item.turn_id, "preview": slice_chars(&item.text, 0, 160)})).collect::<Vec<_>>();
                page(turns, offset, limit)
            }
            RecallQuery::Search {
                query,
                turn_id,
                offset,
                limit,
            } => {
                if query.trim().is_empty() {
                    return Err(io::Error::other("search query must not be empty"));
                }
                let needle = query.to_lowercase();
                let matches = self
                    .items
                    .iter()
                    .filter(|item| {
                        turn_id.as_ref().is_none_or(|id| *id == item.turn_id)
                            && item.text.to_lowercase().contains(&needle)
                    })
                    .map(|item| item_preview(item, 512))
                    .collect();
                page(matches, offset, limit)
            }
            RecallQuery::ReadTurn {
                turn_id,
                detail,
                offset,
                limit,
            } => {
                if !self.items.iter().any(|item| item.turn_id == turn_id) {
                    return Err(io::Error::other(
                        "turn is not present in the original archive",
                    ));
                }
                let items = self
                    .items
                    .iter()
                    .filter(|item| item.turn_id == turn_id)
                    .filter(|item| match detail {
                        RecallDetail::Dialogue => item.dialogue,
                        RecallDetail::Tools => item.tool,
                        RecallDetail::Summary | RecallDetail::Full => true,
                    })
                    .map(|item| {
                        item_preview(
                            item,
                            if matches!(detail, RecallDetail::Summary) {
                                160
                            } else {
                                512
                            },
                        )
                    })
                    .collect();
                page(items, offset, limit)
            }
            RecallQuery::ReadItem {
                item_id,
                start_char,
                max_chars,
            } => {
                let item = self
                    .items
                    .iter()
                    .find(|item| item.id == item_id)
                    .ok_or_else(|| {
                        io::Error::other("item is not present in the original archive")
                    })?;
                let total_chars = item.text.chars().count();
                let make_page = |count| {
                    let text = slice_chars(&item.text, start_char, count);
                    let end = start_char.saturating_add(text.chars().count());
                    json!({"item_id": item.id, "turn_id": item.turn_id, "text": text,
                        "start_char": start_char, "total_chars": total_chars,
                        "next_char": (end < total_chars).then_some(end)})
                };
                let mut low = 0;
                let mut high = max_chars
                    .clamp(1, 8000)
                    .min(total_chars.saturating_sub(start_char));
                while low < high {
                    let middle = low + (high - low).div_ceil(2);
                    if make_page(middle).to_string().len() <= MAX_PAGE_BYTES {
                        low = middle;
                    } else {
                        high = middle - 1;
                    }
                }
                let page = make_page(low);
                if page.to_string().len() > MAX_PAGE_BYTES || (low == 0 && start_char < total_chars)
                {
                    return Err(io::Error::other(
                        "archive item identifiers exceed recall page budget",
                    ));
                }
                Ok(page)
            }
        }
    }
}

fn item_preview(item: &OriginalItem, chars: usize) -> Value {
    let text = slice_chars(&item.text, 0, chars);
    let next = text.chars().count();
    json!({"item_id": item.id, "turn_id": item.turn_id, "kind": item.kind, "text": text,
        "next_char": (next < item.text.chars().count()).then_some(next)})
}

fn slice_chars(text: &str, start: usize, length: usize) -> String {
    text.chars().skip(start).take(length).collect()
}

fn page(items: Vec<Value>, offset: usize, limit: usize) -> io::Result<Value> {
    let total = items.len();
    let mut data = items
        .into_iter()
        .skip(offset)
        .take(limit.clamp(1, 10))
        .collect::<Vec<_>>();
    loop {
        let end = offset.saturating_add(data.len());
        let result =
            json!({"data": data, "total": total, "next_offset": (end < total).then_some(end)});
        if result.to_string().len() <= MAX_PAGE_BYTES {
            return Ok(result);
        }
        if data.len() <= 1 {
            return Err(io::Error::other(
                "archive item identifiers exceed recall page budget",
            ));
        }
        data.pop();
    }
}

#[cfg(test)]
#[path = "recall_tests.rs"]
mod tests;
