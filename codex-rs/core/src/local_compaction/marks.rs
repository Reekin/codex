//! Durable validated tool marks.
//!
//! Each validated batch appends one JSON line. Restoring replays the latest decision per record
//! and keeps only records that are still present with unchanged content.

use std::collections::HashMap;
use std::path::PathBuf;

use codex_context_compaction::Decision;
use codex_context_compaction::StagedDecisions;
use codex_history::ResponseItemEnvelope;
use serde::Deserialize;
use serde::Serialize;
use sha1::Digest;
use sha1::Sha1;
use tokio::io::AsyncWriteExt;

use crate::config::Config;
use crate::session::session::Session;

#[derive(Serialize, Deserialize)]
struct Mark {
    fingerprint: String,
    decision: Decision,
}

/// Only threads with a rollout can resume; ephemeral threads keep marks in memory.
pub(super) fn path(sess: &Session, config: &Config) -> Option<PathBuf> {
    sess.live_thread()?;
    Some(
        config
            .codex_home
            .as_path()
            .join("local_compaction")
            .join(format!("{}.jsonl", sess.thread_id)),
    )
}

pub(super) async fn append(path: PathBuf, decisions: &StagedDecisions) {
    let marks: Vec<_> = decisions
        .marks()
        .map(|(item, decision)| Mark {
            fingerprint: fingerprint(item),
            decision: decision.clone(),
        })
        .collect();
    if marks.is_empty() {
        return;
    }
    let result = async {
        let mut line = serde_json::to_vec(&marks)?;
        line.push(b'\n');
        if let Some(parent) = path.parent() {
            tokio::fs::create_dir_all(parent).await?;
        }
        let mut file = tokio::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(&path)
            .await?;
        file.write_all(&line).await?;
        file.flush().await
    }
    .await;
    if let Err(error) = result {
        tracing::warn!(%error, path = %path.display(), "cannot persist local tool marks");
    }
}

pub(super) async fn load(path: PathBuf, current: &[ResponseItemEnvelope]) -> StagedDecisions {
    let text = match tokio::fs::read_to_string(&path).await {
        Ok(text) => text,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => {
            return StagedDecisions::default();
        }
        Err(error) => {
            tracing::warn!(%error, path = %path.display(), "cannot read local tool marks");
            return StagedDecisions::default();
        }
    };
    let mut latest = HashMap::new();
    // A torn final line from an interrupted append loses only that batch.
    for marks in text
        .lines()
        .filter_map(|line| serde_json::from_str::<Vec<Mark>>(line).ok())
    {
        for mark in marks {
            latest.insert(mark.decision.id().to_string(), mark);
        }
    }
    StagedDecisions::from_marks(
        current
            .iter()
            .filter_map(|item| {
                let mark = latest.remove(item.item.id()?.as_str())?;
                (mark.fingerprint == fingerprint(item)).then(|| (item.clone(), mark.decision))
            })
            .collect(),
    )
}

fn fingerprint(item: &ResponseItemEnvelope) -> String {
    let bytes = serde_json::to_vec(&item.item).unwrap_or_default();
    format!("{:x}", Sha1::digest(bytes))
}
