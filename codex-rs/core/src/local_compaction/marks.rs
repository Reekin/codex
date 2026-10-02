//! Durable validated tool marks.
//!
//! Each validated batch appends one JSON line. Restoring replays the latest decision per record
//! and keeps only records that are still present with unchanged content. A forked thread also
//! reads its parent's marks and copies the still-valid ones into its own file, so later resumes
//! and further forks do not depend on the parent.

use std::collections::HashMap;
use std::collections::HashSet;
use std::path::Path;
use std::path::PathBuf;

use codex_context_compaction::Decision;
use codex_context_compaction::StagedDecisions;
use codex_history::ResponseItemEnvelope;
use codex_protocol::ThreadId;
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
    Some(thread_path(config, sess.thread_id))
}

fn thread_path(config: &Config, thread_id: ThreadId) -> PathBuf {
    config
        .codex_home
        .as_path()
        .join("local_compaction")
        .join(format!("{thread_id}.jsonl"))
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

/// Restores this thread's marks, plus still-valid marks inherited from the thread it forked from.
pub(super) async fn restore(
    sess: &Session,
    config: &Config,
    current: &[ResponseItemEnvelope],
) -> StagedDecisions {
    let Some(own) = path(sess, config) else {
        return StagedDecisions::default();
    };
    let mut latest = HashMap::new();
    let mut inherited = HashSet::new();
    if let Some(parent) = sess.forked_from_thread_id().await {
        for mark in read(&thread_path(config, parent)).await {
            inherited.insert(mark.decision.id().to_string());
            latest.insert(mark.decision.id().to_string(), mark);
        }
    }
    // The thread's own decisions are newer than anything it inherited.
    for mark in read(&own).await {
        inherited.remove(mark.decision.id());
        latest.insert(mark.decision.id().to_string(), mark);
    }
    let restored = StagedDecisions::from_marks(
        current
            .iter()
            .filter_map(|item| {
                let mark = latest.remove(item.item.id()?.as_str())?;
                (mark.fingerprint == fingerprint(item)).then(|| (item.clone(), mark.decision))
            })
            .collect(),
    );
    let adopted = StagedDecisions::from_marks(
        restored
            .marks()
            .filter(|(_, decision)| inherited.contains(decision.id()))
            .map(|(item, decision)| (item.clone(), decision.clone()))
            .collect(),
    );
    append(own, &adopted).await;
    restored
}

async fn read(path: &Path) -> Vec<Mark> {
    let text = match tokio::fs::read_to_string(path).await {
        Ok(text) => text,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Vec::new(),
        Err(error) => {
            tracing::warn!(%error, path = %path.display(), "cannot read local tool marks");
            return Vec::new();
        }
    };
    // A torn final line from an interrupted append loses only that batch.
    text.lines()
        .filter_map(|line| serde_json::from_str::<Vec<Mark>>(line).ok())
        .flatten()
        .collect()
}

fn fingerprint(item: &ResponseItemEnvelope) -> String {
    let bytes = serde_json::to_vec(&item.item).unwrap_or_default();
    format!("{:x}", Sha1::digest(bytes))
}
