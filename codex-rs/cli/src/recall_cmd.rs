use std::path::PathBuf;

use clap::Parser;
use codex_rollout::recall::RecallArchive;
use codex_rollout::recall::RecallQuery;

/// Query original local history without starting a model or contacting a service.
#[derive(Debug, Parser)]
pub struct RecallCommand {
    /// Canonical rollout JSONL (plain or compressed).
    #[arg(long)]
    rollout: PathBuf,
    /// Owning Codex home, needed to resolve a fork's inherited history prefix.
    #[arg(long)]
    codex_home: Option<PathBuf>,
    /// JSON query: action=list_turns|search|read_turn|read_item, with the model tool's arguments.
    #[arg(long)]
    query: String,
    /// For read_item, add the page's original images (data URLs) as an `images` array.
    #[arg(long)]
    images: bool,
}

impl RecallCommand {
    pub async fn run(self) -> anyhow::Result<()> {
        let query: RecallQuery = serde_json::from_str(&self.query)?;
        let archive = RecallArchive::load(&self.rollout, self.codex_home.as_deref()).await?;
        let images = archive.read_item_images(&query);
        let mut page = archive.query(query)?;
        if self.images {
            page["images"] = images.into();
        }
        println!("{page}");
        Ok(())
    }
}
