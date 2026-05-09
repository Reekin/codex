use codex_app_server_protocol::ChatTreeProjection;
use codex_app_server_protocol::apply_chat_tree_event_to_projection;
use codex_app_server_protocol::chat_tree_overlay_entries;
use codex_app_server_protocol::refresh_chat_tree_projection;
use codex_protocol::protocol::EventMsg;
use codex_protocol::protocol::Op;
use crossterm::event::KeyCode;
use crossterm::event::KeyEvent;
use crossterm::event::KeyModifiers;
use ratatui::buffer::Buffer;
use ratatui::layout::Rect;
use ratatui::style::Stylize;
use ratatui::text::Line;
use ratatui::widgets::Block;
use ratatui::widgets::Paragraph;
use ratatui::widgets::Widget;
use unicode_width::UnicodeWidthStr;

use crate::app_command::AppCommand;
use crate::app_event::AppEvent;
use crate::app_event_sender::AppEventSender;
use crate::bottom_pane::BottomPaneView;
use crate::bottom_pane::CancellationEvent;
use crate::bottom_pane::ViewCompletion;
use crate::bottom_pane::popup_consts::MAX_POPUP_ROWS;
use crate::render::renderable::Renderable;
use crate::style::user_message_style;

#[cfg(test)]
use codex_app_server_protocol::ChatTreeNode;
#[cfg(test)]
use codex_app_server_protocol::ChatTreeNodeStatus;
#[cfg(test)]
use codex_protocol::protocol::ChatTreeCurrentNodeChangedEvent;
#[cfg(test)]
use codex_protocol::protocol::ChatTreeNodeStartedEvent;

#[derive(Clone, Debug)]
pub(super) struct ChatTreeUiState {
    projection: ChatTreeProjection,
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct ChatTreeRow {
    node_id: String,
    depth: usize,
    summary: String,
    is_current: bool,
}

impl ChatTreeUiState {
    pub(super) fn set_projection(&mut self, projection: ChatTreeProjection) {
        self.projection = projection;
    }

    pub(super) fn revision(&self) -> u64 {
        self.projection.revision
    }

    #[allow(dead_code)]
    pub(super) fn apply_event(&mut self, msg: &EventMsg) {
        apply_chat_tree_event_to_projection(&mut self.projection, msg);
    }

    #[allow(dead_code)]
    pub(super) fn apply_replay_event(&mut self, msg: &EventMsg) {
        match chat_tree_event_revision(msg) {
            Some(revision) if revision <= self.projection.revision => return,
            None if self.projection.revision != 0 => return,
            Some(_) | None => {}
        }
        self.apply_event(msg);
    }

    pub(super) fn view(&self, app_event_tx: AppEventSender) -> Option<ChatTreeView> {
        let rows = self.rows();
        (!rows.is_empty()).then(|| {
            let selected_idx = rows
                .iter()
                .position(|row| row.is_current)
                .unwrap_or_else(|| rows.len().saturating_sub(1));
            ChatTreeView::new(rows, selected_idx, self.projection.revision, app_event_tx)
        })
    }

    fn rows(&self) -> Vec<ChatTreeRow> {
        chat_tree_overlay_entries(&self.projection)
            .into_iter()
            .map(|entry| ChatTreeRow {
                node_id: entry.node_id,
                depth: entry.depth,
                summary: entry.summary,
                is_current: entry.is_current,
            })
            .collect()
    }

    #[allow(dead_code)]
    fn rebuild_visible_projection(&mut self) {
        refresh_chat_tree_projection(&mut self.projection);
    }

    #[cfg(test)]
    pub(super) fn projection(&self) -> &ChatTreeProjection {
        &self.projection
    }
}

#[allow(dead_code)]
fn chat_tree_event_revision(msg: &EventMsg) -> Option<u64> {
    match msg {
        EventMsg::ChatTreeNodeStarted(payload) => Some(payload.revision),
        EventMsg::ChatTreeNodeFinalized(payload) => Some(payload.revision),
        EventMsg::ChatTreeNodeSummaryUpdated(payload) => Some(payload.revision),
        EventMsg::ChatTreeCurrentNodeChanged(payload) => Some(payload.revision),
        EventMsg::ThreadRolledBack(_) => None,
        _ => None,
    }
}

impl Default for ChatTreeUiState {
    fn default() -> Self {
        Self {
            projection: ChatTreeProjection {
                version: 1,
                revision: 0,
                current_node_id: None,
                visible_node_ids: Vec::new(),
                visible_turn_ids: Vec::new(),
                nodes: Vec::new(),
            },
        }
    }
}

pub(super) struct ChatTreeView {
    rows: Vec<ChatTreeRow>,
    selected_idx: usize,
    scroll_top: usize,
    revision: u64,
    app_event_tx: AppEventSender,
    completion: Option<ViewCompletion>,
}

impl ChatTreeView {
    fn new(
        rows: Vec<ChatTreeRow>,
        selected_idx: usize,
        revision: u64,
        app_event_tx: AppEventSender,
    ) -> Self {
        let mut view = Self {
            rows,
            selected_idx,
            scroll_top: 0,
            revision,
            app_event_tx,
            completion: None,
        };
        view.ensure_selected_visible();
        view
    }

    fn move_up(&mut self) {
        if self.rows.is_empty() {
            return;
        }
        self.selected_idx = if self.selected_idx == 0 {
            self.rows.len() - 1
        } else {
            self.selected_idx - 1
        };
        self.ensure_selected_visible();
    }

    fn move_down(&mut self) {
        if self.rows.is_empty() {
            return;
        }
        self.selected_idx = (self.selected_idx + 1) % self.rows.len();
        self.ensure_selected_visible();
    }

    fn ensure_selected_visible(&mut self) {
        if self.selected_idx < self.scroll_top {
            self.scroll_top = self.selected_idx;
        }
        let bottom = self.scroll_top + MAX_POPUP_ROWS.saturating_sub(1);
        if self.selected_idx > bottom {
            self.scroll_top = self.selected_idx + 1 - MAX_POPUP_ROWS;
        }
    }

    fn accept(&mut self) {
        let Some(row) = self.rows.get(self.selected_idx) else {
            return;
        };
        self.app_event_tx.send(AppEvent::CodexOp(AppCommand::from(
            Op::SetCurrentChatTreeNode {
                node_id: row.node_id.clone(),
                expected_revision: Some(self.revision),
            },
        )));
        self.completion = Some(ViewCompletion::Accepted);
    }

    fn cancel(&mut self) {
        self.completion = Some(ViewCompletion::Cancelled);
    }

    fn visible_rows(&self) -> &[ChatTreeRow] {
        let end = (self.scroll_top + MAX_POPUP_ROWS).min(self.rows.len());
        &self.rows[self.scroll_top..end]
    }

    fn row_lines(row: &ChatTreeRow, selected: bool, width: u16) -> Vec<Line<'static>> {
        let selector = if selected { "> " } else { "  " };
        let current = if row.is_current { "[*] " } else { "[ ] " };
        let indent = "  ".repeat(row.depth);
        let prefix = format!("{selector}{current}{indent}");
        let width = width.max(1) as usize;
        let prefix_width = UnicodeWidthStr::width(prefix.as_str());
        let subsequent_indent = " ".repeat(prefix_width);
        let options = textwrap::Options::new(width)
            .initial_indent(prefix.as_str())
            .subsequent_indent(subsequent_indent.as_str());
        textwrap::wrap(row.summary.as_str(), options)
            .into_iter()
            .map(|line| {
                let line = line.into_owned();
                if selected {
                    Line::from(line.bold())
                } else if row.is_current {
                    Line::from(line.cyan())
                } else {
                    Line::from(line)
                }
            })
            .collect()
    }
}

impl BottomPaneView for ChatTreeView {
    fn handle_key_event(&mut self, key_event: KeyEvent) {
        match key_event {
            KeyEvent {
                code: KeyCode::Up | KeyCode::Char('k'),
                modifiers: KeyModifiers::NONE,
                ..
            } => self.move_up(),
            KeyEvent {
                code: KeyCode::Down | KeyCode::Char('j'),
                modifiers: KeyModifiers::NONE,
                ..
            } => self.move_down(),
            KeyEvent {
                code: KeyCode::Enter | KeyCode::Char(' '),
                modifiers: KeyModifiers::NONE,
                ..
            } => self.accept(),
            KeyEvent {
                code: KeyCode::Esc | KeyCode::Char('q'),
                modifiers: KeyModifiers::NONE,
                ..
            }
            | KeyEvent {
                code: KeyCode::Char('c'),
                modifiers: KeyModifiers::CONTROL,
                ..
            } => self.cancel(),
            _ => {}
        }
    }

    fn is_complete(&self) -> bool {
        self.completion.is_some()
    }

    fn completion(&self) -> Option<ViewCompletion> {
        self.completion
    }

    fn on_ctrl_c(&mut self) -> CancellationEvent {
        self.cancel();
        CancellationEvent::Handled
    }

    fn prefer_esc_to_handle_key_event(&self) -> bool {
        true
    }
}

impl Renderable for ChatTreeView {
    fn render(&self, area: Rect, buf: &mut Buffer) {
        if area.is_empty() {
            return;
        }
        Block::default()
            .style(user_message_style())
            .render(area, buf);
        let inner = Rect {
            x: area.x.saturating_add(2),
            y: area.y.saturating_add(1),
            width: area.width.saturating_sub(4),
            height: area.height.saturating_sub(2),
        };
        if inner.is_empty() {
            return;
        }

        let mut lines = vec![
            Line::from("Chat tree".bold()),
            Line::from("Select a node for future turns.".dim()),
        ];
        for (offset, row) in self.visible_rows().iter().enumerate() {
            let selected = self.scroll_top + offset == self.selected_idx;
            lines.extend(Self::row_lines(row, selected, inner.width));
        }
        lines.push(Line::from(
            "Space/Enter switch · ↑/↓/j/k move · q/Esc close".dim(),
        ));
        Paragraph::new(lines).render(inner, buf);
    }

    fn desired_height(&self, width: u16) -> u16 {
        let content_width = width.saturating_sub(4).max(1);
        let rows_height = self
            .visible_rows()
            .iter()
            .enumerate()
            .map(|(offset, row)| {
                Self::row_lines(
                    row,
                    self.scroll_top + offset == self.selected_idx,
                    content_width,
                )
                .len() as u16
            })
            .sum::<u16>();
        rows_height.saturating_add(5)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use pretty_assertions::assert_eq;

    #[test]
    fn rebuild_visible_projection_preserves_legacy_visible_turn_ids() {
        let mut state = ChatTreeUiState::default();
        state.set_projection(ChatTreeProjection {
            version: 1,
            revision: 1,
            current_node_id: Some("node-a".to_string()),
            visible_node_ids: vec!["node-a".to_string()],
            visible_turn_ids: vec!["legacy-turn".to_string(), "turn-a".to_string()],
            nodes: vec![ChatTreeNode {
                node_id: "node-a".to_string(),
                parent_node_id: None,
                turn_id: Some("turn-a".to_string()),
                order: 0,
                status: ChatTreeNodeStatus::Completed,
                summary: Some("A".to_string()),
            }],
        });

        state.apply_event(&EventMsg::ChatTreeCurrentNodeChanged(Box::new(
            ChatTreeCurrentNodeChangedEvent {
                revision: 2,
                node_id: "node-a".to_string(),
            },
        )));

        assert_eq!(
            state.projection().visible_turn_ids,
            vec!["legacy-turn".to_string(), "turn-a".to_string()]
        );
    }

    #[test]
    fn replay_event_does_not_rewind_seeded_projection() {
        let mut state = ChatTreeUiState::default();
        state.set_projection(ChatTreeProjection {
            version: 1,
            revision: 2,
            current_node_id: Some("node-b".to_string()),
            visible_node_ids: vec!["node-a".to_string(), "node-b".to_string()],
            visible_turn_ids: vec!["turn-a".to_string(), "turn-b".to_string()],
            nodes: vec![
                ChatTreeNode {
                    node_id: "node-a".to_string(),
                    parent_node_id: None,
                    turn_id: Some("turn-a".to_string()),
                    order: 0,
                    status: ChatTreeNodeStatus::Completed,
                    summary: Some("A".to_string()),
                },
                ChatTreeNode {
                    node_id: "node-b".to_string(),
                    parent_node_id: Some("node-a".to_string()),
                    turn_id: Some("turn-b".to_string()),
                    order: 1,
                    status: ChatTreeNodeStatus::Pending,
                    summary: None,
                },
            ],
        });
        let seeded = state.projection().clone();

        state.apply_replay_event(&EventMsg::ChatTreeCurrentNodeChanged(Box::new(
            ChatTreeCurrentNodeChangedEvent {
                revision: 1,
                node_id: "node-a".to_string(),
            },
        )));

        assert_eq!(state.projection(), &seeded);
    }

    #[test]
    fn replay_event_builds_projection_when_unseeded() {
        let mut state = ChatTreeUiState::default();

        state.apply_replay_event(&EventMsg::ChatTreeNodeStarted(Box::new(
            ChatTreeNodeStartedEvent {
                revision: 1,
                node_id: "node-a".to_string(),
                parent_node_id: None,
                turn_id: Some("turn-a".to_string()),
                order: 0,
            },
        )));

        assert_eq!(
            state.projection().visible_turn_ids,
            vec!["turn-a".to_string()]
        );
    }
}
