use super::session_lifecycle_requests::recorded_params;
use super::session_lifecycle_requests::start_recording_app_server;
use super::*;
use codex_app_server_protocol::ChatTreeProjection;

#[tokio::test]
async fn chat_tree_switch_refresh_replaces_visible_transcript() -> Result<()> {
    let (mut app, mut events, _operations) = make_test_app_with_channels().await;
    let codex_home = tempdir()?;
    app.config.codex_home = codex_home.path().to_path_buf().abs();
    let thread_id = ThreadId::from_string(
        &app_test_support::create_fake_rollout(
            codex_home.path(),
            "2026-01-02T00-00-00",
            "2026-01-02T00:00:00Z",
            "selected branch prompt",
            Some(app.config.model_provider_id.as_str()),
            /*git_info*/ None,
        )
        .expect("selected branch rollout"),
    )?;
    let (mut app_server, requests, proxy) = start_recording_app_server(
        &app.config,
        /*blocked_thread_list*/ None,
        /*failed_thread_name*/ None,
    )
    .await?;
    let resumed = app_server
        .resume_thread(
            app.config.clone(),
            thread_id,
            crate::app_server_session::ResumeModelSettings::RestoreFromThread,
        )
        .await?;
    app.primary_thread_id = Some(thread_id);
    app.active_thread_id = Some(thread_id);
    app.chat_widget
        .handle_thread_session_quiet(resumed.session.clone());
    app.thread_event_channels.insert(
        thread_id,
        ThreadEventChannel::new_with_session(
            THREAD_EVENT_CHANNEL_CAPACITY,
            resumed.session,
            resumed.turns,
        ),
    );
    app.transcript_cells = vec![Arc::new(PlainHistoryCell::new(vec![
        "stale sibling transcript".into(),
    ]))];
    while events.try_recv().is_ok() {}

    let mut tui = crate::tui::test_support::make_test_tui()?;
    app.handle_event(
        &mut tui,
        &mut app_server,
        AppEvent::RefreshChatTreeTranscript {
            thread_id,
            chat_tree: ChatTreeProjection {
                version: 1,
                revision: 1,
                current_node_id: None,
                visible_node_ids: Vec::new(),
                visible_turn_ids: Vec::new(),
                nodes: Vec::new(),
            },
        },
    )
    .await?;
    while let Ok(event) = events.try_recv() {
        if matches!(event, AppEvent::InsertHistoryCell(_)) {
            app.handle_event(&mut tui, &mut app_server, event).await?;
        }
    }

    let transcript = app
        .transcript_cells
        .iter()
        .flat_map(|cell| cell.display_lines(/*width*/ 100))
        .map(|line| line.to_string())
        .collect::<Vec<_>>()
        .join("\n");
    assert!(
        transcript.contains("selected branch prompt"),
        "{transcript}"
    );
    assert!(
        !transcript.contains("stale sibling transcript"),
        "{transcript}"
    );
    assert!(
        recorded_params(&requests, "thread/read")
            .iter()
            .any(|params| params["threadId"] == thread_id.to_string()
                && params["includeTurns"] == true)
    );

    app_server.shutdown().await?;
    proxy.await??;
    Ok(())
}
