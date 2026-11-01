//! Startup identity context for spawned subagents, enabled by `Feature::SubagentIdentity`.

use super::*;

/// Starts a session that runs as the given thread-spawned subagent with identity enabled.
async fn subagent_identity_initial_context(
    enable_multi_agent_v2: bool,
    session_source: SessionSource,
) -> Vec<Vec<String>> {
    let (session, mut turn_context) =
        make_multi_agent_v2_usage_hint_test_session(enable_multi_agent_v2).await;
    session
        .state
        .lock()
        .await
        .session_configuration
        .session_source = session_source.clone();
    let turn_context_mut =
        Arc::get_mut(&mut turn_context).expect("thread settings should not be shared");
    turn_context_mut.session_source = session_source;
    Arc::make_mut(&mut turn_context_mut.config)
        .features
        .enable(Feature::SubagentIdentity)
        .expect("test config should allow subagent identity");

    let initial_context = build_initial_context(&session, &turn_context).await;
    developer_message_texts(&initial_context)
        .into_iter()
        .map(|message| message.into_iter().map(str::to_string).collect())
        .collect()
}

#[tokio::test]
async fn build_initial_context_adds_subagent_identity_without_fork_wording() {
    let developer_messages = subagent_identity_initial_context(
        /*enable_multi_agent_v2*/ true,
        SessionSource::SubAgent(SubAgentSource::ThreadSpawn {
            parent_thread_id: ThreadId::new(),
            depth: 1,
            agent_path: Some(AgentPath::try_from("/root/worker").expect("agent path should parse")),
            agent_nickname: Some("worker".to_string()),
            agent_role: None,
        }),
    )
    .await;

    let identity = developer_messages
        .iter()
        .flatten()
        .find(|text| text.contains("Your current canonical agent path is `/root/worker`"))
        .expect("subagent identity developer message should be present");
    assert!(identity.contains("your parent thread id is"));
    assert!(identity.contains("nickname is `worker`"));
    assert!(
        !developer_messages
            .iter()
            .flatten()
            .any(|text| text.contains("forked parent conversation history")),
        "ordinary startup must not use fork-history wording, got {developer_messages:?}"
    );
}

#[tokio::test]
async fn build_initial_context_marks_pathless_subagent_as_not_root() {
    let parent_thread_id = ThreadId::new();
    let developer_messages = subagent_identity_initial_context(
        /*enable_multi_agent_v2*/ false,
        SessionSource::SubAgent(SubAgentSource::ThreadSpawn {
            parent_thread_id,
            depth: 2,
            agent_path: None,
            agent_nickname: Some("worker".to_string()),
            agent_role: Some("researcher".to_string()),
        }),
    )
    .await;

    let identity = developer_messages
        .iter()
        .flatten()
        .find(|text| text.contains("You do not have a canonical agent path"))
        .expect("pathless subagent identity should be present");
    assert!(identity.contains("still not the `/root` main agent"));
    assert!(identity.contains(&parent_thread_id.to_string()));
    assert!(identity.contains("depth is 2"));
    assert!(identity.contains("nickname is `worker`"));
    assert!(identity.contains("configured role is `researcher`"));
}
