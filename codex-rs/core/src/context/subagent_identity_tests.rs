use super::*;
use codex_protocol::AgentPath;
use codex_protocol::ThreadId;
use codex_protocol::protocol::SessionSource;
use codex_protocol::protocol::SubAgentSource;

fn high_entropy_task_name() -> String {
    const ALPHABET: &[u8] = b"abcdefghijklmnopqrstuvwxyz0123456789_";
    (0..AgentPath::MAX_AGENT_NAME_BYTES)
        .map(|index| char::from(ALPHABET[index.wrapping_mul(17) % ALPHABET.len()]))
        .collect()
}

fn near_max_agent_path() -> AgentPath {
    let task_name = high_entropy_task_name();
    let mut path = AgentPath::root();
    while let Ok(next) = path.join(&task_name) {
        path = next;
    }
    assert!(path.as_str().len() > AgentPath::MAX_PATH_BYTES - AgentPath::MAX_AGENT_NAME_BYTES - 1);
    path
}

fn thread_spawn_source(
    agent_path: Option<AgentPath>,
    nickname: String,
    role: String,
) -> SessionSource {
    SessionSource::SubAgent(SubAgentSource::ThreadSpawn {
        parent_thread_id: ThreadId::new(),
        depth: i32::MAX,
        agent_path,
        agent_nickname: Some(nickname),
        agent_role: Some(role),
    })
}

#[test]
fn legal_identity_facts_render_exactly_with_a_hard_byte_bound() {
    let agent_path = near_max_agent_path();
    let nickname = "\u{754c}".repeat(IDENTITY_LABEL_MAX_BYTES / "\u{754c}".len());
    let role = "\u{1f9ea}".repeat(IDENTITY_LABEL_MAX_BYTES / "\u{1f9ea}".len());
    assert!(nickname.len() <= IDENTITY_LABEL_MAX_BYTES);
    assert_eq!(role.len(), IDENTITY_LABEL_MAX_BYTES);

    let identity = AgentIdentity::from_session_source(&thread_spawn_source(
        Some(agent_path.clone()),
        nickname.clone(),
        role.clone(),
    ));
    let current_agent_name = identity.current_agent_name(ThreadId::new());
    let startup = SubagentIdentity::new(identity.clone()).render();
    let fork_end = ForkedHistoryBoundary::end(identity).render();
    let expected_path = format!("current canonical agent path is `{current_agent_name}`");

    assert_eq!(current_agent_name, agent_path.as_str());
    for fragment in [&startup, &fork_end] {
        assert!(fragment.contains(&expected_path));
        assert!(fragment.contains(&format!("nickname is `{nickname}`")));
        assert!(fragment.contains(&format!("configured role is `{role}`")));
        assert!(fragment.len() <= IDENTITY_FRAGMENT_MAX_BYTES);
    }
}

#[test]
fn oversized_legacy_labels_are_explicitly_omitted() {
    let oversized_nickname = "\u{754c}".repeat(IDENTITY_LABEL_MAX_BYTES / "\u{754c}".len() + 1);
    let oversized_role = "\u{1f9ea}".repeat(IDENTITY_LABEL_MAX_BYTES / "\u{1f9ea}".len() + 1);
    let identity = AgentIdentity::from_session_source(&thread_spawn_source(
        Some(AgentPath::root().join("worker").expect("valid path")),
        oversized_nickname.clone(),
        oversized_role.clone(),
    ));

    for fragment in [
        SubagentIdentity::new(identity.clone()).render(),
        ForkedHistoryBoundary::end(identity).render(),
    ] {
        assert!(
            fragment.contains(
                "nickname was omitted because it exceeds the identity-context safety limit"
            )
        );
        assert!(fragment.contains(
            "configured role was omitted because it exceeds the identity-context safety limit"
        ));
        assert!(!fragment.contains(&oversized_nickname));
        assert!(!fragment.contains(&oversized_role));
        assert!(fragment.len() <= IDENTITY_FRAGMENT_MAX_BYTES);
    }
}
