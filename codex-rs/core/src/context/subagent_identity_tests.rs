use super::*;
use codex_protocol::AgentPath;
use codex_protocol::ThreadId;
use codex_protocol::protocol::SessionSource;
use codex_protocol::protocol::SubAgentSource;
use codex_utils_string::approx_bytes_for_tokens;
use codex_utils_string::approx_token_count;

const MODEL_CONTEXT_ITEM_LIMIT_TOKENS: usize = 10_000;

#[test]
fn identity_fragments_bound_oversized_path_nickname_and_role() {
    let oversized = "a".repeat(approx_bytes_for_tokens(
        MODEL_CONTEXT_ITEM_LIMIT_TOKENS + 2_000,
    ));
    let sources = [
        (
            SessionSource::SubAgent(SubAgentSource::ThreadSpawn {
                parent_thread_id: ThreadId::new(),
                depth: 1,
                agent_path: Some(
                    AgentPath::root()
                        .join(&oversized)
                        .expect("oversized task name should remain schema-valid"),
                ),
                agent_nickname: Some(oversized.clone()),
                agent_role: Some(oversized.clone()),
            }),
            "Your current canonical agent path is",
        ),
        (
            SessionSource::SubAgent(SubAgentSource::ThreadSpawn {
                parent_thread_id: ThreadId::new(),
                depth: 2,
                agent_path: None,
                agent_nickname: Some(oversized.clone()),
                agent_role: Some(oversized.clone()),
            }),
            "You do not have a canonical agent path",
        ),
    ];

    for (source, expected_path_text) in sources {
        let identity = AgentIdentity::from_session_source(&source);
        let startup = SubagentIdentity::new(identity.clone()).render();
        let fork_end = ForkedHistoryBoundary::end(identity).render();

        assert!(startup.contains(expected_path_text));
        for fragment in [&startup, &fork_end] {
            assert!(
                approx_token_count(fragment) < MODEL_CONTEXT_ITEM_LIMIT_TOKENS,
                "identity fragment must stay below the per-item model-context limit"
            );
            assert!(fragment.contains("nickname is"));
            assert!(fragment.contains("configured role is"));
            assert!(!fragment.contains(&oversized));
        }
    }
}
