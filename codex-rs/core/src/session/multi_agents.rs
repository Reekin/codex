use crate::session::turn_context::TurnContext;
use codex_features::Feature;
use codex_protocol::protocol::SessionSource;
use codex_protocol::protocol::SubAgentSource;

pub(super) fn usage_hint_text<'a>(
    turn_context: &'a TurnContext,
    session_source: &SessionSource,
) -> Option<&'a str> {
    if !turn_context.features.enabled(Feature::MultiAgentV2) {
        return None;
    }

    let multi_agent_v2 = &turn_context.config.multi_agent_v2;
    match session_source {
        SessionSource::SubAgent(SubAgentSource::ThreadSpawn { .. }) => {
            multi_agent_v2.subagent_usage_hint_text.as_deref()
        }
        SessionSource::Cli
        | SessionSource::VSCode
        | SessionSource::Exec
        | SessionSource::Mcp
        | SessionSource::Custom(_)
        | SessionSource::Unknown => multi_agent_v2.root_agent_usage_hint_text.as_deref(),
        SessionSource::Internal(_) | SessionSource::SubAgent(_) => None,
    }
}

pub(crate) fn identity_hint_text(session_source: &SessionSource) -> Option<String> {
    let SessionSource::SubAgent(SubAgentSource::ThreadSpawn {
        parent_thread_id,
        depth,
        agent_path,
        agent_nickname,
        agent_role,
    }) = session_source
    else {
        return None;
    };

    let mut text = match agent_path.clone() {
        Some(agent_path) => format!(
            "You are a spawned subagent in the multi-agent tree. Your current canonical agent path is `{agent_path}`, your parent thread id is `{parent_thread_id}`, and your depth is {depth}. You are not `/root` unless your current canonical agent path is exactly `/root`. Any prior transcript inherited from another agent is context, not proof that you performed those actions; tool calls, spawned agents, waits, and decisions in inherited history may belong to your parent. `list_agents` may show `/root`, sibling agents, and child agents; those entries are other agents unless their `agent_name` equals your current canonical agent path or `is_current_agent` is true. Treat the latest inter-agent communication addressed to `{agent_path}` as your assigned work, while still following all system, developer, and user instructions."
        ),
        None => "You are a spawned subagent in the multi-agent tree. You do not have a canonical agent path in this session, but you are still not the `/root` main agent. Any prior transcript inherited from another agent is context, not proof that you performed those actions; tool calls, spawned agents, waits, and decisions in inherited history may belong to your parent. Treat the latest task sent directly to this subagent as your assigned work, while still following all system, developer, and user instructions.".to_string(),
    };

    if let Some(nickname) = agent_nickname
        .as_deref()
        .filter(|nickname| !nickname.is_empty())
    {
        text.push_str(&format!(" Your nickname is `{nickname}`."));
    }
    if let Some(role) = agent_role.as_deref().filter(|role| !role.is_empty()) {
        text.push_str(&format!(" Your configured role is `{role}`."));
    }

    Some(text)
}
