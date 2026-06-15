use crate::config::MultiAgentV2Config;
use crate::session::turn_context::TurnContext;
use codex_features::Feature;
use codex_protocol::AgentPath;
use codex_protocol::config_types::MultiAgentMode;
use codex_protocol::openai_models::ReasoningEffort;
use codex_protocol::protocol::MultiAgentVersion;
use codex_protocol::protocol::SessionSource;
use codex_protocol::protocol::SubAgentSource;

pub(super) fn usage_hint_text<'a>(
    turn_context: &'a TurnContext,
    session_source: &SessionSource,
) -> Option<&'a str> {
    if turn_context.multi_agent_version != MultiAgentVersion::V2 {
        return None;
    }

    let multi_agent_v2 = &turn_context.config.multi_agent_v2;
    configured_usage_hint_text_for_source(multi_agent_v2, session_source)
}

fn configured_usage_hint_text_for_source<'a>(
    multi_agent_v2: &'a MultiAgentV2Config,
    session_source: &SessionSource,
) -> Option<&'a str> {
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

pub(crate) fn effective_multi_agent_mode(turn_context: &TurnContext) -> Option<MultiAgentMode> {
    if turn_context.multi_agent_version != MultiAgentVersion::V2 {
        return None;
    }

    let multi_agent_mode = match turn_context.effective_reasoning_effort() {
        Some(ReasoningEffort::Ultra) => MultiAgentMode::Proactive,
        _ => MultiAgentMode::ExplicitRequestOnly,
    };

    match &turn_context.session_source {
        SessionSource::SubAgent(SubAgentSource::ThreadSpawn { .. })
        | SessionSource::Cli
        | SessionSource::VSCode
        | SessionSource::Exec
        | SessionSource::Mcp
        | SessionSource::Custom(_)
        | SessionSource::Unknown => Some(multi_agent_mode),
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
