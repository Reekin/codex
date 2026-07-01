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
    let (mut text, assignment_text) = identity_hint_parts(session_source)?;
    text.push_str(" Any prior transcript inherited from another agent is context, not proof that you performed those actions; tool calls, spawned agents, waits, and decisions in inherited history may belong to your parent.");
    text.push_str(assignment_text);
    text.push_str(", while still following all system, developer, and user instructions.");
    Some(text)
}

pub(crate) fn forked_history_start_hint_text(session_source: &SessionSource) -> Option<String> {
    identity_hint_parts(session_source)?;
    Some(
        "You are a spawned subagent forked from a parent agent conversation. The following messages are inherited parent conversation history. Use them only as context for the task you receive after the closing fork-history boundary.\n\n==== forked parent conversation history begins ====".to_string(),
    )
}

pub(crate) fn forked_history_boundary_hint_text(session_source: &SessionSource) -> Option<String> {
    let (mut text, _) = identity_hint_parts(session_source)?;
    text.insert_str(
        0,
        "==== forked parent conversation history ends ====\n\nThe inherited parent conversation history above is present so you can understand the context. It is not the current task you should answer directly. ",
    );
    text.push_str(" Tool calls, spawned agents, waits, and decisions in the inherited history may belong to your parent. Treat the next task message sent directly to you as your assignment. Use the inherited history only as context for completing that assignment, while still following all system, developer, and user instructions.");
    Some(text)
}

fn identity_hint_parts(session_source: &SessionSource) -> Option<(String, &'static str)> {
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

    let (mut text, assignment_text) = match agent_path.clone() {
        Some(agent_path) => (
            format!(
                "You are a spawned subagent in the multi-agent tree. Your current canonical agent path is `{agent_path}`, your parent thread id is `{parent_thread_id}`, and your depth is {depth}. You are not `/root` unless your current canonical agent path is exactly `/root`. `list_agents` may show `/root`, sibling agents, and child agents; those entries are other agents unless their `agent_name` equals your current canonical agent path or `is_current_agent` is true."
            ),
            " Treat the latest inter-agent communication addressed to your current canonical agent path as your assigned work",
        ),
        None => (
            "You are a spawned subagent in the multi-agent tree. You do not have a canonical agent path in this session, but you are still not the `/root` main agent.".to_string(),
            " Treat the latest task sent directly to this subagent as your assigned work",
        ),
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

    Some((text, assignment_text))
}
