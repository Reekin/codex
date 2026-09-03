use codex_protocol::AgentPath;
use codex_protocol::ThreadId;
use codex_protocol::protocol::SessionSource;
use codex_protocol::protocol::SubAgentSource;

/// Authoritative identity facts for the agent that owns a session.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct AgentIdentity {
    canonical_path: Option<AgentPath>,
    kind: AgentIdentityKind,
}

#[derive(Clone, Debug, PartialEq, Eq)]
enum AgentIdentityKind {
    Root,
    ThreadSpawn {
        parent_thread_id: ThreadId,
        depth: i32,
        nickname: Option<String>,
        role: Option<String>,
    },
    OtherNonRoot,
}

pub(crate) struct ThreadSpawnIdentity<'a> {
    pub(crate) canonical_path: Option<&'a AgentPath>,
    pub(crate) parent_thread_id: ThreadId,
    pub(crate) depth: i32,
    pub(crate) nickname: Option<&'a str>,
    pub(crate) role: Option<&'a str>,
}

impl AgentIdentity {
    pub(crate) fn from_session_source(session_source: &SessionSource) -> Self {
        match session_source {
            SessionSource::SubAgent(SubAgentSource::ThreadSpawn {
                parent_thread_id,
                depth,
                agent_path,
                agent_nickname,
                agent_role,
            }) => Self {
                canonical_path: agent_path.clone(),
                kind: AgentIdentityKind::ThreadSpawn {
                    parent_thread_id: *parent_thread_id,
                    depth: *depth,
                    nickname: agent_nickname.clone(),
                    role: agent_role.clone(),
                },
            },
            SessionSource::Cli
            | SessionSource::VSCode
            | SessionSource::Exec
            | SessionSource::Mcp
            | SessionSource::Custom(_)
            | SessionSource::Unknown => Self {
                canonical_path: Some(AgentPath::root()),
                kind: AgentIdentityKind::Root,
            },
            SessionSource::Internal(_) | SessionSource::SubAgent(_) => Self {
                canonical_path: None,
                kind: AgentIdentityKind::OtherNonRoot,
            },
        }
    }

    pub(crate) fn current_agent_name(&self, thread_id: ThreadId) -> String {
        self.canonical_path
            .as_ref()
            .map(ToString::to_string)
            .unwrap_or_else(|| thread_id.to_string())
    }

    pub(crate) fn thread_spawn(&self) -> Option<ThreadSpawnIdentity<'_>> {
        let AgentIdentityKind::ThreadSpawn {
            parent_thread_id,
            depth,
            nickname,
            role,
        } = &self.kind
        else {
            return None;
        };
        Some(ThreadSpawnIdentity {
            canonical_path: self.canonical_path.as_ref(),
            parent_thread_id: *parent_thread_id,
            depth: *depth,
            nickname: nickname.as_deref(),
            role: role.as_deref(),
        })
    }
}
