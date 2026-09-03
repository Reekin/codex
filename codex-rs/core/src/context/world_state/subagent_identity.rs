use super::PreviousSectionState;
use super::WorldStateHash;
use super::WorldStateSection;
use crate::agent::AgentIdentity;
use crate::context::ContextualUserFragment;
use crate::context::SubagentIdentity;

/// Model-visible identity for the spawned subagent that owns the current session.
#[derive(Clone, Debug)]
pub(crate) struct SubagentIdentityState {
    identity: SubagentIdentity,
}

impl SubagentIdentityState {
    pub(crate) fn new(identity: AgentIdentity) -> Self {
        Self {
            identity: SubagentIdentity::new(identity),
        }
    }
}

impl WorldStateSection for SubagentIdentityState {
    const ID: &'static str = "subagent_identity";
    type Snapshot = WorldStateHash;

    fn snapshot(&self) -> Self::Snapshot {
        WorldStateHash::from_fragment(&self.identity)
    }

    fn matches_current_legacy_fragment(&self, role: &str, text: &str) -> bool {
        role == self.identity.role() && text == self.identity.render()
    }

    fn render_diff(
        &self,
        previous: PreviousSectionState<'_, Self::Snapshot>,
    ) -> Option<Box<dyn ContextualUserFragment>> {
        match previous {
            PreviousSectionState::Known(previous) if previous == &self.snapshot() => None,
            PreviousSectionState::Unknown => None,
            PreviousSectionState::Known(_) | PreviousSectionState::Absent => {
                Some(Box::new(self.identity.clone()))
            }
        }
    }
}
