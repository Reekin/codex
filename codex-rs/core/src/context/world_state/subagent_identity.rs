use super::PreviousSectionState;
use super::SectionTransition;
use super::WorldStateHash;
use super::WorldStateSection;
use super::WorldStateUpdate;
use crate::agent::AgentIdentity;
use crate::context::ContextualUserFragment;
use crate::context::SubagentIdentity;

/// Model-visible identity for the spawned subagent that owns the current session.
#[derive(Clone, Debug)]
pub(crate) struct SubagentIdentityState {
    identity: SubagentIdentity,
    fingerprint: WorldStateHash,
}

impl SubagentIdentityState {
    pub(crate) fn new(identity: AgentIdentity) -> Self {
        let identity = SubagentIdentity::new(identity);
        let fingerprint = WorldStateHash::from_fragment(&identity);
        Self {
            identity,
            fingerprint,
        }
    }
}

impl WorldStateSection for SubagentIdentityState {
    const ID: &'static str = "subagent_identity";
    type Snapshot = WorldStateHash;

    fn matches_current_legacy_fragment(&self, role: &str, text: &str) -> bool {
        role == self.identity.role() && self.identity.render().into_text().as_deref() == Some(text)
    }

    fn render_diff(
        &self,
        previous: PreviousSectionState<'_, Self::Snapshot>,
    ) -> SectionTransition<Self::Snapshot> {
        let fragment: Option<Box<dyn ContextualUserFragment>> = match previous {
            PreviousSectionState::Known(previous) if previous == &self.fingerprint => {
                return (None, Vec::new());
            }
            PreviousSectionState::Unknown => None,
            PreviousSectionState::Known(_) | PreviousSectionState::Absent => {
                Some(Box::new(self.identity.clone()))
            }
        };
        (
            Some(self.fingerprint.clone()),
            WorldStateUpdate::optional_standalone_boxed_fragment(fragment),
        )
    }
}
