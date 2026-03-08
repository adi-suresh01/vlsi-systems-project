/// Agent lifecycle state machine.
///
/// Enforces valid state transitions at compile-time via the `transition()` method.
/// Invalid transitions return `TransitionError`.
///
/// Valid transitions:
///   Idle -> Loading
///   Loading -> Running | Terminated
///   Running -> Paused | Terminated
///   Paused -> Running | Terminated

use std::time::Instant;

/// Possible agent states.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum AgentState {
    Idle,
    Loading,
    Running,
    Paused,
    Terminated,
}

impl std::fmt::Display for AgentState {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            AgentState::Idle => write!(f, "idle"),
            AgentState::Loading => write!(f, "loading"),
            AgentState::Running => write!(f, "running"),
            AgentState::Paused => write!(f, "paused"),
            AgentState::Terminated => write!(f, "terminated"),
        }
    }
}

#[derive(Debug, thiserror::Error)]
pub enum TransitionError {
    #[error("invalid state transition: {from} -> {to}")]
    InvalidTransition { from: AgentState, to: AgentState },

    #[error("agent is terminated and cannot transition")]
    AlreadyTerminated,
}

/// State machine tracking current state and transition history.
pub struct StateMachine {
    current: AgentState,
    entered_at: Instant,
    history: Vec<(AgentState, Instant)>,
}

impl StateMachine {
    pub fn new() -> Self {
        let now = Instant::now();
        Self {
            current: AgentState::Idle,
            entered_at: now,
            history: vec![(AgentState::Idle, now)],
        }
    }

    pub fn current(&self) -> AgentState {
        self.current
    }

    /// Attempt a state transition. Returns the new state on success.
    pub fn transition(&mut self, to: AgentState) -> Result<AgentState, TransitionError> {
        if self.current == AgentState::Terminated {
            return Err(TransitionError::AlreadyTerminated);
        }

        if !Self::is_valid_transition(self.current, to) {
            return Err(TransitionError::InvalidTransition {
                from: self.current,
                to,
            });
        }

        let now = Instant::now();
        self.current = to;
        self.entered_at = now;
        self.history.push((to, now));

        Ok(to)
    }

    /// Duration spent in the current state.
    pub fn time_in_state(&self) -> std::time::Duration {
        self.entered_at.elapsed()
    }

    pub fn history(&self) -> &[(AgentState, Instant)] {
        &self.history
    }

    fn is_valid_transition(from: AgentState, to: AgentState) -> bool {
        matches!(
            (from, to),
            (AgentState::Idle, AgentState::Loading)
                | (AgentState::Loading, AgentState::Running)
                | (AgentState::Loading, AgentState::Terminated)
                | (AgentState::Running, AgentState::Paused)
                | (AgentState::Running, AgentState::Terminated)
                | (AgentState::Paused, AgentState::Running)
                | (AgentState::Paused, AgentState::Terminated)
        )
    }
}

impl Default for StateMachine {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_initial_state() {
        let sm = StateMachine::new();
        assert_eq!(sm.current(), AgentState::Idle);
    }

    #[test]
    fn test_valid_lifecycle() {
        let mut sm = StateMachine::new();
        assert!(sm.transition(AgentState::Loading).is_ok());
        assert!(sm.transition(AgentState::Running).is_ok());
        assert!(sm.transition(AgentState::Paused).is_ok());
        assert!(sm.transition(AgentState::Running).is_ok());
        assert!(sm.transition(AgentState::Terminated).is_ok());
    }

    #[test]
    fn test_invalid_transition() {
        let mut sm = StateMachine::new();
        // Idle -> Running is invalid (must go through Loading)
        assert!(sm.transition(AgentState::Running).is_err());
    }

    #[test]
    fn test_terminated_is_final() {
        let mut sm = StateMachine::new();
        sm.transition(AgentState::Loading).unwrap();
        sm.transition(AgentState::Terminated).unwrap();
        // Cannot transition from Terminated
        assert!(sm.transition(AgentState::Idle).is_err());
    }

    #[test]
    fn test_history_tracking() {
        let mut sm = StateMachine::new();
        sm.transition(AgentState::Loading).unwrap();
        sm.transition(AgentState::Running).unwrap();
        assert_eq!(sm.history().len(), 3); // Idle, Loading, Running
    }
}
