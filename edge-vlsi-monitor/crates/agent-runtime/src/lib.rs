//! Agent Runtime
//!
//! Manages inference agent lifecycles on edge devices. Each agent has:
//!   - A state machine (Idle -> Loading -> Running -> Paused -> Terminated)
//!   - Optional ONNX model for real inference (via tract)
//!   - VLSI hardware simulation engine
//!   - Per-agent metrics collection (latency, power, thermal)
//!
//! The scheduler coordinates multiple agents via async message passing.

pub mod agent;
pub mod inference;
pub mod scheduler;
pub mod state;

pub use agent::{Agent, AgentInfo};
pub use inference::{InferenceResult, OnnxModel};
pub use scheduler::{Scheduler, SchedulerCommand, SchedulerHandle};
pub use state::{AgentState, StateMachine, TransitionError};
