/// Agent scheduler — manages multiple agents via async command processing.
///
/// Replaces the monolithic pipeline_controller.py with a concurrent,
/// message-driven architecture using tokio channels.

use std::collections::HashMap;
use std::path::PathBuf;

use ndarray::Array2;
use tokio::sync::mpsc;
use uuid::Uuid;

use crate::agent::{Agent, AgentInfo};
use vlsi_sim::SimulationResult;

/// Commands the scheduler accepts.
#[derive(Debug)]
pub enum SchedulerCommand {
    /// Spawn a new agent in simulation mode
    SpawnAgent {
        name: String,
        respond: mpsc::Sender<AgentInfo>,
    },
    /// Spawn an agent with an ONNX model
    SpawnAgentWithModel {
        name: String,
        model_path: PathBuf,
        respond: mpsc::Sender<Result<AgentInfo, String>>,
    },
    /// Pause an agent
    PauseAgent {
        id: Uuid,
        respond: mpsc::Sender<Result<(), String>>,
    },
    /// Resume a paused agent
    ResumeAgent {
        id: Uuid,
        respond: mpsc::Sender<Result<(), String>>,
    },
    /// Terminate an agent
    TerminateAgent {
        id: Uuid,
        respond: mpsc::Sender<Result<(), String>>,
    },
    /// Run hardware simulation on an agent
    RunSimulation {
        id: Uuid,
        samples: Vec<Array2<f64>>,
        respond: mpsc::Sender<Result<SimulationResult, String>>,
    },
    /// List all agents
    ListAgents {
        respond: mpsc::Sender<Vec<AgentInfo>>,
    },
    /// Get info for a specific agent
    GetAgent {
        id: Uuid,
        respond: mpsc::Sender<Option<AgentInfo>>,
    },
    /// Shut down the scheduler
    Shutdown,
}

/// The scheduler manages agent lifecycles.
pub struct Scheduler {
    agents: HashMap<Uuid, Agent>,
    cmd_rx: mpsc::Receiver<SchedulerCommand>,
}

/// Handle used to send commands to the scheduler.
pub type SchedulerHandle = mpsc::Sender<SchedulerCommand>;

impl Scheduler {
    /// Create a new scheduler and return it along with a command handle.
    pub fn new(buffer_size: usize) -> (Self, SchedulerHandle) {
        let (tx, rx) = mpsc::channel(buffer_size);
        (
            Self {
                agents: HashMap::new(),
                cmd_rx: rx,
            },
            tx,
        )
    }

    /// Run the scheduler's main event loop.
    pub async fn run(&mut self) {
        tracing::info!("Scheduler started");

        while let Some(cmd) = self.cmd_rx.recv().await {
            match cmd {
                SchedulerCommand::SpawnAgent { name, respond } => {
                    let mut agent = Agent::new(name);
                    let _ = agent.start_simulation_mode();
                    let info = agent.info();
                    self.agents.insert(agent.id, agent);
                    let _ = respond.send(info).await;
                }

                SchedulerCommand::SpawnAgentWithModel {
                    name,
                    model_path,
                    respond,
                } => {
                    let mut agent = Agent::new(name);
                    match agent.load_model(&model_path) {
                        Ok(()) => {
                            let info = agent.info();
                            self.agents.insert(agent.id, agent);
                            let _ = respond.send(Ok(info)).await;
                        }
                        Err(e) => {
                            let _ = respond.send(Err(e.to_string())).await;
                        }
                    }
                }

                SchedulerCommand::PauseAgent { id, respond } => {
                    let result = match self.agents.get_mut(&id) {
                        Some(agent) => agent.pause().map_err(|e| e.to_string()),
                        None => Err("Agent not found".into()),
                    };
                    let _ = respond.send(result).await;
                }

                SchedulerCommand::ResumeAgent { id, respond } => {
                    let result = match self.agents.get_mut(&id) {
                        Some(agent) => agent.resume().map_err(|e| e.to_string()),
                        None => Err("Agent not found".into()),
                    };
                    let _ = respond.send(result).await;
                }

                SchedulerCommand::TerminateAgent { id, respond } => {
                    let result = match self.agents.get_mut(&id) {
                        Some(agent) => agent.terminate().map_err(|e| e.to_string()),
                        None => Err("Agent not found".into()),
                    };
                    let _ = respond.send(result).await;
                }

                SchedulerCommand::RunSimulation {
                    id,
                    samples,
                    respond,
                } => {
                    let result = match self.agents.get_mut(&id) {
                        Some(agent) => Ok(agent.run_hw_simulation(&samples)),
                        None => Err("Agent not found".into()),
                    };
                    let _ = respond.send(result).await;
                }

                SchedulerCommand::ListAgents { respond } => {
                    let infos: Vec<AgentInfo> =
                        self.agents.values().map(|a| a.info()).collect();
                    let _ = respond.send(infos).await;
                }

                SchedulerCommand::GetAgent { id, respond } => {
                    let info = self.agents.get(&id).map(|a| a.info());
                    let _ = respond.send(info).await;
                }

                SchedulerCommand::Shutdown => {
                    tracing::info!("Scheduler shutting down");
                    // Terminate all agents
                    for agent in self.agents.values_mut() {
                        let _ = agent.terminate();
                    }
                    break;
                }
            }
        }

        tracing::info!("Scheduler stopped");
    }

    pub fn agent_count(&self) -> usize {
        self.agents.len()
    }
}
