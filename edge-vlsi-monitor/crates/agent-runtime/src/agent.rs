/// Individual inference agent.
///
/// Ties together:
///   - State machine (lifecycle management)
///   - ONNX model inference (via tract)
///   - VLSI hardware simulation (via vlsi-sim)
///   - Metrics collection (latency, power, thermal)

use std::path::Path;
use std::time::Instant;

use anyhow::Result;
use chrono::Utc;
use ndarray::Array2;
use uuid::Uuid;

use crate::inference::{InferenceResult, OnnxModel};
use crate::state::{AgentState, StateMachine, TransitionError};
use metrics_collector::{AgentMetrics, PowerSample};
use vlsi_sim::{
    ConvConfig, ConvEngine, PipelineConfig, PowerModel,
    SimulationResult, ThermalModel,
};

/// Serializable agent information for the dashboard.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct AgentInfo {
    pub id: String,
    pub name: String,
    pub state: AgentState,
    pub inference_count: u64,
    pub total_mac_ops: u64,
    pub total_relu_ops: u64,
    pub avg_latency_us: f64,
    pub p99_latency_us: u64,
    pub power_w: f64,
    pub temperature_c: f64,
    pub throughput_samples_per_sec: f64,
}

/// A single inference agent with its own state, model, and metrics.
pub struct Agent {
    pub id: Uuid,
    pub name: String,
    state: StateMachine,
    model: Option<OnnxModel>,
    conv_engine: ConvEngine,
    power_model: PowerModel,
    thermal: ThermalModel,
    pub metrics: AgentMetrics,
    total_mac_ops: u64,
    total_relu_ops: u64,
    last_sim_result: Option<SimulationResult>,
}

impl Agent {
    pub fn new(name: String) -> Self {
        Self {
            id: Uuid::new_v4(),
            name,
            state: StateMachine::new(),
            model: None,
            conv_engine: ConvEngine::new(ConvConfig::default()),
            power_model: PowerModel::default(),
            thermal: ThermalModel::default(),
            metrics: AgentMetrics::new(),
            total_mac_ops: 0,
            total_relu_ops: 0,
            last_sim_result: None,
        }
    }

    /// Load an ONNX model for this agent.
    pub fn load_model(&mut self, path: &Path) -> Result<()> {
        self.state.transition(AgentState::Loading)?;
        match OnnxModel::load(path) {
            Ok(model) => {
                self.model = Some(model);
                self.state.transition(AgentState::Running)?;
                Ok(())
            }
            Err(e) => {
                self.state.transition(AgentState::Terminated)?;
                Err(e)
            }
        }
    }

    /// Start the agent without a model (simulation-only mode).
    pub fn start_simulation_mode(&mut self) -> Result<(), TransitionError> {
        self.state.transition(AgentState::Loading)?;
        self.state.transition(AgentState::Running)?;
        Ok(())
    }

    /// Run ONNX model inference.
    pub fn run_inference(&mut self, input: &[f32], batch_size: usize) -> Result<InferenceResult> {
        let model = self
            .model
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("No model loaded"))?;

        let result = model.run(input, batch_size)?;

        self.metrics.record_inference(result.latency_us);
        Ok(result)
    }

    /// Run hardware simulation on a 2D sample.
    pub fn run_hw_simulation(&mut self, samples: &[Array2<f64>]) -> SimulationResult {
        let start = Instant::now();

        let config = PipelineConfig {
            conv: self.conv_engine.config().clone(),
            dvfs: self.power_model.config.clone(),
            thermal: self.thermal.config().clone(),
        };

        let result = vlsi_sim::run_simulation(samples, &config);

        let latency_us = start.elapsed().as_micros() as u64;
        self.metrics.record_inference(latency_us);

        self.total_mac_ops += result.mac_operations;
        self.total_relu_ops += result.relu_operations;

        // Record power sample
        self.metrics.record_power(PowerSample {
            timestamp: Utc::now(),
            power_w: result.estimated_power_w,
            mac_ops: result.mac_operations,
            relu_ops: result.relu_operations,
            temperature_c: result.thermal_state.junction_temp_c,
        });

        self.last_sim_result = Some(result.clone());
        result
    }

    pub fn pause(&mut self) -> Result<(), TransitionError> {
        self.state.transition(AgentState::Paused).map(|_| ())
    }

    pub fn resume(&mut self) -> Result<(), TransitionError> {
        self.state.transition(AgentState::Running).map(|_| ())
    }

    pub fn terminate(&mut self) -> Result<(), TransitionError> {
        self.state.transition(AgentState::Terminated).map(|_| ())
    }

    pub fn state(&self) -> AgentState {
        self.state.current()
    }

    pub fn last_simulation(&self) -> Option<&SimulationResult> {
        self.last_sim_result.as_ref()
    }

    /// Get serializable agent info for the dashboard.
    pub fn info(&self) -> AgentInfo {
        let latency_snap = self.metrics.latency_hist.snapshot();
        let power_summary = self.metrics.power.summary();

        AgentInfo {
            id: self.id.to_string(),
            name: self.name.clone(),
            state: self.state.current(),
            inference_count: self.metrics.inference_count,
            total_mac_ops: self.total_mac_ops,
            total_relu_ops: self.total_relu_ops,
            avg_latency_us: latency_snap.mean,
            p99_latency_us: latency_snap.p99,
            power_w: power_summary.avg_power_w,
            temperature_c: power_summary.avg_temp_c,
            throughput_samples_per_sec: if latency_snap.mean > 0.0 {
                1_000_000.0 / latency_snap.mean
            } else {
                0.0
            },
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array2;

    #[test]
    fn test_agent_lifecycle() {
        let mut agent = Agent::new("test-agent".into());
        assert_eq!(agent.state(), AgentState::Idle);

        agent.start_simulation_mode().unwrap();
        assert_eq!(agent.state(), AgentState::Running);

        agent.pause().unwrap();
        assert_eq!(agent.state(), AgentState::Paused);

        agent.resume().unwrap();
        assert_eq!(agent.state(), AgentState::Running);

        agent.terminate().unwrap();
        assert_eq!(agent.state(), AgentState::Terminated);
    }

    #[test]
    fn test_agent_hw_simulation() {
        let mut agent = Agent::new("hw-agent".into());
        agent.start_simulation_mode().unwrap();

        let samples: Vec<Array2<f64>> = (0..3)
            .map(|_| Array2::ones((28, 28)))
            .collect();

        let result = agent.run_hw_simulation(&samples);
        assert_eq!(result.results.len(), 3);
        assert!(result.mac_operations > 0);

        let info = agent.info();
        assert_eq!(info.inference_count, 1);
        assert!(info.total_mac_ops > 0);
    }
}
