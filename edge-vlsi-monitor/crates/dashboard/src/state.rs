/// Shared application state for the Axum web server.
///
/// Wraps the scheduler handle and broadcast channel for WebSocket streaming.

use std::collections::HashMap;
use std::sync::Arc;

use tokio::sync::{broadcast, RwLock};

use agent_runtime::{AgentInfo, SchedulerHandle};
use vlsi_sim::PowerBreakdown;

/// Shared state accessible from all request handlers.
#[derive(Clone)]
pub struct AppState {
    pub scheduler: SchedulerHandle,
    pub agents_cache: Arc<RwLock<HashMap<String, AgentInfo>>>,
    pub ws_broadcast: broadcast::Sender<WsMessage>,
}

impl AppState {
    pub fn new(scheduler: SchedulerHandle) -> Self {
        let (ws_tx, _) = broadcast::channel(256);
        Self {
            scheduler,
            agents_cache: Arc::new(RwLock::new(HashMap::new())),
            ws_broadcast: ws_tx,
        }
    }
}

/// Messages broadcast to all WebSocket clients.
#[derive(Debug, Clone, serde::Serialize)]
#[serde(tag = "type", content = "data")]
pub enum WsMessage {
    MetricsUpdate(MetricsSnapshot),
    AgentStateChange {
        id: String,
        name: String,
        state: String,
    },
    SimulationComplete(SimResultPayload),
    Error {
        message: String,
    },
}

/// Periodic metrics snapshot broadcast via WebSocket.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct MetricsSnapshot {
    pub timestamp: String,
    pub agent_count: usize,
    pub agents: Vec<AgentMetricsPayload>,
}

/// Per-agent metrics for the dashboard.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct AgentMetricsPayload {
    pub id: String,
    pub name: String,
    pub state: String,
    pub inference_count: u64,
    pub avg_latency_us: f64,
    pub p99_latency_us: u64,
    pub power_w: f64,
    pub temperature_c: f64,
    pub mac_ops: u64,
    pub relu_ops: u64,
    pub throughput_samples_per_sec: f64,
}

impl From<&AgentInfo> for AgentMetricsPayload {
    fn from(info: &AgentInfo) -> Self {
        Self {
            id: info.id.clone(),
            name: info.name.clone(),
            state: info.state.to_string(),
            inference_count: info.inference_count,
            avg_latency_us: info.avg_latency_us,
            p99_latency_us: info.p99_latency_us,
            power_w: info.power_w,
            temperature_c: info.temperature_c,
            mac_ops: info.total_mac_ops,
            relu_ops: info.total_relu_ops,
            throughput_samples_per_sec: info.throughput_samples_per_sec,
        }
    }
}

/// Hardware simulation result payload for the dashboard.
/// Matches the JSON schema from the original Python api.py lines 71-103.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct SimResultPayload {
    pub timestamp: String,
    pub sample_count: usize,
    pub performance: PerformancePayload,
    pub speedup: SpeedupPayload,
    pub power: PowerPayload,
    pub hardware_details: HardwareDetailsPayload,
    pub power_breakdown: PowerBreakdown,
    pub thermal: ThermalPayload,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct PerformancePayload {
    pub hardware_time: f64,
    pub hardware_per_sample: f64,
    pub theoretical_hw_time: f64,
    pub throughput: f64,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct SpeedupPayload {
    /// Ratio of wall-clock time vs theoretical hardware time
    pub simulation_vs_hardware: f64,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct PowerPayload {
    pub total_watts: f64,
    pub total_milliwatts: f64,
    pub total_microwatts: f64,
    pub energy_per_inference_uj: f64,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct HardwareDetailsPayload {
    pub cycles: u64,
    pub operations: u64,
    pub mac_ops: u64,
    pub relu_ops: u64,
    pub clock_frequency_mhz: f64,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct ThermalPayload {
    pub junction_temp_c: f64,
    pub should_throttle: bool,
    pub headroom_c: f64,
}
