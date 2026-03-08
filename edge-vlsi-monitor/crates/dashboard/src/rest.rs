/// REST API endpoints.
///
/// Ports the FastAPI endpoints from api.py and adds agent management.

use std::sync::Arc;

use axum::{
    extract::State,
    extract::Path,
    http::StatusCode,
    response::Json,
    routing::{delete, get, post},
    Router,
};
use chrono::Utc;
use tokio::sync::mpsc;
use uuid::Uuid;

use agent_runtime::{AgentInfo, SchedulerCommand};
use vlsi_sim::{generate_test_samples, PipelineConfig, SimulationResult};

use crate::state::*;

/// Build the REST API router.
pub fn router() -> Router<Arc<AppState>> {
    Router::new()
        .route("/api/status", get(get_status))
        .route("/api/benchmark", get(run_benchmark))
        .route("/api/simulate", post(run_simulation))
        .route("/api/agents", get(list_agents))
        .route("/api/agents", post(spawn_agent))
        .route("/api/agents/{id}", get(get_agent))
        .route("/api/agents/{id}", delete(terminate_agent))
        .route("/api/agents/{id}/pause", post(pause_agent))
        .route("/api/agents/{id}/resume", post(resume_agent))
        .route("/api/agents/{id}/simulate", post(simulate_agent))
        .route("/api/metrics", get(get_metrics))
}

/// GET /api/status — system health check.
async fn get_status(State(state): State<Arc<AppState>>) -> Json<serde_json::Value> {
    let (tx, mut rx) = mpsc::channel(1);
    let _ = state
        .scheduler
        .send(SchedulerCommand::ListAgents { respond: tx })
        .await;

    let agents = rx.recv().await.unwrap_or_default();

    Json(serde_json::json!({
        "ready": true,
        "agent_count": agents.len(),
        "message": "Edge VLSI Monitor is running"
    }))
}

/// GET /api/benchmark — run a full benchmark and return results.
/// Ports the /api/benchmark endpoint from api.py.
async fn run_benchmark(
    State(state): State<Arc<AppState>>,
) -> Result<Json<SimResultPayload>, (StatusCode, String)> {
    let sample_count = 10;
    let samples = generate_test_samples(sample_count, 28, 28);
    let config = PipelineConfig::default();

    let result = vlsi_sim::run_simulation(&samples, &config);

    let payload = build_sim_payload(&result, sample_count);

    // Broadcast to WebSocket clients
    let _ = state
        .ws_broadcast
        .send(WsMessage::SimulationComplete(payload.clone()));

    Ok(Json(payload))
}

#[derive(serde::Deserialize)]
struct SimulateRequest {
    #[serde(default = "default_samples")]
    samples: usize,
    #[serde(default = "default_clock")]
    clock_mhz: f64,
    #[serde(default)]
    parallel: bool,
    #[serde(default = "default_threads")]
    threads: usize,
}

fn default_samples() -> usize { 10 }
fn default_clock() -> f64 { 200.0 }
fn default_threads() -> usize { 0 }

/// POST /api/simulate — run a custom simulation.
async fn run_simulation(
    State(state): State<Arc<AppState>>,
    Json(req): Json<SimulateRequest>,
) -> Result<Json<SimResultPayload>, (StatusCode, String)> {
    let samples = generate_test_samples(req.samples, 28, 28);
    let mut config = PipelineConfig::default();
    config.dvfs.frequency_mhz = req.clock_mhz;

    let result = if req.parallel {
        let threads = if req.threads == 0 { None } else { Some(req.threads) };
        vlsi_sim::run_simulation_parallel(&samples, &config, threads)
    } else {
        vlsi_sim::run_simulation(&samples, &config)
    };

    let payload = build_sim_payload(&result, req.samples);

    let _ = state
        .ws_broadcast
        .send(WsMessage::SimulationComplete(payload.clone()));

    Ok(Json(payload))
}

/// GET /api/agents — list all agents.
async fn list_agents(State(state): State<Arc<AppState>>) -> Json<Vec<AgentInfo>> {
    let (tx, mut rx) = mpsc::channel(1);
    let _ = state
        .scheduler
        .send(SchedulerCommand::ListAgents { respond: tx })
        .await;

    let agents = rx.recv().await.unwrap_or_default();
    Json(agents)
}

#[derive(serde::Deserialize)]
struct SpawnRequest {
    name: String,
}

/// POST /api/agents — spawn a new agent.
async fn spawn_agent(
    State(state): State<Arc<AppState>>,
    Json(req): Json<SpawnRequest>,
) -> Result<(StatusCode, Json<AgentInfo>), (StatusCode, String)> {
    let (tx, mut rx) = mpsc::channel(1);
    let _ = state
        .scheduler
        .send(SchedulerCommand::SpawnAgent {
            name: req.name.clone(),
            respond: tx,
        })
        .await;

    match rx.recv().await {
        Some(info) => {
            let _ = state.ws_broadcast.send(WsMessage::AgentStateChange {
                id: info.id.clone(),
                name: info.name.clone(),
                state: info.state.to_string(),
            });
            Ok((StatusCode::CREATED, Json(info)))
        }
        None => Err((
            StatusCode::INTERNAL_SERVER_ERROR,
            "Failed to spawn agent".into(),
        )),
    }
}

/// GET /api/agents/:id — get agent info.
async fn get_agent(
    State(state): State<Arc<AppState>>,
    Path(id): Path<String>,
) -> Result<Json<AgentInfo>, (StatusCode, String)> {
    let uuid = Uuid::parse_str(&id).map_err(|_| (StatusCode::BAD_REQUEST, "Invalid UUID".into()))?;

    let (tx, mut rx) = mpsc::channel(1);
    let _ = state
        .scheduler
        .send(SchedulerCommand::GetAgent { id: uuid, respond: tx })
        .await;

    match rx.recv().await.flatten() {
        Some(info) => Ok(Json(info)),
        None => Err((StatusCode::NOT_FOUND, "Agent not found".into())),
    }
}

/// DELETE /api/agents/:id — terminate an agent.
async fn terminate_agent(
    State(state): State<Arc<AppState>>,
    Path(id): Path<String>,
) -> Result<StatusCode, (StatusCode, String)> {
    let uuid = Uuid::parse_str(&id).map_err(|_| (StatusCode::BAD_REQUEST, "Invalid UUID".into()))?;

    let (tx, mut rx) = mpsc::channel(1);
    let _ = state
        .scheduler
        .send(SchedulerCommand::TerminateAgent { id: uuid, respond: tx })
        .await;

    match rx.recv().await {
        Some(Ok(())) => {
            let _ = state.ws_broadcast.send(WsMessage::AgentStateChange {
                id: id.clone(),
                name: String::new(),
                state: "terminated".into(),
            });
            Ok(StatusCode::NO_CONTENT)
        }
        Some(Err(e)) => Err((StatusCode::BAD_REQUEST, e)),
        None => Err((StatusCode::INTERNAL_SERVER_ERROR, "Scheduler error".into())),
    }
}

/// POST /api/agents/:id/pause — pause an agent.
async fn pause_agent(
    State(state): State<Arc<AppState>>,
    Path(id): Path<String>,
) -> Result<StatusCode, (StatusCode, String)> {
    let uuid = Uuid::parse_str(&id).map_err(|_| (StatusCode::BAD_REQUEST, "Invalid UUID".into()))?;

    let (tx, mut rx) = mpsc::channel(1);
    let _ = state
        .scheduler
        .send(SchedulerCommand::PauseAgent { id: uuid, respond: tx })
        .await;

    match rx.recv().await {
        Some(Ok(())) => Ok(StatusCode::OK),
        Some(Err(e)) => Err((StatusCode::BAD_REQUEST, e)),
        None => Err((StatusCode::INTERNAL_SERVER_ERROR, "Scheduler error".into())),
    }
}

/// POST /api/agents/:id/resume — resume a paused agent.
async fn resume_agent(
    State(state): State<Arc<AppState>>,
    Path(id): Path<String>,
) -> Result<StatusCode, (StatusCode, String)> {
    let uuid = Uuid::parse_str(&id).map_err(|_| (StatusCode::BAD_REQUEST, "Invalid UUID".into()))?;

    let (tx, mut rx) = mpsc::channel(1);
    let _ = state
        .scheduler
        .send(SchedulerCommand::ResumeAgent { id: uuid, respond: tx })
        .await;

    match rx.recv().await {
        Some(Ok(())) => Ok(StatusCode::OK),
        Some(Err(e)) => Err((StatusCode::BAD_REQUEST, e)),
        None => Err((StatusCode::INTERNAL_SERVER_ERROR, "Scheduler error".into())),
    }
}

#[derive(serde::Deserialize)]
struct AgentSimRequest {
    #[serde(default = "default_samples")]
    samples: usize,
}

/// POST /api/agents/:id/simulate — run simulation on a specific agent.
async fn simulate_agent(
    State(state): State<Arc<AppState>>,
    Path(id): Path<String>,
    Json(req): Json<AgentSimRequest>,
) -> Result<Json<SimResultPayload>, (StatusCode, String)> {
    let uuid = Uuid::parse_str(&id).map_err(|_| (StatusCode::BAD_REQUEST, "Invalid UUID".into()))?;

    let samples = generate_test_samples(req.samples, 28, 28);

    let (tx, mut rx) = mpsc::channel(1);
    let _ = state
        .scheduler
        .send(SchedulerCommand::RunSimulation {
            id: uuid,
            samples,
            respond: tx,
        })
        .await;

    match rx.recv().await {
        Some(Ok(result)) => {
            let payload = build_sim_payload(&result, req.samples);
            Ok(Json(payload))
        }
        Some(Err(e)) => Err((StatusCode::BAD_REQUEST, e)),
        None => Err((StatusCode::INTERNAL_SERVER_ERROR, "Scheduler error".into())),
    }
}

/// GET /api/metrics — latest metrics snapshot.
async fn get_metrics(State(state): State<Arc<AppState>>) -> Json<MetricsSnapshot> {
    let (tx, mut rx) = mpsc::channel(1);
    let _ = state
        .scheduler
        .send(SchedulerCommand::ListAgents { respond: tx })
        .await;

    let agents = rx.recv().await.unwrap_or_default();
    let metrics: Vec<AgentMetricsPayload> = agents.iter().map(AgentMetricsPayload::from).collect();

    Json(MetricsSnapshot {
        timestamp: Utc::now().to_rfc3339(),
        agent_count: metrics.len(),
        agents: metrics,
    })
}

/// Build a SimResultPayload from a SimulationResult.
fn build_sim_payload(result: &SimulationResult, sample_count: usize) -> SimResultPayload {
    let hw_time = result.execution_time_secs;
    let per_sample = if sample_count > 0 {
        hw_time / sample_count as f64 * 1000.0 // ms
    } else {
        0.0
    };

    let total_power = result.power_breakdown.total_power_w;
    let energy_per_inf = if sample_count > 0 {
        total_power * hw_time / sample_count as f64 * 1_000_000.0 // µJ
    } else {
        0.0
    };

    SimResultPayload {
        timestamp: Utc::now().to_rfc3339(),
        sample_count,
        performance: PerformancePayload {
            hardware_time: hw_time,
            hardware_per_sample: per_sample,
            theoretical_hw_time: result.theoretical_hw_time_secs,
            throughput: result.throughput_samples_per_sec,
        },
        speedup: SpeedupPayload {
            simulation_vs_hardware: if result.theoretical_hw_time_secs > 0.0 {
                hw_time / result.theoretical_hw_time_secs
            } else {
                0.0
            },
        },
        power: PowerPayload {
            total_watts: total_power,
            total_milliwatts: total_power * 1000.0,
            total_microwatts: total_power * 1_000_000.0,
            energy_per_inference_uj: energy_per_inf,
        },
        hardware_details: HardwareDetailsPayload {
            cycles: result.simulated_cycles,
            operations: result.operations_count,
            mac_ops: result.mac_operations,
            relu_ops: result.relu_operations,
            clock_frequency_mhz: result.clock_frequency_mhz,
        },
        power_breakdown: result.power_breakdown.clone(),
        thermal: ThermalPayload {
            junction_temp_c: result.thermal_state.junction_temp_c,
            should_throttle: result.thermal_state.should_throttle,
            headroom_c: result.thermal_state.headroom_c,
        },
    }
}
