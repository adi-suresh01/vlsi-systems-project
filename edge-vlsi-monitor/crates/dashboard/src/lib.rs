//! Dashboard Web Server
//!
//! Axum-based web server providing:
//!   - REST API for benchmarks, agent management, and metrics
//!   - WebSocket for real-time metric streaming
//!   - Static file serving for the frontend

pub mod rest;
pub mod state;
pub mod ws;

use std::net::SocketAddr;
use std::path::Path;
use std::sync::Arc;

use axum::Router;
use tower_http::services::ServeDir;

use agent_runtime::SchedulerHandle;
use state::AppState;

/// Start the dashboard web server.
pub async fn serve(
    bind_addr: SocketAddr,
    frontend_dir: &Path,
    scheduler: SchedulerHandle,
    metrics_interval_ms: u64,
) -> anyhow::Result<()> {
    let state = Arc::new(AppState::new(scheduler));

    // Start the metrics broadcaster in the background
    let broadcaster_state = state.clone();
    tokio::spawn(async move {
        ws::metrics_broadcaster(broadcaster_state, metrics_interval_ms).await;
    });

    let app = Router::new()
        .merge(rest::router())
        .route("/ws", axum::routing::get(ws::ws_handler))
        .fallback_service(ServeDir::new(frontend_dir))
        .with_state(state);

    tracing::info!("Dashboard server starting on {}", bind_addr);
    tracing::info!("Frontend served from {}", frontend_dir.display());

    let listener = tokio::net::TcpListener::bind(bind_addr).await?;
    axum::serve(listener, app).await?;

    Ok(())
}
