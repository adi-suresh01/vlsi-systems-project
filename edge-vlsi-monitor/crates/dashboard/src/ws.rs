/// WebSocket handler for real-time metric streaming.
///
/// Replaces the poll-based fetch('/api/benchmark') approach from the
/// original dashboard.js with persistent WebSocket connections.

use std::sync::Arc;

use axum::{
    extract::{
        ws::{Message, WebSocket, WebSocketUpgrade},
        State,
    },
    response::IntoResponse,
};
use tokio::sync::broadcast;

use crate::state::{AppState, WsMessage};

/// WebSocket upgrade handler — GET /ws
pub async fn ws_handler(
    ws: WebSocketUpgrade,
    State(state): State<Arc<AppState>>,
) -> impl IntoResponse {
    ws.on_upgrade(move |socket| handle_socket(socket, state))
}

async fn handle_socket(mut socket: WebSocket, state: Arc<AppState>) {
    tracing::info!("WebSocket client connected");

    let mut rx = state.ws_broadcast.subscribe();

    // Send initial state
    let initial = serde_json::json!({
        "type": "connected",
        "data": { "message": "Connected to Edge VLSI Monitor" }
    });
    if socket
        .send(Message::Text(initial.to_string().into()))
        .await
        .is_err()
    {
        return;
    }

    loop {
        tokio::select! {
            // Forward broadcast messages to the client
            msg = rx.recv() => {
                match msg {
                    Ok(ws_msg) => {
                        match serde_json::to_string(&ws_msg) {
                            Ok(json) => {
                                if socket.send(Message::Text(json.into())).await.is_err() {
                                    break; // Client disconnected
                                }
                            }
                            Err(e) => {
                                tracing::warn!("Failed to serialize WS message: {}", e);
                            }
                        }
                    }
                    Err(broadcast::error::RecvError::Lagged(n)) => {
                        tracing::warn!("WebSocket client lagged by {} messages", n);
                        // Continue — client will get the next message
                    }
                    Err(broadcast::error::RecvError::Closed) => break,
                }
            }
            // Handle incoming messages from client (ping/pong, close)
            msg = socket.recv() => {
                match msg {
                    Some(Ok(Message::Ping(data))) => {
                        if socket.send(Message::Pong(data)).await.is_err() {
                            break;
                        }
                    }
                    Some(Ok(Message::Close(_))) | None => break,
                    _ => {} // Ignore text/binary from client
                }
            }
        }
    }

    tracing::info!("WebSocket client disconnected");
}

/// Background task that periodically broadcasts metrics to all WebSocket clients.
pub async fn metrics_broadcaster(state: Arc<AppState>, interval_ms: u64) {
    use tokio::sync::mpsc;
    use agent_runtime::SchedulerCommand;
    use crate::state::{AgentMetricsPayload, MetricsSnapshot};

    let mut interval = tokio::time::interval(tokio::time::Duration::from_millis(interval_ms));

    loop {
        interval.tick().await;

        // Fetch current agent metrics from the scheduler
        let (tx, mut rx) = mpsc::channel(1);
        if state
            .scheduler
            .send(SchedulerCommand::ListAgents { respond: tx })
            .await
            .is_err()
        {
            break; // Scheduler shut down
        }

        if let Some(agents) = rx.recv().await {
            let metrics: Vec<AgentMetricsPayload> =
                agents.iter().map(AgentMetricsPayload::from).collect();

            let snapshot = MetricsSnapshot {
                timestamp: chrono::Utc::now().to_rfc3339(),
                agent_count: metrics.len(),
                agents: metrics,
            };

            // Broadcast to all connected WebSocket clients
            let _ = state
                .ws_broadcast
                .send(WsMessage::MetricsUpdate(snapshot));
        }
    }
}
