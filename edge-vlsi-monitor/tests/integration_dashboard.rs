/// Integration tests for the dashboard state types.

use dashboard::state::*;
use vlsi_sim::{generate_test_samples, run_simulation, PipelineConfig, PowerBreakdown};

#[test]
fn test_sim_result_payload_serialization() {
    let samples = generate_test_samples(5, 28, 28);
    let config = PipelineConfig::default();
    let result = run_simulation(&samples, &config);

    // Build payload (replicating what rest.rs does)
    let hw_time = result.execution_time_secs;
    let total_power = result.power_breakdown.total_power_w;

    let payload = SimResultPayload {
        timestamp: "2026-01-01T00:00:00Z".to_string(),
        sample_count: 5,
        performance: PerformancePayload {
            hardware_time: hw_time,
            hardware_per_sample: hw_time / 5.0 * 1000.0,
            theoretical_hw_time: result.theoretical_hw_time_secs,
            throughput: result.throughput_samples_per_sec,
        },
        speedup: SpeedupPayload {
            simulation_vs_hardware: hw_time / result.theoretical_hw_time_secs,
        },
        power: PowerPayload {
            total_watts: total_power,
            total_milliwatts: total_power * 1000.0,
            total_microwatts: total_power * 1_000_000.0,
            energy_per_inference_uj: total_power * hw_time / 5.0 * 1_000_000.0,
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
    };

    // Verify it serializes to valid JSON
    let json = serde_json::to_string_pretty(&payload).unwrap();
    assert!(json.contains("\"sample_count\":5"));
    assert!(json.contains("\"mac_ops\""));
    assert!(json.contains("\"junction_temp_c\""));

    // Verify it round-trips
    let deserialized: SimResultPayload = serde_json::from_str(&json).unwrap();
    assert_eq!(deserialized.sample_count, 5);
    assert_eq!(deserialized.hardware_details.mac_ops, result.mac_operations);
}

#[test]
fn test_agent_metrics_payload() {
    let payload = AgentMetricsPayload {
        id: "test-id".to_string(),
        name: "test-agent".to_string(),
        state: "running".to_string(),
        inference_count: 42,
        avg_latency_us: 150.5,
        p99_latency_us: 500,
        power_w: 0.012,
        temperature_c: 35.0,
        mac_ops: 10000,
        relu_ops: 500,
        throughput_samples_per_sec: 6600.0,
    };

    let json = serde_json::to_string(&payload).unwrap();
    assert!(json.contains("\"state\":\"running\""));
}

#[test]
fn test_ws_message_serialization() {
    let msg = WsMessage::AgentStateChange {
        id: "abc-123".to_string(),
        name: "agent-1".to_string(),
        state: "paused".to_string(),
    };

    let json = serde_json::to_string(&msg).unwrap();
    assert!(json.contains("\"type\":\"AgentStateChange\""));
    assert!(json.contains("\"state\":\"paused\""));
}
