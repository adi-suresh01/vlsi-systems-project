/// Integration tests for the VLSI simulation engine.

use vlsi_sim::{
    generate_test_samples, run_simulation, run_simulation_parallel, ConvConfig, ConvEngine,
    DvfsConfig, MacUnit, PipelineConfig, PowerModel, ReluUnit, ThermalModel,
};

#[test]
fn test_end_to_end_simulation() {
    let samples = generate_test_samples(5, 28, 28);
    let config = PipelineConfig::default();
    let result = run_simulation(&samples, &config);

    assert_eq!(result.results.len(), 5);
    assert!(result.execution_time_secs > 0.0);
    assert!(result.mac_operations > 0);
    assert!(result.relu_operations > 0);
    assert!(result.estimated_power_w > 0.0);
    assert!(result.throughput_samples_per_sec > 0.0);
    assert!((result.clock_frequency_mhz - 200.0).abs() < 1e-10);

    // Verify deterministic operation counts
    // 5 samples * 8 kernels * (7*7) patches * 9 MACs = 5 * 8 * 49 * 9 = 17640
    assert_eq!(result.mac_operations, 17640);
    assert_eq!(result.relu_operations, 40); // 5 * 8
}

#[test]
fn test_parallel_matches_sequential_op_counts() {
    let samples = generate_test_samples(3, 28, 28);
    let config = PipelineConfig::default();

    let seq = run_simulation(&samples, &config);
    let par = run_simulation_parallel(&samples, &config, Some(2));

    // Operation counts must match exactly
    assert_eq!(seq.mac_operations, par.mac_operations);
    assert_eq!(seq.relu_operations, par.relu_operations);
    assert_eq!(seq.simulated_cycles, par.simulated_cycles);
}

#[test]
fn test_dvfs_power_scaling() {
    let samples = generate_test_samples(5, 28, 28);

    let config_high = PipelineConfig {
        dvfs: DvfsConfig {
            voltage: 1.1,
            frequency_mhz: 250.0,
            ..DvfsConfig::default()
        },
        ..PipelineConfig::default()
    };

    let config_low = PipelineConfig {
        dvfs: DvfsConfig {
            voltage: 0.7,
            frequency_mhz: 100.0,
            ..DvfsConfig::default()
        },
        ..PipelineConfig::default()
    };

    let result_high = run_simulation(&samples, &config_high);
    let result_low = run_simulation(&samples, &config_low);

    // Higher voltage/freq should consume more power
    assert!(result_high.estimated_power_w > result_low.estimated_power_w);
    // Lower frequency = longer theoretical HW time
    assert!(result_low.theoretical_hw_time_secs > result_high.theoretical_hw_time_secs);
}

#[test]
fn test_conv_engine_different_sizes() {
    let config = ConvConfig {
        num_kernels: 4,
        kernel_size: 3,
        stride: 2,
        weight_scale: 0.1,
    };
    let mut engine = ConvEngine::new(config);

    // 14x14 image
    let small = ndarray::Array2::ones((14, 14));
    let result = engine.process_sample(&small);
    assert_eq!(result.feature_maps.len(), 4);
    assert!(result.mac_ops > 0);
}

#[test]
fn test_thermal_model_under_load() {
    let mut thermal = ThermalModel::default();
    let power = PowerModel::default();

    // Simulate heavy workload
    let breakdown = power.estimate(1_000_000, 100_000);

    // Run thermal simulation for 1 second at that power level
    for _ in 0..100 {
        thermal.update(breakdown.total_power_w, 0.01);
    }

    // Temperature should have risen above ambient
    assert!(thermal.state().junction_temp_c > 25.0);
}
