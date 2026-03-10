/// Hardware pipeline simulation.
///
/// Ports `run_fast_hardware_simulation()` from hardware_sim.py and
/// `HardwarePipeline` from pipeline_concurrent.py. Provides both
/// sequential and parallel (rayon) execution modes.

use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::Instant;

use ndarray::Array2;
use rayon::prelude::*;

use crate::conv::{ConvConfig, ConvEngine, ConvResult};
use crate::power::{DvfsConfig, PowerBreakdown, PowerModel};
use crate::thermal::{ThermalConfig, ThermalModel, ThermalState};
use crate::workload::WorkloadProfile;

/// Complete simulation result — matches the Python dict from hardware_sim.py lines 76-93.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct SimulationResult {
    /// Per-sample output values
    pub results: Vec<f64>,
    /// Wall-clock execution time (seconds)
    pub execution_time_secs: f64,
    /// Total simulated clock cycles
    pub simulated_cycles: u64,
    /// Estimated total power (W)
    pub estimated_power_w: f64,
    /// Throughput (samples/second)
    pub throughput_samples_per_sec: f64,
    /// Total operations (MAC + ReLU)
    pub operations_count: u64,
    /// MAC operations only
    pub mac_operations: u64,
    /// ReLU operations only
    pub relu_operations: u64,
    /// Theoretical hardware execution time at clock frequency
    pub theoretical_hw_time_secs: f64,
    /// Clock frequency used (MHz)
    pub clock_frequency_mhz: f64,
    /// Detailed power breakdown
    pub power_breakdown: PowerBreakdown,
    /// Thermal state at end of simulation
    pub thermal_state: ThermalState,
}

/// Pipeline configuration.
#[derive(Debug, Clone)]
pub struct PipelineConfig {
    pub conv: ConvConfig,
    pub dvfs: DvfsConfig,
    pub thermal: ThermalConfig,
}

impl Default for PipelineConfig {
    fn default() -> Self {
        Self {
            conv: ConvConfig::default(),
            dvfs: DvfsConfig::default(),
            thermal: ThermalConfig::default(),
        }
    }
}

/// Run sequential hardware simulation (single-threaded).
///
/// Direct port of `run_fast_hardware_simulation()` from hardware_sim.py.
pub fn run_simulation(samples: &[Array2<f64>], config: &PipelineConfig) -> SimulationResult {
    let start = Instant::now();

    let mut engine = ConvEngine::new(config.conv.clone());
    let results: Vec<ConvResult> = engine.process_batch(samples);

    let execution_time = start.elapsed().as_secs_f64();

    let total_mac = engine.total_mac_ops();
    let total_relu = engine.total_relu_ops();
    let total_ops = total_mac + total_relu;

    // Cycle calculation: 1 cycle per MAC + 1 cycle per ReLU (from hardware_sim.py lines 66-69)
    let total_cycles = total_mac + total_relu;

    let freq_mhz = config.dvfs.frequency_mhz;
    let theoretical_hw_time = total_cycles as f64 / (freq_mhz * 1e6);

    // Power estimation
    let power_model = PowerModel::new(config.dvfs.clone());
    let power_breakdown = power_model.estimate(total_mac, total_relu);

    // Thermal simulation over theoretical HW time
    let mut thermal = ThermalModel::new(config.thermal.clone());
    if theoretical_hw_time > 0.0 {
        let steps = 10;
        let dt = theoretical_hw_time / steps as f64;
        for _ in 0..steps {
            thermal.update(power_breakdown.total_power_w, dt);
        }
    }

    let throughput = if execution_time > 0.0 {
        samples.len() as f64 / execution_time
    } else {
        0.0
    };

    SimulationResult {
        results: results.iter().map(|r| r.output).collect(),
        execution_time_secs: execution_time,
        simulated_cycles: total_cycles,
        estimated_power_w: power_breakdown.total_power_w,
        throughput_samples_per_sec: throughput,
        operations_count: total_ops,
        mac_operations: total_mac,
        relu_operations: total_relu,
        theoretical_hw_time_secs: theoretical_hw_time,
        clock_frequency_mhz: freq_mhz,
        power_breakdown,
        thermal_state: thermal.state().clone(),
    }
}

/// Run parallel hardware simulation using rayon work-stealing.
///
/// Port of C++ `run_simulation()` with `std::async` parallelism.
pub fn run_simulation_parallel(
    samples: &[Array2<f64>],
    config: &PipelineConfig,
    num_threads: Option<usize>,
) -> SimulationResult {
    if let Some(n) = num_threads {
        rayon::ThreadPoolBuilder::new()
            .num_threads(n)
            .build_global()
            .ok(); // Ignore error if pool already initialized
    }

    let start = Instant::now();

    let total_mac = Arc::new(AtomicU64::new(0));
    let total_relu = Arc::new(AtomicU64::new(0));

    let results: Vec<f64> = samples
        .par_iter()
        .map(|sample| {
            let mut engine = ConvEngine::new(config.conv.clone());
            let result = engine.process_sample(sample);
            total_mac.fetch_add(result.mac_ops, Ordering::Relaxed);
            total_relu.fetch_add(result.relu_ops, Ordering::Relaxed);
            result.output
        })
        .collect();

    let execution_time = start.elapsed().as_secs_f64();

    let mac_ops = total_mac.load(Ordering::Relaxed);
    let relu_ops = total_relu.load(Ordering::Relaxed);
    let total_ops = mac_ops + relu_ops;
    let total_cycles = mac_ops + relu_ops;

    let freq_mhz = config.dvfs.frequency_mhz;
    let theoretical_hw_time = total_cycles as f64 / (freq_mhz * 1e6);

    let power_model = PowerModel::new(config.dvfs.clone());
    let power_breakdown = power_model.estimate(mac_ops, relu_ops);

    let mut thermal = ThermalModel::new(config.thermal.clone());
    if theoretical_hw_time > 0.0 {
        let steps = 10;
        let dt = theoretical_hw_time / steps as f64;
        for _ in 0..steps {
            thermal.update(power_breakdown.total_power_w, dt);
        }
    }

    let throughput = if execution_time > 0.0 {
        samples.len() as f64 / execution_time
    } else {
        0.0
    };

    SimulationResult {
        results,
        execution_time_secs: execution_time,
        simulated_cycles: total_cycles,
        estimated_power_w: power_breakdown.total_power_w,
        throughput_samples_per_sec: throughput,
        operations_count: total_ops,
        mac_operations: mac_ops,
        relu_operations: relu_ops,
        theoretical_hw_time_secs: theoretical_hw_time,
        clock_frequency_mhz: freq_mhz,
        power_breakdown,
        thermal_state: thermal.state().clone(),
    }
}

/// Generate random test samples (MNIST-like 28x28 images).
pub fn generate_test_samples(count: usize, rows: usize, cols: usize) -> Vec<Array2<f64>> {
    use rand::Rng;
    let mut rng = rand::thread_rng();
    (0..count)
        .map(|_| Array2::from_shape_fn((rows, cols), |_| rng.gen::<f64>()))
        .collect()
}

/// Run hardware simulation from a generic workload profile.
///
/// Model-agnostic entry point. Takes pre-computed operation counts and feeds
/// them through the power and thermal models. Use this for real ONNX model
/// results, transformer attention sweeps, or any workload described by op counts.
pub fn run_simulation_from_profile(
    profile: &WorkloadProfile,
    dvfs: &DvfsConfig,
    thermal_config: &ThermalConfig,
) -> SimulationResult {
    let start = Instant::now();

    let total_mac = profile.total_mac_ops();
    let total_activation = profile.total_activation_ops();
    let total_ops = total_mac + total_activation;
    let total_cycles = total_mac + total_activation; // 1 cycle per op

    let freq_mhz = dvfs.frequency_mhz;
    let theoretical_hw_time = total_cycles as f64 / (freq_mhz * 1e6);

    let power_model = PowerModel::new(dvfs.clone());
    let power_breakdown = power_model.estimate(total_mac, total_activation);

    let mut thermal = ThermalModel::new(thermal_config.clone());
    if theoretical_hw_time > 0.0 {
        let steps = 10;
        let dt = theoretical_hw_time / steps as f64;
        for _ in 0..steps {
            thermal.update(power_breakdown.total_power_w, dt);
        }
    }

    let execution_time = start.elapsed().as_secs_f64();
    let throughput = if execution_time > 0.0 {
        profile.num_inferences as f64 / execution_time
    } else {
        0.0
    };

    SimulationResult {
        results: vec![],
        execution_time_secs: execution_time,
        simulated_cycles: total_cycles,
        estimated_power_w: power_breakdown.total_power_w,
        throughput_samples_per_sec: throughput,
        operations_count: total_ops,
        mac_operations: total_mac,
        relu_operations: total_activation,
        theoretical_hw_time_secs: theoretical_hw_time,
        clock_frequency_mhz: freq_mhz,
        power_breakdown,
        thermal_state: thermal.state().clone(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_simulation_sequential() {
        let config = PipelineConfig::default();
        let samples = generate_test_samples(5, 28, 28);
        let result = run_simulation(&samples, &config);

        assert_eq!(result.results.len(), 5);
        assert!(result.execution_time_secs > 0.0);
        assert!(result.mac_operations > 0);
        assert!(result.relu_operations > 0);
        assert!(result.estimated_power_w > 0.0);
        assert!(result.throughput_samples_per_sec > 0.0);
        assert!((result.clock_frequency_mhz - 200.0).abs() < 1e-10);
    }

    #[test]
    fn test_simulation_parallel() {
        let config = PipelineConfig::default();
        let samples = generate_test_samples(5, 28, 28);
        let result = run_simulation_parallel(&samples, &config, Some(2));

        assert_eq!(result.results.len(), 5);
        assert!(result.mac_operations > 0);
        assert!(result.relu_operations > 0);
    }

    #[test]
    fn test_simulation_op_counts_match() {
        let config = PipelineConfig::default();
        let samples = generate_test_samples(3, 28, 28);

        let seq = run_simulation(&samples, &config);

        // MAC count should be deterministic: 3 samples * 8 kernels * 49 patches * 9 MACs
        let expected_mac = 3 * 8 * 49 * 9;
        assert_eq!(seq.mac_operations, expected_mac as u64);
        assert_eq!(seq.relu_operations, 3 * 8); // 1 ReLU per kernel per sample
    }

    #[test]
    fn test_thermal_feedback_during_sim() {
        let config = PipelineConfig::default();
        let samples = generate_test_samples(5, 28, 28);
        let result = run_simulation(&samples, &config);

        // Temperature should still be near ambient for such a small workload
        assert!(result.thermal_state.junction_temp_c >= 25.0);
        assert!(!result.thermal_state.should_throttle);
    }
}
