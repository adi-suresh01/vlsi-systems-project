/// Power estimation model with DVFS (Dynamic Voltage and Frequency Scaling).
///
/// Ports constants from:
///   - hardware_sim.py lines 65-74:
///       clock_freq = 200 MHz, base_power = 10 mW, dynamic = 1 nW/MAC
///   - hardware_simulator.cpp lines 64, 72:
///       1e-9 per MAC, 1e-10 per ReLU
///   - power_estimator.py lines 19-23:
///       base 0.5W + 0.0001/cycle + 0.0001/op (alternative model)
///
/// Extends with DVFS voltage/frequency scaling and leakage power.

/// Power breakdown for a simulation run.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct PowerBreakdown {
    /// Static base power (W). Default: 0.010 (10 mW)
    pub base_power_w: f64,
    /// Dynamic switching power from all operations (W)
    pub dynamic_power_w: f64,
    /// Power contribution from MAC operations (W)
    pub mac_power_w: f64,
    /// Power contribution from ReLU operations (W)
    pub relu_power_w: f64,
    /// Temperature-dependent leakage power (W)
    pub leakage_power_w: f64,
    /// Total power (W)
    pub total_power_w: f64,
}

/// DVFS configuration — voltage/frequency operating points.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct DvfsConfig {
    /// Current operating voltage (V)
    pub voltage: f64,
    /// Current operating frequency (MHz)
    pub frequency_mhz: f64,
    /// Available DVFS levels: (voltage, frequency_mhz)
    pub levels: Vec<(f64, f64)>,
}

impl Default for DvfsConfig {
    fn default() -> Self {
        Self {
            voltage: 1.0,
            frequency_mhz: 200.0, // from hardware_sim.py line 65
            levels: vec![
                (0.6, 50.0),   // Ultra-low power
                (0.7, 100.0),  // Low power
                (0.8, 150.0),  // Balanced
                (1.0, 200.0),  // Performance (default)
                (1.1, 250.0),  // Turbo
            ],
        }
    }
}

/// Physics-based power estimation model.
#[derive(Debug, Clone)]
pub struct PowerModel {
    pub config: DvfsConfig,
    /// Base (static) power at nominal voltage (W)
    base_power_w: f64,
    /// Energy per MAC operation at nominal voltage (J)
    energy_per_mac_j: f64,
    /// Energy per ReLU operation at nominal voltage (J)
    energy_per_relu_j: f64,
    /// Leakage power coefficient (W, temperature-independent part)
    leakage_coeff_w: f64,
}

impl PowerModel {
    pub fn new(config: DvfsConfig) -> Self {
        Self {
            config,
            base_power_w: 0.010,    // 10 mW, from hardware_sim.py line 72
            energy_per_mac_j: 1e-9,  // 1 nW, from hardware_sim.py line 73 & cpp line 64
            energy_per_relu_j: 1e-10, // 0.1 nW, from cpp line 72
            leakage_coeff_w: 0.001,  // 1 mW base leakage
        }
    }

    /// Estimate power from operation counts (no thermal feedback).
    pub fn estimate(&self, mac_ops: u64, relu_ops: u64) -> PowerBreakdown {
        self.estimate_with_thermal(mac_ops, relu_ops, 25.0) // assume 25°C
    }

    /// Estimate power with thermal feedback.
    ///
    /// Dynamic power scales with V^2 * f (DVFS-aware).
    /// Leakage doubles approximately every 10°C above 25°C.
    pub fn estimate_with_thermal(&self, mac_ops: u64, relu_ops: u64, temp_c: f64) -> PowerBreakdown {
        let v = self.config.voltage;
        let f_ratio = self.config.frequency_mhz / 200.0; // normalize to default 200 MHz

        // Dynamic power scales as V^2 * f
        let dvfs_scale = v * v * f_ratio;

        let mac_power = mac_ops as f64 * self.energy_per_mac_j * dvfs_scale;
        let relu_power = relu_ops as f64 * self.energy_per_relu_j * dvfs_scale;
        let dynamic_power = mac_power + relu_power;

        // Base power scales linearly with voltage
        let base_power = self.base_power_w * (v / 1.0);

        // Leakage power: doubles every 10°C above 25°C
        let temp_factor = 2.0_f64.powf((temp_c - 25.0) / 10.0);
        let leakage_power = self.leakage_coeff_w * temp_factor * (v / 1.0);

        let total = base_power + dynamic_power + leakage_power;

        PowerBreakdown {
            base_power_w: base_power,
            dynamic_power_w: dynamic_power,
            mac_power_w: mac_power,
            relu_power_w: relu_power,
            leakage_power_w: leakage_power,
            total_power_w: total,
        }
    }

    /// Switch to a different DVFS level by index.
    pub fn set_dvfs_level(&mut self, level_index: usize) -> Option<(f64, f64)> {
        if let Some(&(v, f)) = self.config.levels.get(level_index) {
            self.config.voltage = v;
            self.config.frequency_mhz = f;
            Some((v, f))
        } else {
            None
        }
    }

    /// Calculate energy per inference (J).
    pub fn energy_per_inference(
        &self,
        power: &PowerBreakdown,
        total_time_secs: f64,
        num_samples: usize,
    ) -> f64 {
        if num_samples == 0 {
            return 0.0;
        }
        power.total_power_w * total_time_secs / num_samples as f64
    }

    pub fn frequency_mhz(&self) -> f64 {
        self.config.frequency_mhz
    }

    pub fn voltage(&self) -> f64 {
        self.config.voltage
    }
}

impl Default for PowerModel {
    fn default() -> Self {
        Self::new(DvfsConfig::default())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_power_basic() {
        let model = PowerModel::default();
        let breakdown = model.estimate(1000, 100);

        // base_power = 10 mW
        assert!((breakdown.base_power_w - 0.010).abs() < 1e-10);
        // mac_power = 1000 * 1e-9 = 1e-6 = 0.001 mW
        assert!((breakdown.mac_power_w - 1e-6).abs() < 1e-12);
        // relu_power = 100 * 1e-10 = 1e-8
        assert!((breakdown.relu_power_w - 1e-8).abs() < 1e-14);
        assert!(breakdown.total_power_w > breakdown.base_power_w);
    }

    #[test]
    fn test_dvfs_scaling() {
        let mut model = PowerModel::default();

        // At nominal (1.0V, 200MHz)
        let p_nominal = model.estimate(10000, 1000);

        // At low power (0.7V, 100MHz)
        model.set_dvfs_level(1); // 0.7V, 100 MHz
        let p_low = model.estimate(10000, 1000);

        // Dynamic power should be lower at reduced V^2 * f
        assert!(p_low.dynamic_power_w < p_nominal.dynamic_power_w);
    }

    #[test]
    fn test_thermal_leakage() {
        let model = PowerModel::default();

        let p_25 = model.estimate_with_thermal(1000, 100, 25.0);
        let p_85 = model.estimate_with_thermal(1000, 100, 85.0);

        // Leakage should be much higher at 85°C
        assert!(p_85.leakage_power_w > p_25.leakage_power_w * 4.0);
    }

    #[test]
    fn test_energy_per_inference() {
        let model = PowerModel::default();
        let breakdown = model.estimate(1000, 100);
        let energy = model.energy_per_inference(&breakdown, 0.1, 10);
        assert!(energy > 0.0);
    }
}
