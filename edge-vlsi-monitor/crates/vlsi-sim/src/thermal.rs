/// Thermal feedback model using an RC thermal network.
///
/// Models junction temperature as a function of power dissipation over time.
/// New feature not in the original Python/C++ project — enables realistic
/// edge AI thermal throttling simulation.
///
/// Model: dT/dt = (P * R_th - (T_j - T_amb)) / (R_th * C_th)
///   where P = power dissipation (W)
///         R_th = thermal resistance (°C/W)
///         C_th = thermal capacitance (J/°C)
///         T_j = junction temperature
///         T_amb = ambient temperature

/// Thermal configuration parameters.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct ThermalConfig {
    /// Ambient temperature (°C)
    pub ambient_temp_c: f64,
    /// Thermal resistance: junction to ambient (°C/W)
    pub thermal_resistance: f64,
    /// Thermal capacitance (J/°C)
    pub thermal_capacitance: f64,
    /// Maximum junction temperature before throttling (°C)
    pub max_temp_c: f64,
}

impl Default for ThermalConfig {
    fn default() -> Self {
        Self {
            ambient_temp_c: 25.0,
            thermal_resistance: 10.0,   // Typical for small SoC
            thermal_capacitance: 0.5,    // Typical for die + package
            max_temp_c: 85.0,            // Industry standard Tj max
        }
    }
}

/// Current thermal state of the simulated die.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct ThermalState {
    /// Current junction temperature (°C)
    pub junction_temp_c: f64,
    /// Whether the chip should throttle to reduce temperature
    pub should_throttle: bool,
    /// Temperature headroom before throttle (°C)
    pub headroom_c: f64,
}

/// Thermal model with RC network dynamics.
#[derive(Debug, Clone)]
pub struct ThermalModel {
    config: ThermalConfig,
    state: ThermalState,
}

impl ThermalModel {
    pub fn new(config: ThermalConfig) -> Self {
        let junction = config.ambient_temp_c;
        let headroom = config.max_temp_c - junction;
        Self {
            state: ThermalState {
                junction_temp_c: junction,
                should_throttle: false,
                headroom_c: headroom,
            },
            config,
        }
    }

    /// Update thermal state given power dissipation over a time step.
    ///
    /// Uses forward Euler integration of the RC thermal model.
    pub fn update(&mut self, power_w: f64, dt_secs: f64) -> &ThermalState {
        let r_th = self.config.thermal_resistance;
        let c_th = self.config.thermal_capacitance;
        let t_amb = self.config.ambient_temp_c;
        let t_j = self.state.junction_temp_c;

        // dT/dt = (P * R_th - (T_j - T_amb)) / (R_th * C_th)
        let dt_dt = (power_w * r_th - (t_j - t_amb)) / (r_th * c_th);
        let new_temp = t_j + dt_dt * dt_secs;

        // Clamp to ambient (can't go below ambient via cooling)
        self.state.junction_temp_c = new_temp.max(self.config.ambient_temp_c);
        self.state.should_throttle = self.state.junction_temp_c >= self.config.max_temp_c;
        self.state.headroom_c = self.config.max_temp_c - self.state.junction_temp_c;

        &self.state
    }

    /// Compute steady-state temperature for a given constant power.
    /// T_ss = T_amb + P * R_th
    pub fn steady_state_temp(&self, power_w: f64) -> f64 {
        self.config.ambient_temp_c + power_w * self.config.thermal_resistance
    }

    /// Reset to ambient temperature.
    pub fn reset(&mut self) {
        self.state.junction_temp_c = self.config.ambient_temp_c;
        self.state.should_throttle = false;
        self.state.headroom_c = self.config.max_temp_c - self.config.ambient_temp_c;
    }

    pub fn state(&self) -> &ThermalState {
        &self.state
    }

    pub fn config(&self) -> &ThermalConfig {
        &self.config
    }
}

impl Default for ThermalModel {
    fn default() -> Self {
        Self::new(ThermalConfig::default())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_thermal_starts_at_ambient() {
        let model = ThermalModel::default();
        assert!((model.state().junction_temp_c - 25.0).abs() < 1e-10);
        assert!(!model.state().should_throttle);
    }

    #[test]
    fn test_thermal_heats_up() {
        let mut model = ThermalModel::default();
        // Apply 1W for 1 second
        model.update(1.0, 1.0);
        assert!(model.state().junction_temp_c > 25.0);
    }

    #[test]
    fn test_thermal_steady_state() {
        let model = ThermalModel::default();
        // T_ss = 25 + 1.0 * 10.0 = 35.0
        let ss = model.steady_state_temp(1.0);
        assert!((ss - 35.0).abs() < 1e-10);
    }

    #[test]
    fn test_thermal_throttle() {
        let mut model = ThermalModel::new(ThermalConfig {
            max_temp_c: 30.0,
            ..Default::default()
        });
        // Push temperature above 30°C
        for _ in 0..100 {
            model.update(5.0, 0.1);
        }
        assert!(model.state().should_throttle);
    }

    #[test]
    fn test_thermal_cools_down() {
        let mut model = ThermalModel::default();
        // Heat up
        for _ in 0..50 {
            model.update(5.0, 0.1);
        }
        let hot_temp = model.state().junction_temp_c;

        // Cool down (zero power)
        for _ in 0..200 {
            model.update(0.0, 0.1);
        }
        assert!(model.state().junction_temp_c < hot_temp);
    }

    #[test]
    fn test_thermal_reset() {
        let mut model = ThermalModel::default();
        model.update(5.0, 1.0);
        model.reset();
        assert!((model.state().junction_temp_c - 25.0).abs() < 1e-10);
    }
}
