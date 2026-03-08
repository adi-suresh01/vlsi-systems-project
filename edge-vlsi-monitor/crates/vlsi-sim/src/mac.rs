/// MAC (Multiply-Accumulate) unit simulation.
///
/// Models the behavior of the SystemVerilog `mac_unit.sv`:
///   - 1-cycle latency per MAC operation
///   - Accumulator register for running sums
///   - Matches C++ `simulate_mac_hw(a, b, c) -> a*b + c`

#[derive(Debug, Clone)]
pub struct MacUnit {
    accumulator: f64,
    op_count: u64,
    latency_cycles: u32,
}

impl MacUnit {
    pub fn new() -> Self {
        Self {
            accumulator: 0.0,
            op_count: 0,
            latency_cycles: 1, // 1 cycle per MAC, from hardware_sim.py line 66
        }
    }

    /// Reset accumulator and counters.
    pub fn reset(&mut self) {
        self.accumulator = 0.0;
        self.op_count = 0;
    }

    /// Execute a*b and accumulate into internal register. Returns new accumulator value.
    pub fn execute_accumulate(&mut self, a: f64, b: f64) -> f64 {
        self.accumulator += a * b;
        self.op_count += 1;
        self.accumulator
    }

    /// Execute a*b + c without touching the accumulator.
    /// Direct port of C++ `simulate_mac_hw(a, b, c)`.
    #[inline]
    pub fn execute(&mut self, a: f64, b: f64, c: f64) -> f64 {
        self.op_count += 1;
        a * b + c
    }

    pub fn op_count(&self) -> u64 {
        self.op_count
    }

    pub fn accumulator(&self) -> f64 {
        self.accumulator
    }

    pub fn latency_cycles(&self) -> u32 {
        self.latency_cycles
    }
}

impl Default for MacUnit {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mac_basic() {
        let mut mac = MacUnit::new();
        // a*b + c = 3*4 + 0 = 12
        let result = mac.execute(3.0, 4.0, 0.0);
        assert!((result - 12.0).abs() < 1e-10);
        assert_eq!(mac.op_count(), 1);
    }

    #[test]
    fn test_mac_accumulate() {
        let mut mac = MacUnit::new();
        // 3*4 = 12, accumulator = 12
        let r1 = mac.execute_accumulate(3.0, 4.0);
        assert!((r1 - 12.0).abs() < 1e-10);
        // 5*6 = 30, accumulator = 42
        let r2 = mac.execute_accumulate(5.0, 6.0);
        assert!((r2 - 42.0).abs() < 1e-10);
        assert_eq!(mac.op_count(), 2);
    }

    #[test]
    fn test_mac_chained() {
        let mut mac = MacUnit::new();
        // Simulates conv_sum = simulate_mac_hw(pixel, kernel, conv_sum)
        let mut conv_sum = 0.0;
        conv_sum = mac.execute(2.0, 0.5, conv_sum); // 1.0
        conv_sum = mac.execute(3.0, 0.5, conv_sum); // 2.5
        conv_sum = mac.execute(4.0, 0.5, conv_sum); // 4.5
        assert!((conv_sum - 4.5).abs() < 1e-10);
        assert_eq!(mac.op_count(), 3);
    }

    #[test]
    fn test_mac_reset() {
        let mut mac = MacUnit::new();
        mac.execute_accumulate(1.0, 1.0);
        mac.reset();
        assert_eq!(mac.op_count(), 0);
        assert!((mac.accumulator() - 0.0).abs() < 1e-10);
    }
}
