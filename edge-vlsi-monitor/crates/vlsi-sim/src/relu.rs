/// ReLU activation unit simulation.
///
/// Models the SystemVerilog `relu_unit.sv`:
///   - Combinational logic: output = max(0, input)
///   - 1-cycle latency in pipeline context
///   - Matches C++ `simulate_relu_hw(x) -> std::max(0.0, x)`

#[derive(Debug, Clone)]
pub struct ReluUnit {
    op_count: u64,
    latency_cycles: u32,
}

impl ReluUnit {
    pub fn new() -> Self {
        Self {
            op_count: 0,
            latency_cycles: 1, // 1 cycle per ReLU, from hardware_sim.py line 67
        }
    }

    /// Apply ReLU activation: max(0, x).
    #[inline]
    pub fn execute(&mut self, x: f64) -> f64 {
        self.op_count += 1;
        x.max(0.0)
    }

    /// Apply ReLU in-place to a batch of values.
    pub fn execute_batch(&mut self, data: &mut [f64]) {
        for val in data.iter_mut() {
            *val = val.max(0.0);
            self.op_count += 1;
        }
    }

    pub fn op_count(&self) -> u64 {
        self.op_count
    }

    pub fn latency_cycles(&self) -> u32 {
        self.latency_cycles
    }

    pub fn reset(&mut self) {
        self.op_count = 0;
    }
}

impl Default for ReluUnit {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_relu_positive() {
        let mut relu = ReluUnit::new();
        assert!((relu.execute(5.0) - 5.0).abs() < 1e-10);
    }

    #[test]
    fn test_relu_zero() {
        let mut relu = ReluUnit::new();
        assert!((relu.execute(0.0) - 0.0).abs() < 1e-10);
    }

    #[test]
    fn test_relu_negative() {
        let mut relu = ReluUnit::new();
        assert!((relu.execute(-3.0) - 0.0).abs() < 1e-10);
    }

    #[test]
    fn test_relu_batch() {
        let mut relu = ReluUnit::new();
        let mut data = vec![-1.0, 0.0, 1.0, -5.0, 3.14];
        relu.execute_batch(&mut data);
        assert_eq!(data, vec![0.0, 0.0, 1.0, 0.0, 3.14]);
        assert_eq!(relu.op_count(), 5);
    }

    #[test]
    fn test_relu_op_count() {
        let mut relu = ReluUnit::new();
        relu.execute(1.0);
        relu.execute(-1.0);
        relu.execute(0.0);
        assert_eq!(relu.op_count(), 3);
    }
}
