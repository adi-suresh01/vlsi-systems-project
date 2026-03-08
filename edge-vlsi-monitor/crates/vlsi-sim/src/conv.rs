/// Convolution engine simulation.
///
/// Ports the convolution loop from `hardware_sim.py` lines 36-57 and
/// `hardware_simulator.cpp` lines 38-89. Performs 2D convolution with
/// random kernels, stride-based patch extraction, and ReLU activation.

use ndarray::Array2;
use rand::Rng;

use crate::mac::MacUnit;
use crate::relu::ReluUnit;

/// Configuration for the convolution engine.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct ConvConfig {
    /// Number of output feature maps (default: 8, from hardware_sim.py line 40)
    pub num_kernels: usize,
    /// Spatial size of each kernel (default: 3 for 3x3)
    pub kernel_size: usize,
    /// Stride for patch extraction (default: 4, from hardware_sim.py line 45)
    pub stride: usize,
    /// Kernel weight scale (default: 0.1, from hardware_sim.py line 41)
    pub weight_scale: f64,
}

impl Default for ConvConfig {
    fn default() -> Self {
        Self {
            num_kernels: 8,
            kernel_size: 3,
            stride: 4,
            weight_scale: 0.1,
        }
    }
}

/// Convolution engine combining MAC and ReLU units.
#[derive(Debug)]
pub struct ConvEngine {
    mac: MacUnit,
    relu: ReluUnit,
    config: ConvConfig,
}

/// Per-sample convolution result.
#[derive(Debug, Clone, serde::Serialize)]
pub struct ConvResult {
    pub output: f64,
    pub feature_maps: Vec<f64>,
    pub mac_ops: u64,
    pub relu_ops: u64,
}

impl ConvEngine {
    pub fn new(config: ConvConfig) -> Self {
        Self {
            mac: MacUnit::new(),
            relu: ReluUnit::new(),
            config,
        }
    }

    /// Process a single 2D sample through convolution + ReLU.
    ///
    /// Direct port of the inner loop from hardware_sim.py lines 40-57:
    /// ```python
    /// for kernel_idx in range(8):
    ///     kernel = np.random.randn(3, 3) * 0.1
    ///     for y in range(0, sample.shape[0]-2, 4):
    ///         for x in range(0, sample.shape[1]-2, 4):
    ///             patch = sample[y:y+3, x:x+3]
    ///             conv_sum += np.sum(patch * kernel)
    ///             total_mac_ops += 9
    ///     relu_result = max(0, conv_sum)
    /// ```
    pub fn process_sample(&mut self, sample: &Array2<f64>) -> ConvResult {
        let mut rng = rand::thread_rng();
        let (rows, cols) = sample.dim();
        let ks = self.config.kernel_size;
        let stride = self.config.stride;

        let mac_before = self.mac.op_count();
        let relu_before = self.relu.op_count();

        let mut feature_maps = Vec::with_capacity(self.config.num_kernels);

        for _kernel_idx in 0..self.config.num_kernels {
            // Generate random kernel weights (matching Python: np.random.randn(3,3) * 0.1)
            let kernel: Vec<Vec<f64>> = (0..ks)
                .map(|_| {
                    (0..ks)
                        .map(|_| rng.gen::<f64>() * 2.0 - 1.0) // approximate randn
                        .map(|v| v * self.config.weight_scale)
                        .collect()
                })
                .collect();

            let mut conv_sum = 0.0;

            // Strided convolution
            let mut y = 0;
            while y + ks <= rows {
                let mut x = 0;
                while x + ks <= cols {
                    // 3x3 patch MAC operations
                    for ky in 0..ks {
                        for kx in 0..ks {
                            let pixel = sample[[y + ky, x + kx]];
                            let weight = kernel[ky][kx];
                            conv_sum = self.mac.execute(pixel, weight, conv_sum);
                        }
                    }
                    x += stride;
                }
                y += stride;
            }

            // ReLU per feature map
            let relu_result = self.relu.execute(conv_sum);
            feature_maps.push(relu_result);
        }

        let output = feature_maps.iter().sum::<f64>() / feature_maps.len() as f64;

        ConvResult {
            output,
            feature_maps,
            mac_ops: self.mac.op_count() - mac_before,
            relu_ops: self.relu.op_count() - relu_before,
        }
    }

    /// Process a batch of 2D samples. Returns per-sample results.
    pub fn process_batch(&mut self, samples: &[Array2<f64>]) -> Vec<ConvResult> {
        samples.iter().map(|s| self.process_sample(s)).collect()
    }

    pub fn total_mac_ops(&self) -> u64 {
        self.mac.op_count()
    }

    pub fn total_relu_ops(&self) -> u64 {
        self.relu.op_count()
    }

    pub fn reset(&mut self) {
        self.mac.reset();
        self.relu.reset();
    }

    pub fn config(&self) -> &ConvConfig {
        &self.config
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array2;

    #[test]
    fn test_conv_sample_28x28() {
        let config = ConvConfig::default(); // 8 kernels, 3x3, stride 4
        let mut engine = ConvEngine::new(config);

        // 28x28 MNIST-like sample
        let sample = Array2::ones((28, 28));
        let result = engine.process_sample(&sample);

        assert_eq!(result.feature_maps.len(), 8);
        assert!(result.mac_ops > 0);
        assert_eq!(result.relu_ops, 8); // 1 ReLU per feature map

        // With stride=4 on 28x28, valid positions per axis: (28-3)/4 + 1 = 7
        // Per kernel: 7 * 7 patches * 9 MACs = 441
        // Total: 8 * 441 = 3528
        let expected_positions_per_axis = (28 - 3) / 4 + 1; // 7
        let expected_macs = 8 * expected_positions_per_axis * expected_positions_per_axis * 9;
        assert_eq!(result.mac_ops, expected_macs as u64);
    }

    #[test]
    fn test_conv_batch() {
        let config = ConvConfig::default();
        let mut engine = ConvEngine::new(config);

        let samples: Vec<Array2<f64>> = (0..5)
            .map(|_| Array2::ones((28, 28)))
            .collect();

        let results = engine.process_batch(&samples);
        assert_eq!(results.len(), 5);

        let total_mac: u64 = results.iter().map(|r| r.mac_ops).sum();
        assert_eq!(total_mac, engine.total_mac_ops());
    }
}
