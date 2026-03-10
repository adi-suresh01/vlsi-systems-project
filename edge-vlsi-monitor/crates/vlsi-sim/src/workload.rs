/// Generic workload profile for any neural network model.
///
/// Decouples the simulation engine from convolution-specific operation counting.
/// Represents CNN, transformer, or mixed workloads as raw operation counts
/// that feed directly into the power and thermal models.

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct WorkloadProfile {
    pub model_name: String,
    /// Multiply-accumulate operations per single inference
    pub mac_ops: u64,
    /// Activation operations per single inference (ReLU, GELU, softmax units)
    pub activation_ops: u64,
    /// Number of inferences in this workload
    pub num_inferences: u64,
    /// Memory read bytes per inference (for future bandwidth modeling)
    pub memory_read_bytes: Option<u64>,
    /// Memory write bytes per inference
    pub memory_write_bytes: Option<u64>,
}

impl WorkloadProfile {
    pub fn new(
        model_name: impl Into<String>,
        mac_ops: u64,
        activation_ops: u64,
        num_inferences: u64,
    ) -> Self {
        Self {
            model_name: model_name.into(),
            mac_ops,
            activation_ops,
            num_inferences,
            memory_read_bytes: None,
            memory_write_bytes: None,
        }
    }

    pub fn with_memory(mut self, read_bytes: u64, write_bytes: u64) -> Self {
        self.memory_read_bytes = Some(read_bytes);
        self.memory_write_bytes = Some(write_bytes);
        self
    }

    pub fn total_mac_ops(&self) -> u64 {
        self.mac_ops * self.num_inferences
    }

    pub fn total_activation_ops(&self) -> u64 {
        self.activation_ops * self.num_inferences
    }

    pub fn total_ops(&self) -> u64 {
        self.total_mac_ops() + self.total_activation_ops()
    }

    /// Build a WorkloadProfile from the existing convolution results.
    pub fn from_conv(samples: u64, macs_per_sample: u64, relus_per_sample: u64) -> Self {
        Self {
            model_name: "synthetic-conv".into(),
            mac_ops: macs_per_sample,
            activation_ops: relus_per_sample,
            num_inferences: samples,
            memory_read_bytes: None,
            memory_write_bytes: None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_workload_profile_totals() {
        let profile = WorkloadProfile::new("test", 1000, 100, 5);
        assert_eq!(profile.total_mac_ops(), 5000);
        assert_eq!(profile.total_activation_ops(), 500);
        assert_eq!(profile.total_ops(), 5500);
    }

    #[test]
    fn test_from_conv_matches_synthetic() {
        let profile = WorkloadProfile::from_conv(3, 3528, 8);
        assert_eq!(profile.total_mac_ops(), 3 * 3528);
        assert_eq!(profile.total_activation_ops(), 3 * 8);
        assert_eq!(profile.model_name, "synthetic-conv");
    }

    #[test]
    fn test_with_memory() {
        let profile = WorkloadProfile::new("test", 100, 10, 1)
            .with_memory(4096, 2048);
        assert_eq!(profile.memory_read_bytes, Some(4096));
        assert_eq!(profile.memory_write_bytes, Some(2048));
    }
}
