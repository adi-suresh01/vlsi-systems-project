//! VLSI Hardware Simulation Engine
//!
//! Rust port of the Python/C++ hardware simulation from the vlsi-ai-integration project.
//! Provides cycle-accurate MAC/ReLU simulation, convolution pipelines, DVFS-aware power
//! modeling, and thermal feedback for edge AI workload analysis.

pub mod mac;
pub mod relu;
pub mod conv;
pub mod pipeline;
pub mod power;
pub mod thermal;
pub mod workload;
pub mod attention;

// Re-export primary types
pub use conv::{ConvConfig, ConvEngine, ConvResult};
pub use mac::MacUnit;
pub use pipeline::{
    generate_test_samples, run_simulation, run_simulation_parallel, run_simulation_from_profile,
    PipelineConfig, SimulationResult,
};
pub use power::{DvfsConfig, PowerBreakdown, PowerModel};
pub use relu::ReluUnit;
pub use thermal::{ThermalConfig, ThermalModel, ThermalState};
pub use workload::WorkloadProfile;
pub use attention::{
    TransformerConfig, TransformerOps, AttentionLayerOps,
    calculate_transformer_ops, transformer_ops_to_profile, sweep_sequence_lengths,
};
