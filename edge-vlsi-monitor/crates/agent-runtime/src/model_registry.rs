/// Model registry for managing known model configurations.
///
/// Stores model metadata and cached workload profiles. Built-in entries
/// cover common edge AI models. Custom models can be registered from
/// ONNX files or manual configuration.

use std::collections::HashMap;
use std::path::PathBuf;

use vlsi_sim::attention::{self, TransformerConfig};
use vlsi_sim::WorkloadProfile;

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct ModelEntry {
    pub name: String,
    pub path: Option<PathBuf>,
    pub input_shape: Vec<usize>,
    pub arch_type: ModelArchType,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub enum ModelArchType {
    Cnn,
    Transformer(TransformerConfig),
    Custom,
}

pub struct ModelRegistry {
    entries: HashMap<String, ModelEntry>,
}

impl ModelRegistry {
    pub fn new() -> Self {
        let mut registry = Self {
            entries: HashMap::new(),
        };
        registry.register_builtins();
        registry
    }

    fn register_builtins(&mut self) {
        self.entries.insert(
            "synthetic-conv".into(),
            ModelEntry {
                name: "synthetic-conv".into(),
                path: None,
                input_shape: vec![1, 1, 28, 28],
                arch_type: ModelArchType::Cnn,
            },
        );

        self.entries.insert(
            "mobilenetv2".into(),
            ModelEntry {
                name: "MobileNetV2".into(),
                path: None,
                input_shape: vec![1, 3, 224, 224],
                arch_type: ModelArchType::Cnn,
            },
        );

        self.entries.insert(
            "tinybert".into(),
            ModelEntry {
                name: "TinyBERT".into(),
                path: None,
                input_shape: vec![1, 128],
                arch_type: ModelArchType::Transformer(TransformerConfig::tinybert()),
            },
        );

        self.entries.insert(
            "distilgpt2".into(),
            ModelEntry {
                name: "DistilGPT-2".into(),
                path: None,
                input_shape: vec![1, 128],
                arch_type: ModelArchType::Transformer(TransformerConfig::distilgpt2()),
            },
        );

        self.entries.insert(
            "bert-base".into(),
            ModelEntry {
                name: "BERT-Base".into(),
                path: None,
                input_shape: vec![1, 128],
                arch_type: ModelArchType::Transformer(TransformerConfig::bert_base()),
            },
        );
    }

    pub fn get(&self, name: &str) -> Option<&ModelEntry> {
        self.entries.get(name)
    }

    /// Build a workload profile for a model at a given sequence length and inference count.
    /// For transformers, computes attention ops from the config.
    /// For CNNs, returns a fixed profile based on known op counts.
    pub fn workload_profile(
        &self,
        name: &str,
        seq_length: Option<usize>,
        num_inferences: u64,
    ) -> Option<WorkloadProfile> {
        let entry = self.entries.get(name)?;
        match &entry.arch_type {
            ModelArchType::Transformer(config) => {
                let seq_len = seq_length.unwrap_or(128);
                let ops = attention::calculate_transformer_ops(config, seq_len);
                Some(attention::transformer_ops_to_profile(&ops, num_inferences))
            }
            ModelArchType::Cnn => {
                if name == "synthetic-conv" {
                    Some(WorkloadProfile::from_conv(num_inferences, 3528, 8))
                } else if name == "mobilenetv2" {
                    Some(WorkloadProfile::new("MobileNetV2", 300_000_000, 3_000_000, num_inferences))
                } else {
                    None
                }
            }
            ModelArchType::Custom => None,
        }
    }

    pub fn register(&mut self, key: String, entry: ModelEntry) {
        self.entries.insert(key, entry);
    }

    pub fn list_names(&self) -> Vec<&str> {
        let mut names: Vec<&str> = self.entries.keys().map(|s| s.as_str()).collect();
        names.sort();
        names
    }
}

impl Default for ModelRegistry {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_registry_builtins() {
        let registry = ModelRegistry::new();
        assert!(registry.get("tinybert").is_some());
        assert!(registry.get("distilgpt2").is_some());
        assert!(registry.get("mobilenetv2").is_some());
        assert!(registry.get("synthetic-conv").is_some());
        assert!(registry.get("nonexistent").is_none());
    }

    #[test]
    fn test_transformer_profile() {
        let registry = ModelRegistry::new();
        let profile = registry.workload_profile("tinybert", Some(256), 1).unwrap();
        assert!(profile.mac_ops > 0);
        assert!(profile.activation_ops > 0);
        assert_eq!(profile.num_inferences, 1);
    }

    #[test]
    fn test_cnn_profile() {
        let registry = ModelRegistry::new();
        let profile = registry.workload_profile("mobilenetv2", None, 5).unwrap();
        assert_eq!(profile.mac_ops, 300_000_000);
        assert_eq!(profile.num_inferences, 5);
    }

    #[test]
    fn test_synthetic_conv_profile() {
        let registry = ModelRegistry::new();
        let profile = registry.workload_profile("synthetic-conv", None, 3).unwrap();
        assert_eq!(profile.mac_ops, 3528);
        assert_eq!(profile.activation_ops, 8);
        assert_eq!(profile.total_mac_ops(), 3528 * 3);
    }

    #[test]
    fn test_sequence_length_affects_profile() {
        let registry = ModelRegistry::new();
        let short = registry.workload_profile("distilgpt2", Some(128), 1).unwrap();
        let long = registry.workload_profile("distilgpt2", Some(512), 1).unwrap();
        assert!(long.mac_ops > short.mac_ops);
    }
}
