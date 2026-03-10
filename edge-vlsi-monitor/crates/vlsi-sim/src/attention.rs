/// Transformer attention operation calculator.
///
/// Computes MAC and activation counts for self-attention given model
/// dimensions and sequence length. Pure math, no model files needed.
///
/// For a single layer with sequence length n, hidden dim d, h heads, d_k = d/h:
///   Q,K,V projections:   3 * n * d * d MACs
///   Attention scores:    n * n * d MACs  (the quadratic term)
///   Value aggregation:   n * n * d MACs  (also quadratic)
///   Output projection:   n * d * d MACs
///   FFN (two layers):    2 * n * d * d_ff MACs

use crate::workload::WorkloadProfile;

/// Transformer architecture parameters.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct TransformerConfig {
    pub name: String,
    pub num_layers: usize,
    pub d_model: usize,
    pub num_heads: usize,
    /// FFN intermediate dimension (typically 4 * d_model)
    pub d_ff: usize,
    pub vocab_size: usize,
}

impl TransformerConfig {
    pub fn tinybert() -> Self {
        Self {
            name: "TinyBERT".into(),
            num_layers: 6,
            d_model: 312,
            num_heads: 12,
            d_ff: 1200,
            vocab_size: 30522,
        }
    }

    pub fn distilgpt2() -> Self {
        Self {
            name: "DistilGPT-2".into(),
            num_layers: 6,
            d_model: 768,
            num_heads: 12,
            d_ff: 3072,
            vocab_size: 50257,
        }
    }

    pub fn bert_base() -> Self {
        Self {
            name: "BERT-Base".into(),
            num_layers: 12,
            d_model: 768,
            num_heads: 12,
            d_ff: 3072,
            vocab_size: 30522,
        }
    }

    pub fn d_k(&self) -> usize {
        self.d_model / self.num_heads
    }
}

/// Per-layer operation breakdown.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct AttentionLayerOps {
    pub qkv_projection_macs: u64,
    pub attention_score_macs: u64,
    pub softmax_activations: u64,
    pub value_aggregation_macs: u64,
    pub output_projection_macs: u64,
    pub ffn_macs: u64,
    pub layernorm_activations: u64,
    pub ffn_activations: u64,
    pub total_macs: u64,
    pub total_activations: u64,
}

/// Full model operation breakdown at a specific sequence length.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct TransformerOps {
    pub config_name: String,
    pub seq_length: usize,
    pub num_layers: usize,
    pub embedding_macs: u64,
    pub per_layer: AttentionLayerOps,
    pub total_macs: u64,
    pub total_activations: u64,
    /// MACs from the O(n^2) attention components only
    pub attention_quadratic_macs: u64,
    /// MACs from linear components (projections, FFN, embedding)
    pub linear_macs: u64,
}

pub fn calculate_transformer_ops(config: &TransformerConfig, seq_length: usize) -> TransformerOps {
    let n = seq_length as u64;
    let d = config.d_model as u64;
    let h = config.num_heads as u64;
    let d_ff = config.d_ff as u64;

    // Q, K, V projections: each is [n, d] x [d, d] = n * d^2. Three of them.
    let qkv_macs = 3 * n * d * d;

    // Attention scores: Q * K^T per head: [n, d_k] x [d_k, n] = n^2 * d_k per head
    // Across h heads with h * d_k = d: n^2 * d total
    let attn_score_macs = n * n * d;

    // Softmax: n elements per query position, n positions, h heads
    let softmax_activations = h * n * n;

    // Value aggregation: attn_weights * V: [n, n] x [n, d_k] per head = n^2 * d total
    let value_agg_macs = n * n * d;

    // Output projection: [n, d] x [d, d] = n * d^2
    let output_proj_macs = n * d * d;

    // FFN: two linear layers through d_ff intermediate
    // [n, d] x [d, d_ff] + [n, d_ff] x [d_ff, d] = 2 * n * d * d_ff
    let ffn_macs = 2 * n * d * d_ff;

    // Layer normalization: 2 per layer (pre-attention, pre-FFN), each over d dims per token
    let layernorm_activations = 2 * n * d;

    // FFN activation (GELU between the two linear layers): n * d_ff
    let ffn_activations = n * d_ff;

    let layer_total_macs = qkv_macs + attn_score_macs + value_agg_macs + output_proj_macs + ffn_macs;
    let layer_total_activations = softmax_activations + layernorm_activations + ffn_activations;

    let per_layer = AttentionLayerOps {
        qkv_projection_macs: qkv_macs,
        attention_score_macs: attn_score_macs,
        softmax_activations,
        value_aggregation_macs: value_agg_macs,
        output_projection_macs: output_proj_macs,
        ffn_macs,
        layernorm_activations,
        ffn_activations,
        total_macs: layer_total_macs,
        total_activations: layer_total_activations,
    };

    let embedding_macs = n * d;
    let num_layers = config.num_layers as u64;
    let total_macs = embedding_macs + num_layers * layer_total_macs;
    let total_activations = num_layers * layer_total_activations;

    let attention_quadratic_macs = num_layers * (attn_score_macs + value_agg_macs);
    let linear_macs = total_macs - attention_quadratic_macs;

    TransformerOps {
        config_name: config.name.clone(),
        seq_length,
        num_layers: config.num_layers,
        embedding_macs,
        per_layer,
        total_macs,
        total_activations,
        attention_quadratic_macs,
        linear_macs,
    }
}

pub fn transformer_ops_to_profile(ops: &TransformerOps, num_inferences: u64) -> WorkloadProfile {
    WorkloadProfile::new(
        &ops.config_name,
        ops.total_macs,
        ops.total_activations,
        num_inferences,
    )
}

pub fn sweep_sequence_lengths(
    config: &TransformerConfig,
    seq_lengths: &[usize],
) -> Vec<TransformerOps> {
    seq_lengths
        .iter()
        .map(|&n| calculate_transformer_ops(config, n))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tinybert_ops() {
        let config = TransformerConfig::tinybert();
        let ops = calculate_transformer_ops(&config, 128);

        // d=312, n=128, d_ff=1200, 6 layers
        // Per-layer QKV: 3 * 128 * 312 * 312 = 37,324,800
        let expected_qkv = 3 * 128 * 312 * 312;
        assert_eq!(ops.per_layer.qkv_projection_macs, expected_qkv as u64);

        // Per-layer attention scores: 128 * 128 * 312 = 5,111,808
        let expected_attn = 128 * 128 * 312;
        assert_eq!(ops.per_layer.attention_score_macs, expected_attn as u64);

        assert!(ops.total_macs > 0);
        assert_eq!(ops.num_layers, 6);
        assert_eq!(ops.total_macs, ops.attention_quadratic_macs + ops.linear_macs);
    }

    #[test]
    fn test_quadratic_scaling() {
        let config = TransformerConfig::distilgpt2();
        let ops_128 = calculate_transformer_ops(&config, 128);
        let ops_256 = calculate_transformer_ops(&config, 256);

        // Doubling sequence length should roughly quadruple the quadratic component
        let ratio = ops_256.attention_quadratic_macs as f64
            / ops_128.attention_quadratic_macs as f64;
        assert!((ratio - 4.0).abs() < 0.01);
    }

    #[test]
    fn test_linear_vs_quadratic_crossover() {
        let config = TransformerConfig::distilgpt2();

        // At short sequences, linear (projections + FFN) should dominate
        let ops_32 = calculate_transformer_ops(&config, 32);
        assert!(ops_32.linear_macs > ops_32.attention_quadratic_macs);

        // At long sequences, quadratic (attention) should dominate
        let ops_8192 = calculate_transformer_ops(&config, 8192);
        assert!(ops_8192.attention_quadratic_macs > ops_8192.linear_macs);
    }

    #[test]
    fn test_sweep() {
        let config = TransformerConfig::tinybert();
        let lengths = vec![64, 128, 256, 512];
        let results = sweep_sequence_lengths(&config, &lengths);
        assert_eq!(results.len(), 4);
        assert!(results[0].total_macs < results[1].total_macs);
        assert!(results[1].total_macs < results[2].total_macs);
        assert!(results[2].total_macs < results[3].total_macs);
    }

    #[test]
    fn test_profile_conversion() {
        let config = TransformerConfig::tinybert();
        let ops = calculate_transformer_ops(&config, 128);
        let profile = transformer_ops_to_profile(&ops, 5);
        assert_eq!(profile.mac_ops, ops.total_macs);
        assert_eq!(profile.activation_ops, ops.total_activations);
        assert_eq!(profile.num_inferences, 5);
        assert_eq!(profile.total_mac_ops(), ops.total_macs * 5);
    }
}
