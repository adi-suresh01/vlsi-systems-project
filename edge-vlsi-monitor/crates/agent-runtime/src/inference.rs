/// ONNX model inference via tract.
///
/// Replaces the Python TFLite inference path (tflite_converter.py, model_utils.py).
/// Uses tract-onnx for zero-dependency native inference on edge devices.

use std::path::Path;
use std::time::Instant;

use anyhow::{Context, Result};
use tract_onnx::prelude::*;

/// Result of a single inference run.
#[derive(Debug, Clone, serde::Serialize)]
pub struct InferenceResult {
    /// Raw model output (batch x classes)
    pub predictions: Vec<Vec<f32>>,
    /// Inference latency in microseconds
    pub latency_us: u64,
    /// Predicted class per sample (argmax)
    pub predicted_classes: Vec<usize>,
}

/// ONNX model wrapper with tract runtime.
pub struct OnnxModel {
    model: SimplePlan<TypedFact, Box<dyn TypedOp>, Graph<TypedFact, Box<dyn TypedOp>>>,
    input_shape: Vec<usize>,
}

impl OnnxModel {
    /// Load an ONNX model from disk.
    pub fn load(path: &Path) -> Result<Self> {
        tracing::info!("Loading ONNX model from {}", path.display());

        let model = tract_onnx::onnx()
            .model_for_path(path)
            .context("Failed to load ONNX model")?
            .into_optimized()
            .context("Failed to optimize model")?
            .into_runnable()
            .context("Failed to create runnable model")?;

        // Extract input shape from the model
        let input_fact = model.model().input_fact(0)?;
        let input_shape: Vec<usize> = input_fact
            .shape
            .iter()
            .map(|d| d.to_i64().unwrap_or(1) as usize)
            .collect();

        tracing::info!("Model loaded. Input shape: {:?}", input_shape);

        Ok(Self {
            model,
            input_shape,
        })
    }

    /// Run inference on a batch of inputs.
    ///
    /// Input should be a flat f32 slice matching the model's expected input shape.
    pub fn run(&self, input: &[f32], batch_size: usize) -> Result<InferenceResult> {
        let start = Instant::now();

        // Build input tensor
        let mut shape = self.input_shape.clone();
        if !shape.is_empty() {
            shape[0] = batch_size;
        }

        let input_tensor: Tensor = tract_ndarray::Array::from_shape_vec(
            tract_ndarray::IxDyn(&shape),
            input.to_vec(),
        )
        .context("Failed to create input tensor")?
        .into();

        let result = self
            .model
            .run(tvec!(input_tensor.into()))
            .context("Inference failed")?;

        let latency_us = start.elapsed().as_micros() as u64;

        // Extract output
        let output = result[0]
            .to_array_view::<f32>()
            .context("Failed to read output tensor")?;

        let output_shape = output.shape();
        let num_classes = if output_shape.len() >= 2 {
            output_shape[1]
        } else {
            output_shape[0]
        };

        let mut predictions = Vec::with_capacity(batch_size);
        let mut predicted_classes = Vec::with_capacity(batch_size);

        for b in 0..batch_size {
            let offset = b * num_classes;
            let end = (offset + num_classes).min(output.len());
            let row: Vec<f32> = output
                .as_slice()
                .unwrap_or(&[])[offset..end]
                .to_vec();

            let class = row
                .iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
                .map(|(idx, _)| idx)
                .unwrap_or(0);

            predicted_classes.push(class);
            predictions.push(row);
        }

        Ok(InferenceResult {
            predictions,
            latency_us,
            predicted_classes,
        })
    }

    pub fn input_shape(&self) -> &[usize] {
        &self.input_shape
    }
}
