//! Metrics Collection Library
//!
//! Lock-free ring buffers, HDR histograms, and per-agent power tracking
//! for real-time edge AI monitoring. Designed for minimal overhead on
//! the inference hot path.

pub mod histogram;
pub mod power_tracker;
pub mod ring_buffer;

pub use histogram::{HdrHistogram, HistogramSnapshot};
pub use power_tracker::{PowerSample, PowerSummary, PowerTracker};
pub use ring_buffer::RingBuffer;

/// Combined metrics state for a single agent.
pub struct AgentMetrics {
    /// Latency histogram (microseconds)
    pub latency_hist: HdrHistogram,
    /// Power tracking with sliding window
    pub power: PowerTracker,
    /// Recent inference latencies ring buffer (microseconds)
    pub inference_buffer: RingBuffer<f64>,
    /// Total inferences completed
    pub inference_count: u64,
}

impl AgentMetrics {
    pub fn new() -> Self {
        Self {
            latency_hist: HdrHistogram::new(10_000_000, 2), // up to 10s in µs
            power: PowerTracker::new(1000),                  // last 1000 samples
            inference_buffer: RingBuffer::new(256),          // last 255 latencies
            inference_count: 0,
        }
    }

    /// Record a completed inference with its latency in microseconds.
    pub fn record_inference(&mut self, latency_us: u64) {
        self.latency_hist.record(latency_us);
        self.inference_buffer.push_overwrite(latency_us as f64);
        self.inference_count += 1;
    }

    /// Record a power measurement.
    pub fn record_power(&mut self, sample: PowerSample) {
        self.power.record(sample);
    }

    pub fn reset(&mut self) {
        self.latency_hist.reset();
        self.power.clear();
        self.inference_count = 0;
    }
}

impl Default for AgentMetrics {
    fn default() -> Self {
        Self::new()
    }
}
