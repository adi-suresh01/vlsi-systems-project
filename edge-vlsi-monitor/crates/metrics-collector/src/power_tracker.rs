/// Per-agent power tracking with sliding window.
///
/// Maintains a time-series of power samples and provides summary
/// statistics for the dashboard. Uses VecDeque for efficient FIFO.

use std::collections::VecDeque;

/// A single power measurement sample.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct PowerSample {
    pub timestamp: chrono::DateTime<chrono::Utc>,
    pub power_w: f64,
    pub mac_ops: u64,
    pub relu_ops: u64,
    pub temperature_c: f64,
}

/// Summary statistics over the tracking window.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct PowerSummary {
    pub avg_power_w: f64,
    pub peak_power_w: f64,
    pub min_power_w: f64,
    pub total_energy_j: f64,
    pub total_mac_ops: u64,
    pub total_relu_ops: u64,
    pub avg_temp_c: f64,
    pub peak_temp_c: f64,
    pub sample_count: usize,
}

/// Sliding-window power tracker.
pub struct PowerTracker {
    samples: VecDeque<PowerSample>,
    max_samples: usize,
}

impl PowerTracker {
    /// Create a tracker with the given sliding window size.
    pub fn new(max_samples: usize) -> Self {
        Self {
            samples: VecDeque::with_capacity(max_samples),
            max_samples,
        }
    }

    /// Record a power sample. Oldest samples are evicted when the window is full.
    pub fn record(&mut self, sample: PowerSample) {
        if self.samples.len() >= self.max_samples {
            self.samples.pop_front();
        }
        self.samples.push_back(sample);
    }

    /// Compute summary statistics over the current window.
    pub fn summary(&self) -> PowerSummary {
        if self.samples.is_empty() {
            return PowerSummary {
                avg_power_w: 0.0,
                peak_power_w: 0.0,
                min_power_w: 0.0,
                total_energy_j: 0.0,
                total_mac_ops: 0,
                total_relu_ops: 0,
                avg_temp_c: 0.0,
                peak_temp_c: 0.0,
                sample_count: 0,
            };
        }

        let n = self.samples.len() as f64;
        let mut total_power = 0.0;
        let mut peak_power = f64::MIN;
        let mut min_power = f64::MAX;
        let mut total_mac = 0u64;
        let mut total_relu = 0u64;
        let mut total_temp = 0.0;
        let mut peak_temp = f64::MIN;
        let mut total_energy = 0.0;

        let mut prev_time: Option<chrono::DateTime<chrono::Utc>> = None;
        for s in &self.samples {
            total_power += s.power_w;
            peak_power = peak_power.max(s.power_w);
            min_power = min_power.min(s.power_w);
            total_mac += s.mac_ops;
            total_relu += s.relu_ops;
            total_temp += s.temperature_c;
            peak_temp = peak_temp.max(s.temperature_c);

            if let Some(prev) = prev_time {
                let dt = (s.timestamp - prev).num_milliseconds() as f64 / 1000.0;
                total_energy += s.power_w * dt;
            }
            prev_time = Some(s.timestamp);
        }

        PowerSummary {
            avg_power_w: total_power / n,
            peak_power_w: peak_power,
            min_power_w: min_power,
            total_energy_j: total_energy,
            total_mac_ops: total_mac,
            total_relu_ops: total_relu,
            avg_temp_c: total_temp / n,
            peak_temp_c: peak_temp,
            sample_count: self.samples.len(),
        }
    }

    /// Get the N most recent samples.
    pub fn recent(&self, n: usize) -> Vec<&PowerSample> {
        self.samples.iter().rev().take(n).collect()
    }

    /// Get all samples.
    pub fn all_samples(&self) -> Vec<&PowerSample> {
        self.samples.iter().collect()
    }

    pub fn clear(&mut self) {
        self.samples.clear();
    }

    pub fn len(&self) -> usize {
        self.samples.len()
    }

    pub fn is_empty(&self) -> bool {
        self.samples.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::Utc;

    fn make_sample(power: f64, mac: u64, temp: f64) -> PowerSample {
        PowerSample {
            timestamp: Utc::now(),
            power_w: power,
            mac_ops: mac,
            relu_ops: 0,
            temperature_c: temp,
        }
    }

    #[test]
    fn test_tracker_basic() {
        let mut tracker = PowerTracker::new(100);
        tracker.record(make_sample(0.01, 1000, 30.0));
        tracker.record(make_sample(0.02, 2000, 35.0));

        let summary = tracker.summary();
        assert_eq!(summary.sample_count, 2);
        assert!((summary.avg_power_w - 0.015).abs() < 1e-10);
        assert!((summary.peak_power_w - 0.02).abs() < 1e-10);
        assert_eq!(summary.total_mac_ops, 3000);
    }

    #[test]
    fn test_tracker_sliding_window() {
        let mut tracker = PowerTracker::new(3);
        for i in 1..=5 {
            tracker.record(make_sample(i as f64, 0, 25.0));
        }
        assert_eq!(tracker.len(), 3);

        let summary = tracker.summary();
        // Should have samples 3, 4, 5
        assert!((summary.avg_power_w - 4.0).abs() < 1e-10);
    }

    #[test]
    fn test_tracker_empty() {
        let tracker = PowerTracker::new(10);
        let summary = tracker.summary();
        assert_eq!(summary.sample_count, 0);
        assert!((summary.avg_power_w - 0.0).abs() < 1e-10);
    }
}
