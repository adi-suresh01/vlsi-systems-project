/// HDR-style histogram for latency tracking.
///
/// Provides O(1) recording and O(n_buckets) percentile queries.
/// Designed for sub-microsecond latency tracking of inference operations.

/// A snapshot of histogram statistics.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct HistogramSnapshot {
    pub min: u64,
    pub max: u64,
    pub mean: f64,
    pub p50: u64,
    pub p90: u64,
    pub p95: u64,
    pub p99: u64,
    pub p999: u64,
    pub count: u64,
    pub sum: u64,
}

/// Logarithmic histogram with configurable range.
pub struct HdrHistogram {
    /// Bucket counts. Index i covers values in [bucket_min(i), bucket_min(i+1)).
    counts: Vec<u64>,
    /// Number of significant digits of precision.
    precision: u32,
    /// Total values recorded.
    total_count: u64,
    /// Sum of all recorded values.
    sum: u64,
    /// Minimum recorded value.
    min_value: u64,
    /// Maximum recorded value.
    max_value: u64,
    /// Maximum trackable value.
    max_trackable: u64,
}

impl HdrHistogram {
    /// Create a histogram that can track values up to `max_trackable_value`
    /// with `precision` significant digits (1-3).
    pub fn new(max_trackable_value: u64, precision: u32) -> Self {
        let precision = precision.clamp(1, 3);
        let bucket_count = compute_bucket_count(max_trackable_value, precision);

        Self {
            counts: vec![0; bucket_count],
            precision,
            total_count: 0,
            sum: 0,
            min_value: u64::MAX,
            max_value: 0,
            max_trackable: max_trackable_value,
        }
    }

    /// Record a single value.
    pub fn record(&mut self, value: u64) {
        let idx = self.value_to_index(value);
        if idx < self.counts.len() {
            self.counts[idx] += 1;
        } else {
            // Overflow: clamp to last bucket
            if let Some(last) = self.counts.last_mut() {
                *last += 1;
            }
        }
        self.total_count += 1;
        self.sum += value;
        self.min_value = self.min_value.min(value);
        self.max_value = self.max_value.max(value);
    }

    /// Record a value N times.
    pub fn record_n(&mut self, value: u64, count: u64) {
        for _ in 0..count {
            self.record(value);
        }
    }

    /// Get the value at a given percentile (0.0 - 100.0).
    pub fn percentile(&self, p: f64) -> u64 {
        if self.total_count == 0 {
            return 0;
        }

        let target_count = ((p / 100.0) * self.total_count as f64).ceil() as u64;
        let mut cumulative = 0u64;

        for (idx, &count) in self.counts.iter().enumerate() {
            cumulative += count;
            if cumulative >= target_count {
                return self.index_to_value(idx);
            }
        }

        self.max_value
    }

    /// Get a full statistics snapshot.
    pub fn snapshot(&self) -> HistogramSnapshot {
        HistogramSnapshot {
            min: if self.total_count > 0 {
                self.min_value
            } else {
                0
            },
            max: self.max_value,
            mean: if self.total_count > 0 {
                self.sum as f64 / self.total_count as f64
            } else {
                0.0
            },
            p50: self.percentile(50.0),
            p90: self.percentile(90.0),
            p95: self.percentile(95.0),
            p99: self.percentile(99.0),
            p999: self.percentile(99.9),
            count: self.total_count,
            sum: self.sum,
        }
    }

    /// Reset all counters.
    pub fn reset(&mut self) {
        self.counts.fill(0);
        self.total_count = 0;
        self.sum = 0;
        self.min_value = u64::MAX;
        self.max_value = 0;
    }

    pub fn count(&self) -> u64 {
        self.total_count
    }

    fn value_to_index(&self, value: u64) -> usize {
        if value == 0 {
            return 0;
        }
        let value = value.min(self.max_trackable);
        // Logarithmic bucketing
        let log_val = (value as f64).log2();
        let scale = 10_u64.pow(self.precision) as f64;
        (log_val * scale / (self.max_trackable as f64).log2()).min((self.counts.len() - 1) as f64)
            as usize
    }

    fn index_to_value(&self, index: usize) -> u64 {
        if index == 0 {
            return 0;
        }
        let scale = 10_u64.pow(self.precision) as f64;
        let log_max = (self.max_trackable as f64).log2();
        let log_val = index as f64 * log_max / scale;
        2.0_f64.powf(log_val) as u64
    }
}

fn compute_bucket_count(max_value: u64, precision: u32) -> usize {
    let scale = 10_usize.pow(precision);
    // Enough buckets to cover the range with the given precision
    (scale + 1).max(64)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_histogram_basic() {
        let mut hist = HdrHistogram::new(1_000_000, 2);
        for v in [100, 200, 300, 400, 500] {
            hist.record(v);
        }
        assert_eq!(hist.count(), 5);

        let snap = hist.snapshot();
        assert_eq!(snap.min, 100);
        assert_eq!(snap.max, 500);
        assert!((snap.mean - 300.0).abs() < 1e-10);
    }

    #[test]
    fn test_histogram_percentiles() {
        let mut hist = HdrHistogram::new(10000, 2);
        for v in 1..=100 {
            hist.record(v);
        }

        let snap = hist.snapshot();
        assert!(snap.p50 > 0);
        assert!(snap.p99 > snap.p50);
    }

    #[test]
    fn test_histogram_empty() {
        let hist = HdrHistogram::new(1000, 2);
        let snap = hist.snapshot();
        assert_eq!(snap.count, 0);
        assert_eq!(snap.min, 0);
        assert!((snap.mean - 0.0).abs() < 1e-10);
    }

    #[test]
    fn test_histogram_reset() {
        let mut hist = HdrHistogram::new(1000, 2);
        hist.record(100);
        hist.record(200);
        hist.reset();
        assert_eq!(hist.count(), 0);
    }
}
