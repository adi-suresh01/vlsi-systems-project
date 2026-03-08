/// Lock-free single-producer single-consumer ring buffer.
///
/// Designed for the hot path: inference loops push latency/power samples,
/// the dashboard reader drains them periodically. Uses atomic head/tail
/// pointers for zero-contention operation.

use std::sync::atomic::{AtomicUsize, Ordering};

pub struct RingBuffer<T: Copy + Default> {
    buf: Box<[T]>,
    capacity: usize,
    head: AtomicUsize, // writer position
    tail: AtomicUsize, // reader position
}

impl<T: Copy + Default> RingBuffer<T> {
    /// Create a ring buffer with the given capacity.
    pub fn new(capacity: usize) -> Self {
        let capacity = capacity.max(1);
        Self {
            buf: vec![T::default(); capacity].into_boxed_slice(),
            capacity,
            head: AtomicUsize::new(0),
            tail: AtomicUsize::new(0),
        }
    }

    /// Push an item. Returns false if the buffer is full (item is dropped).
    pub fn push(&self, item: T) -> bool {
        let head = self.head.load(Ordering::Relaxed);
        let tail = self.tail.load(Ordering::Acquire);
        let next_head = (head + 1) % self.capacity;

        if next_head == tail {
            return false; // Full
        }

        // SAFETY: single-producer guarantee — only one thread writes.
        // We cast away const to write into the slot.
        let buf_ptr = self.buf.as_ptr() as *mut T;
        unsafe {
            buf_ptr.add(head).write(item);
        }

        self.head.store(next_head, Ordering::Release);
        true
    }

    /// Push an item, overwriting the oldest if full.
    pub fn push_overwrite(&self, item: T) {
        let head = self.head.load(Ordering::Relaxed);
        let tail = self.tail.load(Ordering::Acquire);
        let next_head = (head + 1) % self.capacity;

        let buf_ptr = self.buf.as_ptr() as *mut T;
        unsafe {
            buf_ptr.add(head).write(item);
        }

        if next_head == tail {
            // Advance tail to drop oldest
            self.tail
                .store((tail + 1) % self.capacity, Ordering::Release);
        }

        self.head.store(next_head, Ordering::Release);
    }

    /// Pop an item from the buffer. Returns None if empty.
    pub fn pop(&self) -> Option<T> {
        let tail = self.tail.load(Ordering::Relaxed);
        let head = self.head.load(Ordering::Acquire);

        if tail == head {
            return None; // Empty
        }

        let item = self.buf[tail];
        self.tail
            .store((tail + 1) % self.capacity, Ordering::Release);
        Some(item)
    }

    /// Drain all available items into a Vec.
    pub fn drain(&self) -> Vec<T> {
        let mut items = Vec::new();
        while let Some(item) = self.pop() {
            items.push(item);
        }
        items
    }

    /// Number of items currently in the buffer.
    pub fn len(&self) -> usize {
        let head = self.head.load(Ordering::Relaxed);
        let tail = self.tail.load(Ordering::Relaxed);
        if head >= tail {
            head - tail
        } else {
            self.capacity - tail + head
        }
    }

    pub fn is_empty(&self) -> bool {
        self.head.load(Ordering::Relaxed) == self.tail.load(Ordering::Relaxed)
    }

    pub fn capacity(&self) -> usize {
        self.capacity - 1 // Usable capacity (one slot reserved for full detection)
    }
}

// SAFETY: RingBuffer is safe to share between threads when used as SPSC.
unsafe impl<T: Copy + Default + Send> Send for RingBuffer<T> {}
unsafe impl<T: Copy + Default + Send> Sync for RingBuffer<T> {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_push_pop() {
        let buf = RingBuffer::<u64>::new(4);
        assert!(buf.push(1));
        assert!(buf.push(2));
        assert!(buf.push(3));
        assert!(!buf.push(4)); // Full (capacity 4, usable 3)

        assert_eq!(buf.pop(), Some(1));
        assert_eq!(buf.pop(), Some(2));
        assert_eq!(buf.pop(), Some(3));
        assert_eq!(buf.pop(), None);
    }

    #[test]
    fn test_push_overwrite() {
        let buf = RingBuffer::<u64>::new(3);
        buf.push_overwrite(1);
        buf.push_overwrite(2);
        buf.push_overwrite(3); // Overwrites oldest

        let items = buf.drain();
        assert_eq!(items.len(), 2);
        assert_eq!(items[0], 2);
        assert_eq!(items[1], 3);
    }

    #[test]
    fn test_drain() {
        let buf = RingBuffer::<f64>::new(8);
        for i in 0..5 {
            buf.push(i as f64);
        }
        let items = buf.drain();
        assert_eq!(items.len(), 5);
        assert!(buf.is_empty());
    }

    #[test]
    fn test_len() {
        let buf = RingBuffer::<i32>::new(8);
        assert_eq!(buf.len(), 0);
        buf.push(1);
        buf.push(2);
        assert_eq!(buf.len(), 2);
        buf.pop();
        assert_eq!(buf.len(), 1);
    }
}
