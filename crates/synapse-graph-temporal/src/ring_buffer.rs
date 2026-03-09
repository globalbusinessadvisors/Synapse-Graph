//! Fixed-capacity circular buffer with FIFO eviction (DDD-003).

/// A fixed-capacity ring buffer that silently overwrites the oldest entry on
/// overflow.
#[derive(Clone, Debug)]
pub struct RingBuffer<T> {
    buf: Vec<Option<T>>,
    capacity: usize,
    /// Write cursor — next slot to write to.
    head: usize,
    /// Number of items currently stored.
    len: usize,
    /// Cumulative count of evicted items.
    evicted: usize,
}

impl<T> RingBuffer<T> {
    pub fn new(capacity: usize) -> Self {
        assert!(capacity > 0, "ring buffer capacity must be > 0");
        let mut buf = Vec::with_capacity(capacity);
        for _ in 0..capacity {
            buf.push(None);
        }
        Self {
            buf,
            capacity,
            head: 0,
            len: 0,
            evicted: 0,
        }
    }

    /// Push an item. Returns `true` if an older item was evicted.
    pub fn push(&mut self, item: T) -> bool {
        let was_full = self.buf[self.head].is_some();
        self.buf[self.head] = Some(item);
        self.head = (self.head + 1) % self.capacity;
        if was_full {
            self.evicted += 1;
            true
        } else {
            self.len += 1;
            false
        }
    }

    pub fn len(&self) -> usize {
        self.len
    }

    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    pub fn capacity(&self) -> usize {
        self.capacity
    }

    pub fn evicted_count(&self) -> usize {
        self.evicted
    }

    /// Iterate over all stored items in insertion order (oldest first).
    pub fn iter(&self) -> impl Iterator<Item = &T> {
        let start = if self.len < self.capacity {
            0
        } else {
            self.head // head points to the oldest slot when full
        };
        let cap = self.capacity;
        let len = self.len;
        (0..len).filter_map(move |i| {
            let idx = (start + i) % cap;
            self.buf[idx].as_ref()
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn push_and_iterate() {
        let mut rb = RingBuffer::new(3);
        rb.push(1);
        rb.push(2);
        rb.push(3);
        let items: Vec<&i32> = rb.iter().collect();
        assert_eq!(items, vec![&1, &2, &3]);
        assert_eq!(rb.len(), 3);
        assert_eq!(rb.evicted_count(), 0);
    }

    #[test]
    fn eviction_on_overflow() {
        let mut rb = RingBuffer::new(3);
        rb.push(1);
        rb.push(2);
        rb.push(3);
        let evicted = rb.push(4); // evicts 1
        assert!(evicted);
        assert_eq!(rb.len(), 3);
        assert_eq!(rb.evicted_count(), 1);
        let items: Vec<&i32> = rb.iter().collect();
        assert_eq!(items, vec![&2, &3, &4]);
    }

    #[test]
    fn multiple_evictions() {
        let mut rb = RingBuffer::new(2);
        rb.push(10);
        rb.push(20);
        rb.push(30); // evicts 10
        rb.push(40); // evicts 20
        assert_eq!(rb.evicted_count(), 2);
        let items: Vec<&i32> = rb.iter().collect();
        assert_eq!(items, vec![&30, &40]);
    }
}
