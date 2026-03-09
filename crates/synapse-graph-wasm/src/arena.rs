//! Arena allocator for WASM linear memory (ADR-006 risk mitigation).
//!
//! Provides a bump-style arena that reduces WASM linear memory fragmentation
//! by pre-allocating a contiguous region for the fast-tier ring buffer and
//! temporary spike processing buffers.

use std::cell::RefCell;

/// A bump-style arena allocator for WASM linear memory.
///
/// Pre-allocates a contiguous region and hands out slices from it,
/// avoiding the fragmentation that comes from many small allocations
/// in WASM's linear memory model.
pub struct Arena {
    buffer: Vec<u8>,
    offset: usize,
}

impl Arena {
    /// Create a new arena with the given capacity in bytes.
    pub fn new(capacity: usize) -> Self {
        Self {
            buffer: vec![0u8; capacity],
            offset: 0,
        }
    }

    /// Allocate `size` bytes from the arena, returning a mutable slice.
    ///
    /// Returns `None` if the arena doesn't have enough remaining capacity.
    pub fn alloc(&mut self, size: usize) -> Option<&mut [u8]> {
        let aligned_size = (size + 7) & !7; // 8-byte alignment
        if self.offset + aligned_size > self.buffer.len() {
            return None;
        }
        let start = self.offset;
        self.offset += aligned_size;
        Some(&mut self.buffer[start..start + size])
    }

    /// Allocate space for `count` f32 values, returning a mutable slice.
    pub fn alloc_f32(&mut self, count: usize) -> Option<&mut [f32]> {
        let byte_size = count * 4;
        let aligned_offset = (self.offset + 3) & !3; // 4-byte align for f32
        if aligned_offset + byte_size > self.buffer.len() {
            return None;
        }
        self.offset = aligned_offset + byte_size;
        let slice = &mut self.buffer[aligned_offset..aligned_offset + byte_size];
        // Safety: aligned to 4 bytes, correct length
        let (prefix, floats, suffix) = unsafe { slice.align_to_mut::<f32>() };
        debug_assert!(prefix.is_empty() && suffix.is_empty());
        Some(floats)
    }

    /// Reset the arena, making all previously allocated memory available again.
    ///
    /// This is the key operation for WASM memory efficiency: instead of
    /// freeing individual allocations (which fragments linear memory),
    /// we reset the entire arena after each processing cycle.
    pub fn reset(&mut self) {
        self.offset = 0;
    }

    /// Returns the number of bytes currently allocated.
    pub fn used(&self) -> usize {
        self.offset
    }

    /// Returns the total capacity of the arena in bytes.
    pub fn capacity(&self) -> usize {
        self.buffer.len()
    }

    /// Returns the number of bytes remaining.
    pub fn remaining(&self) -> usize {
        self.buffer.len() - self.offset
    }
}

// Thread-local arena for per-cycle temporary allocations in WASM.
// In WASM (single-threaded), this provides a global scratch space
// that is reset after each ingest/predict cycle.
thread_local! {
    static SCRATCH_ARENA: RefCell<Arena> = RefCell::new(Arena::new(256 * 1024)); // 256 KB
}

/// Execute a closure with access to the thread-local scratch arena.
/// The arena is automatically reset after the closure returns.
pub fn with_scratch_arena<F, R>(f: F) -> R
where
    F: FnOnce(&mut Arena) -> R,
{
    SCRATCH_ARENA.with(|cell| {
        let mut arena = cell.borrow_mut();
        let result = f(&mut arena);
        arena.reset();
        result
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn arena_basic_allocation() {
        let mut arena = Arena::new(1024);
        assert_eq!(arena.capacity(), 1024);
        assert_eq!(arena.used(), 0);

        let slice = arena.alloc(100).unwrap();
        assert_eq!(slice.len(), 100);
        assert!(arena.used() >= 100);

        let slice2 = arena.alloc(200).unwrap();
        assert_eq!(slice2.len(), 200);
    }

    #[test]
    fn arena_f32_allocation() {
        let mut arena = Arena::new(4096);
        let floats = arena.alloc_f32(16).unwrap();
        assert_eq!(floats.len(), 16);

        // Write and verify
        for (i, f) in floats.iter_mut().enumerate() {
            *f = i as f32;
        }
        let floats2 = arena.alloc_f32(8).unwrap();
        assert_eq!(floats2.len(), 8);
    }

    #[test]
    fn arena_reset_reclaims_memory() {
        let mut arena = Arena::new(512);
        arena.alloc(400).unwrap();
        assert!(arena.used() >= 400);
        assert!(arena.remaining() < 200);

        arena.reset();
        assert_eq!(arena.used(), 0);
        assert_eq!(arena.remaining(), 512);

        // Can allocate again after reset.
        let slice = arena.alloc(500).unwrap();
        assert_eq!(slice.len(), 500);
    }

    #[test]
    fn arena_exhaustion_returns_none() {
        let mut arena = Arena::new(64);
        assert!(arena.alloc(60).is_some());
        assert!(arena.alloc(60).is_none()); // not enough space
    }

    #[test]
    fn scratch_arena_resets_after_use() {
        let used_inside = with_scratch_arena(|arena| {
            arena.alloc(1000).unwrap();
            arena.used()
        });
        assert!(used_inside >= 1000);

        // After with_scratch_arena, the arena should be reset.
        let used_after = with_scratch_arena(|arena| arena.used());
        assert_eq!(used_after, 0);
    }
}
