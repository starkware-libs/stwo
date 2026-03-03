//! Pre-allocated memory pools for the proving pipeline.
//!
//! The [`BaseColumnPool`] manages reusable [`Col<B, BaseField>`] buffers for polynomial evaluation,
//! avoiding repeated allocation/deallocation of large column buffers during proving.

use dashmap::DashMap;

use crate::core::fields::m31::BaseField;
use crate::prover::backend::{Col, Column, ColumnOps};

/// A pool of pre-allocated [`Col<B, BaseField>`] buffers, organized by log_size.
///
/// Used to avoid repeated allocation of evaluation buffers during polynomial commitment.
pub struct BaseColumnPool<B: ColumnOps<BaseField>> {
    /// Map from log_size -> stack of available buffers.
    pools: DashMap<u32, Vec<Col<B, BaseField>>>,
}

impl<B: ColumnOps<BaseField>> BaseColumnPool<B> {
    /// Creates a new empty base column pool.
    pub fn new() -> Self {
        Self {
            pools: DashMap::new(),
        }
    }

    /// Pre-allocates `count` zero-initialized buffers of size `1 << log_size`.
    pub fn reserve(&self, log_size: u32, count: usize) {
        let mut pool = self.pools.entry(log_size).or_default();
        for _ in 0..count {
            pool.push(Col::<B, BaseField>::zeros(1 << log_size));
        }
    }

    /// Takes a buffer from the pool for the given `log_size`.
    ///
    /// # Panics
    ///
    /// Panics if no buffer of the requested size is available.
    pub fn take(&self, log_size: u32) -> Col<B, BaseField> {
        self.pools
            .get_mut(&log_size)
            .and_then(|mut pool| pool.pop())
            .unwrap_or_else(|| {
                panic!("BaseColumnPool: no buffer available for log_size={log_size}")
            })
    }

    /// Takes a buffer from the pool, or allocates a new zero-initialized one if none is available.
    pub fn take_or_alloc(&self, log_size: u32) -> Col<B, BaseField> {
        self.pools
            .get_mut(&log_size)
            .and_then(|mut pool| pool.pop())
            .unwrap_or_else(|| unsafe { Col::<B, BaseField>::uninitialized(1 << log_size) })
    }

    /// Returns a buffer to the pool. The caller is responsible for ensuring the buffer's log_size
    /// matches.
    pub fn give_back(&self, log_size: u32, buf: Col<B, BaseField>) {
        debug_assert_eq!(buf.len(), 1 << log_size);
        self.pools.entry(log_size).or_default().push(buf);
    }
}

impl<B: ColumnOps<BaseField>> Default for BaseColumnPool<B> {
    fn default() -> Self {
        Self::new()
    }
}
