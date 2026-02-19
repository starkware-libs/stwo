//! Pre-allocated memory pools for the proving pipeline.
//!
//! The [`ColumnPool`] manages reusable [`SecureColumnByCoords`] buffers organized by log_size,
//! avoiding repeated allocation/deallocation of large column buffers during proving.
//! The [`ProverMemPool`] pre-allocates all needed buffers upfront based on the proving workload.

use std::collections::HashMap;

use crate::core::fields::m31::BaseField;
use crate::prover::backend::ColumnOps;
use crate::prover::secure_column::SecureColumnByCoords;

/// A pool of pre-allocated [`SecureColumnByCoords`] buffers, organized by log_size.
pub struct ColumnPool<B: ColumnOps<BaseField>> {
    /// Map from log_size -> stack of available buffers.
    pools: HashMap<u32, Vec<SecureColumnByCoords<B>>>,
}

impl<B: ColumnOps<BaseField>> ColumnPool<B> {
    /// Creates a new empty column pool.
    pub fn new() -> Self {
        Self {
            pools: HashMap::new(),
        }
    }

    /// Pre-allocates `count` zero-initialized buffers of size `1 << log_size`.
    pub fn reserve(&mut self, log_size: u32, count: usize) {
        let pool = self.pools.entry(log_size).or_default();
        for _ in 0..count {
            pool.push(SecureColumnByCoords::zeros(1 << log_size));
        }
    }

    /// Takes a buffer from the pool for the given `log_size`.
    ///
    /// # Panics
    ///
    /// Panics if no buffer of the requested size is available.
    pub fn take(&mut self, log_size: u32) -> SecureColumnByCoords<B> {
        self.pools
            .get_mut(&log_size)
            .and_then(|pool| pool.pop())
            .unwrap_or_else(|| panic!("ColumnPool: no buffer available for log_size={log_size}"))
    }

    /// Takes a buffer from the pool, or allocates a new zero-initialized one if none is available.
    pub fn take_or_alloc(&mut self, log_size: u32) -> SecureColumnByCoords<B> {
        self.pools
            .get_mut(&log_size)
            .and_then(|pool| pool.pop())
            .unwrap_or_else(|| SecureColumnByCoords::zeros(1 << log_size))
    }

    /// Returns a buffer to the pool. The caller is responsible for ensuring the buffer's log_size
    /// matches.
    pub fn give_back(&mut self, log_size: u32, buf: SecureColumnByCoords<B>) {
        debug_assert_eq!(buf.len(), 1 << log_size);
        self.pools.entry(log_size).or_default().push(buf);
    }

    /// Takes a buffer from the pool, zeroing it before returning. Falls back to allocating a new
    /// zero-initialized buffer if none is available.
    pub fn take_zeroed(&mut self, log_size: u32) -> SecureColumnByCoords<B> {
        if let Some(mut buf) = self.pools.get_mut(&log_size).and_then(|pool| pool.pop()) {
            zero_secure_column(&mut buf);
            buf
        } else {
            SecureColumnByCoords::zeros(1 << log_size)
        }
    }

    /// Returns the number of available buffers for a given log_size.
    pub fn available(&self, log_size: u32) -> usize {
        self.pools.get(&log_size).map_or(0, |pool| pool.len())
    }

    /// Returns the total number of buffers across all sizes.
    pub fn total_available(&self) -> usize {
        self.pools.values().map(|pool| pool.len()).sum()
    }
}

impl<B: ColumnOps<BaseField>> Default for ColumnPool<B> {
    fn default() -> Self {
        Self::new()
    }
}

/// Zeroes out all columns in a [`SecureColumnByCoords`].
fn zero_secure_column<B: ColumnOps<BaseField>>(col: &mut SecureColumnByCoords<B>) {
    let len = col.len();
    *col = SecureColumnByCoords::zeros(len);
}

/// Pre-allocated memory for the entire proving pipeline.
///
/// Created by analyzing the component structure and PCS configuration before proving begins.
/// Contains pools of reusable column buffers that various parts of the prover can draw from
/// instead of allocating on-demand.
pub struct ProverMemPool<B: ColumnOps<BaseField>> {
    /// Pool of reusable [`SecureColumnByCoords`] buffers.
    pub column_pool: ColumnPool<B>,
}

impl<B: ColumnOps<BaseField>> ProverMemPool<B> {
    /// Creates a new workspace with an empty column pool.
    pub fn new() -> Self {
        Self {
            column_pool: ColumnPool::new(),
        }
    }

    /// Creates a workspace with pre-allocated buffers based on the specified requirements.
    ///
    /// `requirements` is a list of `(log_size, count)` pairs indicating how many buffers of each
    /// size to pre-allocate.
    pub fn with_requirements(requirements: &[(u32, usize)]) -> Self {
        let mut workspace = Self::new();
        for &(log_size, count) in requirements {
            workspace.column_pool.reserve(log_size, count);
        }
        workspace
    }
}

impl<B: ColumnOps<BaseField>> Default for ProverMemPool<B> {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::prover::backend::CpuBackend;

    #[test]
    fn test_column_pool_reserve_and_take() {
        let mut pool = ColumnPool::<CpuBackend>::new();
        pool.reserve(4, 3);
        assert_eq!(pool.available(4), 3);

        let buf = pool.take(4);
        assert_eq!(buf.len(), 1 << 4);
        assert_eq!(pool.available(4), 2);
    }

    #[test]
    fn test_column_pool_give_back() {
        let mut pool = ColumnPool::<CpuBackend>::new();
        pool.reserve(5, 1);
        let buf = pool.take(5);
        assert_eq!(pool.available(5), 0);

        pool.give_back(5, buf);
        assert_eq!(pool.available(5), 1);
    }

    #[test]
    fn test_column_pool_take_or_alloc() {
        let mut pool = ColumnPool::<CpuBackend>::new();

        // No pre-allocated buffer, should allocate.
        let buf = pool.take_or_alloc(3);
        assert_eq!(buf.len(), 1 << 3);
        assert_eq!(pool.available(3), 0);

        // Return and take again.
        pool.give_back(3, buf);
        assert_eq!(pool.available(3), 1);
        let _buf = pool.take_or_alloc(3);
        assert_eq!(pool.available(3), 0);
    }

    #[test]
    fn test_column_pool_take_zeroed() {
        let mut pool = ColumnPool::<CpuBackend>::new();
        pool.reserve(4, 1);

        let buf = pool.take_zeroed(4);
        assert_eq!(buf.len(), 1 << 4);
        // Verify all values are zero.
        for i in 0..buf.len() {
            assert!(buf.at(i).is_zero(), "non-zero at index {i}");
        }
    }

    #[test]
    #[should_panic(expected = "no buffer available")]
    fn test_column_pool_take_panics_when_empty() {
        let mut pool = ColumnPool::<CpuBackend>::new();
        pool.take(4);
    }

    #[test]
    fn test_prover_mempool_with_requirements() {
        let workspace = ProverMemPool::<CpuBackend>::with_requirements(&[(4, 2), (5, 3), (6, 1)]);
        assert_eq!(workspace.column_pool.available(4), 2);
        assert_eq!(workspace.column_pool.available(5), 3);
        assert_eq!(workspace.column_pool.available(6), 1);
        assert_eq!(workspace.column_pool.total_available(), 6);
    }

    use num_traits::Zero;
}
