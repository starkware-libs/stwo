use std::fmt::Debug;

pub use cpu::CpuBackend;

use crate::core::channel::MerkleChannel;
use crate::core::fields::m31::BaseField;
use crate::core::fields::qm31::SecureField;
use crate::core::proof_of_work::GrindOps;
use crate::prover::fri::FriOps;
use crate::prover::lookups::gkr_prover::GkrOps;
use crate::prover::poly::circle::PolyOps;
use crate::prover::vcs_lifted::ops::MerkleOpsLifted;
use crate::prover::{AccumulationOps, QuotientOps};

pub mod cpu;
#[cfg(feature = "cuda")]
pub mod cuda;
#[cfg(feature = "cuda")]
pub use cuda::CudaBackend;
pub mod simd;

pub trait Backend:
    Copy
    + Clone
    + Debug
    + ColumnOps<BaseField>
    + ColumnOps<SecureField>
    + PolyOps
    + QuotientOps
    + FriOps
    + AccumulationOps
    + GkrOps
{
}

pub trait BackendForChannel<MC: MerkleChannel>:
    Backend + MerkleOpsLifted<MC::H> + GrindOps<MC::C>
{
}

pub trait ColumnOps<T> {
    type Column: Column<T>;
    /// Whether [`BaseColumnPool`](crate::prover::mempool::BaseColumnPool) recycling works on this
    /// backend: the pool only recycles if the backend's allocation paths draw from it. A backend
    /// whose allocations bypass the pool must opt out, or `give_back` deposits accumulate
    /// unboundedly in a long-lived pool (deposits with no withdrawals).
    const RECYCLES_COLUMNS: bool = true;
    fn bit_reverse_column(column: &mut Self::Column);
}

pub type Col<B, T> = <B as ColumnOps<T>>::Column;

// TODO(alont): Consider removing the generic parameter and only support BaseField.
pub trait Column<T>: Clone + Debug + FromIterator<T> + Send + Sync {
    /// Creates a new column of zeros with the given length.
    fn zeros(len: usize) -> Self;
    /// Creates a new column of uninitialized values with the given length.
    /// # Safety
    /// The caller must ensure that the column is populated before being used.
    unsafe fn uninitialized(len: usize) -> Self;
    /// Returns a cpu vector of the column.
    fn to_cpu(&self) -> Vec<T>;
    /// Returns the length of the column.
    fn len(&self) -> usize;
    /// Returns true if the column is empty.
    fn is_empty(&self) -> bool {
        self.len() == 0
    }
    /// Retrieves the element at the given index.
    fn at(&self, index: usize) -> T;
    /// Retrieves the elements at the given indices, in the same order as `indices`.
    ///
    /// Semantically identical to `indices.iter().map(|&i| self.at(i)).collect()` — the default impl
    /// is exactly that. Backends that can batch the read (e.g. CUDA, where each `at` is a separate
    /// device→host copy) override this with a bulk gather that returns the SAME values in the SAME
    /// order, so the only observable change is transfer batching, never the produced bytes.
    fn batch_at(&self, indices: &[usize]) -> Vec<T> {
        indices.iter().map(|&i| self.at(i)).collect()
    }
    /// Retrieves a single element on the LATENCY-CRITICAL serial root-read path (a Merkle/FRI layer
    /// root that is immediately mixed into the channel to draw the next challenge).
    ///
    /// Semantically identical to [`Column::at`] — same value, same index — the default impl is
    /// exactly `self.at(index)`. Backends where a per-element read is a device→host copy (CUDA)
    /// override this to use a pinned staging buffer + a copy-stream sync instead of the pageable
    /// blocking default-stream read, cutting the per-hop latency. CPU/SIMD keep the default, so
    /// their behavior is unchanged. The only observable change is the transfer path, never the
    /// bytes.
    fn at_root_pinned(&self, index: usize) -> T {
        self.at(index)
    }
    /// Sets the element at the given index.
    fn set(&mut self, index: usize, value: T);
    /// Splits the column into two halves.
    fn split_at_mid(self) -> (Self, Self);
}
