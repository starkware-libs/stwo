use bytemuck::Zeroable;
use itertools::Itertools;
use stwo_prover::core::backend::simd::column::BaseColumn;
use stwo_prover::core::backend::simd::m31::{PackedM31, LOG_N_LANES, N_LANES};
use stwo_prover::core::backend::simd::SimdBackend;
use stwo_prover::core::fields::m31::M31;
use stwo_prover::core::poly::circle::{CanonicCoset, CircleEvaluation};
use stwo_prover::core::poly::BitReversedOrder;

use super::row_iterator::{ParRowIterMut, RowIterMut};

/// A 2D Matrix of [`PackedM31`] values.
///
/// Used for generating the witness of 'Stwo' proofs.\
/// Stored as an array of `N` columns, each column is a vector of [`PackedM31`] values.
/// All columns are of the same length.\
/// Exposes an iterator over mutable references to the rows of the matrix.
///
/// # Example
///
///  ```text
/// Computation trace of a^2 + (a + 1)^2 for a in 0..256
/// ```
/// ```
/// use stwo_air_utils::trace::component_trace::ComponentTrace;
/// use itertools::Itertools;
/// use stwo_prover::core::backend::simd::m31::{PackedM31, N_LANES};
/// use stwo_prover::core::fields::m31::M31;
/// use stwo_prover::core::fields::FieldExpOps;
///
/// const N_COLUMNS: usize = 3;
/// const LOG_SIZE: u32 = 8;
/// let mut trace = ComponentTrace::<N_COLUMNS>::zeroed(LOG_SIZE);
/// let example_input = (0..1 << LOG_SIZE).map(M31::from).collect_vec(); // 0..256
/// trace
///     .iter_mut()
///     .zip(example_input.chunks(N_LANES))
///     .chunks(4)
///     .into_iter()
///     .for_each(|chunk| {
///         chunk.into_iter().for_each(|(mut row, input)| {
///             *row[0] = PackedM31::from_array(input.try_into().unwrap());
///             *row[1] = *row[0] + PackedM31::broadcast(M31(1));
///             *row[2] = row[0].square() + row[1].square();
///         })
///     });
///
/// let first_3_rows = (0..N_COLUMNS).map(|i| trace.row_at(i)).collect::<Vec<_>>();
/// assert_eq!(first_3_rows, [[0,1,1], [1,2,5], [2,3,13]].map(|row| row.map(M31::from)));
/// ```
#[derive(Debug)]
pub struct ComponentTrace<const N: usize> {
    /// Columns are assumed to be of the same length.
    data: [Vec<PackedM31>; N],

    /// Log number of non-packed rows in each column.
    log_size: u32,
}

impl<const N: usize> ComponentTrace<N> {
    /// Creates a new `ComponentTrace` with all values initialized to zero.
    /// The number of rows in each column is `2^log_size`.
    ///
    /// # Panics
    ///
    /// if log_size < 4.
    pub fn zeroed(log_size: u32) -> Self {
        assert!(
            log_size >= LOG_N_LANES,
            "log_size < LOG_N_LANES not supported!"
        );
        let n_simd_elems = 1 << (log_size - LOG_N_LANES);
        let data = [(); N].map(|_| vec![PackedM31::zeroed(); n_simd_elems]);
        Self { data, log_size }
    }

    /// Creates a new `ComponentTrace` with all values uninitialized.
    ///
    /// # Safety
    ///
    /// The caller must ensure that the column is populated before being used.
    /// The number of rows in each column is `2^log_size`.
    ///
    /// # Panics
    ///
    /// if `log_size` < 4.
    #[allow(clippy::uninit_vec)]
    pub unsafe fn uninitialized(log_size: u32) -> Self {
        assert!(
            log_size >= LOG_N_LANES,
            "log_size < LOG_N_LANES not supported!"
        );
        let n_simd_elems = 1 << (log_size - LOG_N_LANES);
        let data = [(); N].map(|_| {
            let mut vec = Vec::with_capacity(n_simd_elems);
            vec.set_len(n_simd_elems);
            vec
        });
        Self { data, log_size }
    }

    pub fn log_size(&self) -> u32 {
        self.log_size
    }

    pub fn iter_mut(&mut self) -> RowIterMut<'_> {
        RowIterMut::new(
            self.data
                .iter_mut()
                .map(|col| col.as_mut_slice())
                .collect_vec(),
        )
    }

    pub fn par_iter_mut(&mut self) -> ParRowIterMut<'_> {
        ParRowIterMut::new(
            self.data
                .iter_mut()
                .map(|col| col.as_mut_slice())
                .collect_vec(),
        )
    }

    pub fn to_evals(self) -> [CircleEvaluation<SimdBackend, M31, BitReversedOrder>; N] {
        let domain = CanonicCoset::new(self.log_size).circle_domain();
        self.data.map(|column| {
            CircleEvaluation::<SimdBackend, M31, BitReversedOrder>::new(
                domain,
                BaseColumn::from_simd(column),
            )
        })
    }

    pub fn row_at(&self, row: usize) -> [M31; N] {
        assert!(row < 1 << self.log_size);
        let packed_row = row / N_LANES;
        let idx_in_simd_vector = row % N_LANES;
        self.data
            .each_ref()
            .map(|column| column[packed_row].to_array()[idx_in_simd_vector])
    }
}

#[cfg(test)]
mod tests {
    use itertools::Itertools;
    use stwo_prover::core::backend::simd::m31::{PackedM31, N_LANES};
    use stwo_prover::core::fields::m31::M31;
    use stwo_prover::core::fields::FieldExpOps;

    #[test]
    fn test_parallel_trace() {
        use rayon::iter::{IndexedParallelIterator, ParallelIterator};
        use rayon::slice::ParallelSlice;

        const N_COLUMNS: usize = 3;
        const LOG_SIZE: u32 = 8;
        const CHUNK_SIZE: usize = 4;
        let mut trace = super::ComponentTrace::<N_COLUMNS>::zeroed(LOG_SIZE);
        let arr = (0..1 << LOG_SIZE).map(M31::from).collect_vec();
        let expected = arr
            .iter()
            .map(|&a| {
                let b = a + M31::from(1);
                let c = a.square() + b.square();
                (a, b, c)
            })
            .multiunzip();

        trace
            .par_iter_mut()
            .zip(arr.par_chunks(N_LANES))
            .chunks(CHUNK_SIZE)
            .for_each(|chunk| {
                chunk.into_iter().for_each(|(mut row, input)| {
                    *row[0] = PackedM31::from_array(input.try_into().unwrap());
                    *row[1] = *row[0] + PackedM31::broadcast(M31(1));
                    *row[2] = row[0].square() + row[1].square();
                });
            });
        let actual = trace
            .data
            .map(|c| {
                c.into_iter()
                    .flat_map(|packed| packed.to_array())
                    .collect_vec()
            })
            .into_iter()
            .next_tuple()
            .unwrap();

        assert_eq!(expected, actual);
    }

    #[test]
    fn test_component_trace_uninitialized_success() {
        const N_COLUMNS: usize = 3;
        const LOG_SIZE: u32 = 4;
        unsafe { super::ComponentTrace::<N_COLUMNS>::uninitialized(LOG_SIZE) };
    }

    #[should_panic = "log_size < LOG_N_LANES not supported!"]
    #[test]
    fn test_component_trace_uninitialized_fails() {
        const N_COLUMNS: usize = 3;
        const LOG_SIZE: u32 = 3;
        unsafe { super::ComponentTrace::<N_COLUMNS>::uninitialized(LOG_SIZE) };
    }
}
