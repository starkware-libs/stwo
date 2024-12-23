use std::marker::PhantomData;

use bytemuck::Zeroable;
use rayon::iter::plumbing::{bridge, Consumer, Producer, ProducerCallback, UnindexedConsumer};
use rayon::prelude::*;
use stwo_prover::core::backend::simd::m31::{PackedM31, LOG_N_LANES, N_LANES};
use stwo_prover::core::backend::simd::SimdBackend;
use stwo_prover::core::fields::m31::M31;
use stwo_prover::core::poly::circle::CircleEvaluation;
use stwo_prover::core::poly::BitReversedOrder;

/// A 2D Matrix of [`PackedM31`] values.
/// Used for generating the witness of 'Stwo' proofs.
/// Stored as an array of `N` columns, each column is a vector of [`PackedM31`] values.
/// Exposes an iterator over mutable references to the rows of the matrix.
///
/// # Example:
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
///         chunk.into_iter().for_each(|(row, input)| {
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
    data: [Vec<PackedM31>; N],

    /// Log number of non-packed rows in each column.
    log_size: u32,
}

impl<const N: usize> ComponentTrace<N> {
    pub fn zeroed(log_size: u32) -> Self {
        let n_simd_elems = 1 << (log_size - LOG_N_LANES);
        let data = [(); N].map(|_| vec![PackedM31::zeroed(); n_simd_elems]);
        Self { data, log_size }
    }

    /// # Safety
    /// The caller must ensure that the column is populated before being used.
    #[allow(clippy::uninit_vec)]
    pub unsafe fn uninitialized(_log_size: u32) -> Self {
        todo!()
    }

    pub fn log_size(&self) -> u32 {
        self.log_size
    }

    pub fn iter_mut(&mut self) -> RowIterMut<'_, N> {
        RowIterMut::new(self.data.each_mut().map(|column| column.as_mut_slice()))
    }

    pub fn par_iter_mut(&mut self) -> ParRowIterMut<'_, N> {
        ParRowIterMut::new(self.data.each_mut().map(|column| column.as_mut_slice()))
    }

    pub fn to_evals(self) -> [CircleEvaluation<SimdBackend, M31, BitReversedOrder>; N] {
        todo!()
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

pub type MutRow<'trace, const N: usize> = [&'trace mut PackedM31; N];

/// An iterator over mutable references to the rows of a [`ComponentTrace`].
pub struct RowIterMut<'trace, const N: usize> {
    v: [*mut [PackedM31]; N],
    phantom: PhantomData<&'trace ()>,
}
impl<'trace, const N: usize> RowIterMut<'trace, N> {
    pub fn new(slice: [&'trace mut [PackedM31]; N]) -> Self {
        Self {
            v: slice.map(|s| s as *mut _),
            phantom: PhantomData,
        }
    }
}
impl<'trace, const N: usize> Iterator for RowIterMut<'trace, N> {
    type Item = MutRow<'trace, N>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.v[0].is_empty() {
            return None;
        }
        let item = std::array::from_fn(|i| unsafe {
            // SAFETY: The self.v contract ensures that any split_at_mut is valid.
            let (head, tail) = self.v[i].split_at_mut(1);
            self.v[i] = tail;
            &mut (*head)[0]
        });
        Some(item)
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let len = self.v[0].len();
        (len, Some(len))
    }
}
impl<const N: usize> ExactSizeIterator for RowIterMut<'_, N> {}
impl<const N: usize> DoubleEndedIterator for RowIterMut<'_, N> {
    fn next_back(&mut self) -> Option<Self::Item> {
        if self.v[0].is_empty() {
            return None;
        }
        let item = std::array::from_fn(|i| unsafe {
            // SAFETY: The self.v contract ensures that any split_at_mut is valid.
            let (head, tail) = self.v[i].split_at_mut(self.v[i].len() - 1);
            self.v[i] = head;
            &mut (*tail)[0]
        });
        Some(item)
    }
}

struct RowProducer<'trace, const N: usize> {
    data: [&'trace mut [PackedM31]; N],
}
impl<'trace, const N: usize> Producer for RowProducer<'trace, N> {
    type Item = MutRow<'trace, N>;

    fn split_at(self, index: usize) -> (Self, Self) {
        let mut left: [_; N] = unsafe { std::mem::zeroed() };
        let mut right: [_; N] = unsafe { std::mem::zeroed() };
        for (i, slice) in self.data.into_iter().enumerate() {
            let (lhs, rhs) = slice.split_at_mut(index);
            left[i] = lhs;
            right[i] = rhs;
        }
        (RowProducer { data: left }, RowProducer { data: right })
    }

    type IntoIter = RowIterMut<'trace, N>;

    fn into_iter(self) -> Self::IntoIter {
        RowIterMut {
            v: self.data.map(|s| s as *mut _),
            phantom: PhantomData,
        }
    }
}

/// A parallel iterator over mutable references to the rows of a [`ComponentTrace`].
/// [`ComponentTrace`] is an array of columns, hence iterating over rows is not trivial.
pub struct ParRowIterMut<'trace, const N: usize> {
    data: [&'trace mut [PackedM31]; N],
}
impl<'trace, const N: usize> ParRowIterMut<'trace, N> {
    pub(super) fn new(data: [&'trace mut [PackedM31]; N]) -> Self {
        Self { data }
    }
}
impl<'trace, const N: usize> ParallelIterator for ParRowIterMut<'trace, N> {
    type Item = MutRow<'trace, N>;

    fn drive_unindexed<D>(self, consumer: D) -> D::Result
    where
        D: UnindexedConsumer<Self::Item>,
    {
        bridge(self, consumer)
    }

    fn opt_len(&self) -> Option<usize> {
        Some(self.len())
    }
}
impl<const N: usize> IndexedParallelIterator for ParRowIterMut<'_, N> {
    fn len(&self) -> usize {
        self.data[0].len()
    }

    fn drive<D: Consumer<Self::Item>>(self, consumer: D) -> D::Result {
        bridge(self, consumer)
    }

    fn with_producer<CB: ProducerCallback<Self::Item>>(self, callback: CB) -> CB::Output {
        callback.callback(RowProducer { data: self.data })
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
            .for_each(|(row, input)| {
                *row[0] = PackedM31::from_array(input.try_into().unwrap());
                *row[1] = *row[0] + PackedM31::broadcast(M31(1));
                *row[2] = row[0].square() + row[1].square();
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
}
