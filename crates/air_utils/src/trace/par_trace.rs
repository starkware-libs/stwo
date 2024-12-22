use std::marker::PhantomData;

use bytemuck::{cast_slice, Zeroable};
use itertools::Itertools;
use rayon::iter::plumbing::{bridge, Consumer, Producer, ProducerCallback, UnindexedConsumer};
use rayon::iter::{IndexedParallelIterator, ParallelIterator};
use stwo_prover::core::backend::simd::m31::{PackedM31, N_LANES};
use stwo_prover::core::backend::simd::very_packed_m31::VectorizedPackedM31;
use stwo_prover::core::backend::simd::SimdBackend;
use stwo_prover::core::fields::m31::M31;
use stwo_prover::core::poly::circle::{CanonicCoset, CircleEvaluation};
use stwo_prover::core::poly::BitReversedOrder;

/// ```
/// // 2D Matrix of packed [`M31`] values.
/// // Used as the witness for STWO proofs.
/// // Stored in column-major order. Exposes a vectorized iterator over rows.
///
/// // # Example
/// ```
#[derive(Debug)]
pub struct ParallelTrace<const N: usize> {
    data: [Vec<PackedM31>; N],

    /// Number of M31 rows in each column.
    length: usize,
}

impl<const N: usize> ParallelTrace<N> {
    pub fn zeroed(log_size: u32) -> Self {
        let length = 1 << log_size;
        let n_simd_elems = length / N_LANES;
        let data = [(); N].map(|_| vec![PackedM31::zeroed(); n_simd_elems]);
        Self { data, length }
    }

    /// # Safety
    /// The caller must ensure that the column is populated before being used.
    #[allow(clippy::uninit_vec)]
    pub unsafe fn uninitialized(_log_size: u32) -> Self {
        todo!()
    }

    pub fn len(&self) -> usize {
        self.length
    }

    pub fn is_empty(&self) -> bool {
        self.length == 0
    }

    pub fn vectorized_iter_mut<const C: usize>(&mut self) -> VectorizedRowIterMut<C, N> {
        let v = self
            .data
            .iter_mut()
            .map(|column| column.as_mut_slice())
            .collect_vec()
            .try_into()
            .unwrap();
        VectorizedRowIterMut::from_packed(v)
    }

    pub fn par_vectorized_iter_mut<const C: usize>(&mut self) -> ParRowIterMut<C, N> {
        let v = self
            .data
            .iter_mut()
            .map(|column| column.as_mut_slice())
            .collect_vec()
            .try_into()
            .unwrap();
        ParRowIterMut::from_packed(v)
    }

    pub fn to_evals(self) -> [CircleEvaluation<SimdBackend, M31, BitReversedOrder>; N] {
        let length = self.len();
        let _domain = CanonicCoset::new(length.ilog2()).circle_domain();
        self.data.map(|_data| todo!())
    }

    pub fn pretty_print(&self, row_limit: usize) -> String {
        assert!(row_limit <= self.len());
        let cpu_trace: Vec<&[u32]> = self
            .data
            .iter()
            .map(|column| cast_slice(column))
            .collect_vec();
        let mut output = String::new();
        for row in 0..row_limit {
            output.push('|');
            for col in &cpu_trace {
                output.push_str(&format!("{:?}", col[row]));
                output.push('|');
            }
            output.push('\n');
        }
        output
    }
}

pub type VectorizedMutRow<'trace, const C: usize, const N: usize> =
    [&'trace mut VectorizedPackedM31<C>; N];

pub struct VectorizedRowIterMut<'trace, const C: usize, const N: usize> {
    v: [*mut [VectorizedPackedM31<C>]; N],
    phantom: PhantomData<&'trace mut VectorizedPackedM31<C>>,
}
impl<'trace, const C: usize, const N: usize> VectorizedRowIterMut<'trace, C, N> {
    pub(super) fn from_packed(slice: [&'trace mut [PackedM31]; N]) -> Self {
        Self {
            v: slice.map(|s| {
                let ptr = s.as_mut_ptr() as *mut VectorizedPackedM31<C>;
                let length = s.len() / C;
                let slice = unsafe { std::slice::from_raw_parts_mut(ptr, length) };
                slice as *mut [VectorizedPackedM31<C>]
            }),
            phantom: PhantomData,
        }
    }
}
impl<'trace, const C: usize, const N: usize> Iterator for VectorizedRowIterMut<'trace, C, N> {
    type Item = VectorizedMutRow<'trace, C, N>;

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
impl<const C: usize, const N: usize> ExactSizeIterator for VectorizedRowIterMut<'_, C, N> {}
impl<const C: usize, const N: usize> DoubleEndedIterator for VectorizedRowIterMut<'_, C, N> {
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

struct RowProducer<'trace, const C: usize, const N: usize> {
    data: [&'trace mut [VectorizedPackedM31<C>]; N],
}

impl<'trace, const C: usize, const N: usize> Producer for RowProducer<'trace, C, N> {
    type Item = VectorizedMutRow<'trace, C, N>;

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

    type IntoIter = VectorizedRowIterMut<'trace, C, N>;

    fn into_iter(self) -> Self::IntoIter {
        VectorizedRowIterMut {
            v: self.data.map(|s| s as *mut [VectorizedPackedM31<C>]),
            phantom: PhantomData,
        }
    }
}

pub struct ParRowIterMut<'trace, const C: usize, const N: usize> {
    data: [&'trace mut [VectorizedPackedM31<C>]; N],
}
impl<'trace, const C: usize, const N: usize> ParRowIterMut<'trace, C, N> {
    pub(super) fn from_packed(slice: [&'trace mut [PackedM31]; N]) -> Self {
        let v = slice.map(|s| {
            let ptr = s.as_mut_ptr() as *mut VectorizedPackedM31<C>;
            let length = s.len() / C;
            let slice = unsafe { std::slice::from_raw_parts_mut(ptr, length) };
            slice
        });
        Self { data: v }
    }
}
impl<'trace, const C: usize, const N: usize> ParallelIterator for ParRowIterMut<'trace, C, N> {
    type Item = VectorizedMutRow<'trace, C, N>;

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

impl<const C: usize, const N: usize> IndexedParallelIterator for ParRowIterMut<'_, C, N> {
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
    // use stwo_prover::core::backend::simd::m31::PackedM31;
    // use stwo_prover::core::fields::m31::M31;
    // use stwo_prover::core::fields::FieldExpOps;

    use itertools::Itertools;
    use rayon::iter::{IndexedParallelIterator, ParallelIterator};
    use stwo_prover::core::backend::simd::m31::N_LANES;
    use stwo_prover::core::backend::simd::very_packed_m31::VectorizedPackedM31;
    use stwo_prover::core::fields::m31::M31;
    use stwo_prover::core::fields::FieldExpOps;
    // use stwo_prover::core::fields::FieldExpOps;

    #[test]
    fn test_row_iter() {
        const N_COLUMNS: usize = 3;
        const LOG_SIZE: u32 = 8;
        const CHUNK_SIZE: usize = 2;
        let mut trace = super::ParallelTrace::<N_COLUMNS>::zeroed(LOG_SIZE);
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
            .par_vectorized_iter_mut::<CHUNK_SIZE>()
            .enumerate()
            .for_each(|(i, row)| {
                *row[0] = VectorizedPackedM31::from_ref(&arr[i * CHUNK_SIZE * N_LANES..]);
                *row[1] = *row[0] + VectorizedPackedM31::broadcast(M31::from(1));
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
