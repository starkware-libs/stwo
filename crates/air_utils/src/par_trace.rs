use std::marker::PhantomData;

use bytemuck::{cast_slice, Zeroable};
use itertools::Itertools;
use stwo_prover::core::backend::simd::column::BaseColumn;
use stwo_prover::core::backend::simd::m31::{PackedM31, N_LANES};
use stwo_prover::core::backend::simd::SimdBackend;
use stwo_prover::core::fields::m31::M31;
use stwo_prover::core::poly::circle::{CanonicCoset, CircleEvaluation};
use stwo_prover::core::poly::BitReversedOrder;

/// ```
/// // 2D Matrix of packed [`M31`] values.
/// // Used as the witness for STWO proofs.
/// // Stored in column-major order, and exposes a row-major view and a parallel iterator it.
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
        let data = [(); N].map(|_| vec![PackedM31::zeroed(); length / N_LANES]);
        Self { data, length }
    }

    /// # Safety
    /// The caller must ensure that the column is populated before being used.
    #[allow(clippy::uninit_vec)]
    pub unsafe fn uninitialized(log_size: u32) -> Self {
        let length = 1 << log_size;
        let data = [(); N].map(|_| {
            let mut v = Vec::with_capacity(length / N_LANES);
            v.set_len(length / N_LANES);
            v
        });
        Self { data, length }
    }

    pub fn len(&self) -> usize {
        self.length
    }

    pub fn is_empty(&self) -> bool {
        self.length == 0
    }

    pub fn simd_row_chunks_mut(&mut self, chunk_size: usize) -> RowChunksMut<N> {
        let v: [_; N] = self
            .data
            .iter_mut()
            .map(|v| v.as_mut_slice())
            .collect_vec()
            .try_into()
            .unwrap();
        RowChunksMut::new(v, chunk_size)
    }

    // pub fn mut_row_par_iter(&mut self) -> ParRowViewIterator<N> {
    //     let data: [_; N] = self
    //         .data
    //         .iter_mut()
    //         .map(|c| c.as_mut_slice())
    //         .collect_vec()
    //         .try_into()
    //         .unwrap();
    //     ParRowViewIterator { data }
    // }

    pub fn to_evals(self) -> [CircleEvaluation<SimdBackend, M31, BitReversedOrder>; N] {
        let length = self.len();
        let domain = CanonicCoset::new(length.ilog2()).circle_domain();
        self.data
            .map(|data| CircleEvaluation::new(domain, BaseColumn { data, length }))
    }

    pub fn pretty_print(&self, row_limit: usize) -> String {
        assert!(row_limit <= self.len());
        let cpu_trace: Vec<&[u32]> = self
            .data
            .iter()
            .map(|column| cast_slice(&column))
            .collect_vec();
        let mut output = String::new();
        for row in 0..row_limit {
            output.push_str("|");
            for col in 0..N {
                output.push_str(&format!("{:?}", cpu_trace[col][row]));
                output.push_str("|");
            }
            output.push_str("\n");
        }
        output
    }
}

pub type MutRow<'trace, const N: usize> = [&'trace mut PackedM31; N];
pub struct RowIterMut<'trace, const N: usize> {
    v: [*mut [PackedM31]; N],
    phantom: PhantomData<&'trace mut PackedM31>,
}
impl<'trace, const N: usize> Iterator for RowIterMut<'trace, N> {
    type Item = MutRow<'trace, N>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.v[0].is_empty() {
            return None;
        }
        let item = std::array::from_fn(|i| unsafe {
            // SAFETY: The self.v contract ensures that any split_at_mut is valid.
            let (head, tail) = self.v[i].split_at_mut(0);
            self.v[i] = tail;
            &mut (*head)[0]
        });
        Some(item)
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
            let (head, tail) = self.v[i].split_at_mut(0);
            self.v[i] = tail;
            &mut (*head)[0]
        });
        Some(item)
    }
}

/// Iterator over chunks of rows of a [`ParallelTrace`].
pub struct RowChunksMut<'trace, const N: usize> {
    /// # Safety
    /// This slice pointer must point at a valid region of `T` with at least length `v.len()`.
    /// Normally, those requirements would mean that we could instead use a `&mut [T]` here,
    /// but we cannot because `__iterator_get_unchecked` needs to return `&mut [T]`, which
    /// guarantees certain aliasing properties that we cannot uphold if we hold on to the full
    /// original `&mut [T]`. Wrapping a raw slice instead lets us hand out non-overlapping
    /// `&mut [T]` subslices of the slice we wrap.
    v: [*mut [PackedM31]; N],
    chunk_size: usize,
    _marker: PhantomData<&'trace mut PackedM31>,
}
impl<'a, const N: usize> RowChunksMut<'a, N> {
    #[allow(dead_code)]
    pub(super) fn new(slice: [&'a mut [PackedM31]; N], size: usize) -> Self {
        Self {
            v: slice.map(|s| s as *mut [PackedM31]),
            chunk_size: size,
            _marker: PhantomData,
        }
    }
}

impl<'trace, const N: usize> Iterator for RowChunksMut<'trace, N> {
    type Item = [&'trace mut [PackedM31]; N];

    fn next(&mut self) -> Option<Self::Item> {
        if self.v[0].is_empty() {
            None
        } else {
            let index = std::cmp::min(self.v[0].len(), self.chunk_size);
            let mut head: [_; N] = unsafe { std::mem::zeroed() };
            let mut tail: [_; N] = unsafe { std::mem::zeroed() };
            for (i, ptr) in self.v.into_iter().enumerate() {
                // SAFETY: The self.v contract ensures that any split_at_mut is valid.
                let (lhs, rhs) = unsafe { ptr.split_at_mut(index) };
                head[i] = unsafe { &mut *lhs };
                tail[i] = rhs;
            }
            self.v = tail;
            // SAFETY: Nothing else points to or will point to the contents of this slice.
            Some(head)
        }
    }
}

// pub struct ChunkProducer<'trace, const N: usize> {
//     data: [&'trace mut [PackedM31]; N],
// }

// impl<'trace, const N: usize> Producer for ChunkProducer<'trace, N> {
//     type Item = RowView<'trace, N>;

//     fn split_at(self, index: usize) -> (Self, Self) {
//         let mut left: [&'trace mut [PackedM31]; N] = unsafe { std::mem::zeroed() };
//         let mut right: [&'trace mut [PackedM31]; N] = unsafe { std::mem::zeroed() };
//         for (i, slice) in self.data.into_iter().enumerate() {
//             let (lhs, rhs) = slice.split_at_mut(index);
//             left[i] = lhs;
//             right[i] = rhs;
//         }
//         (ChunkProducer { data: left }, ChunkProducer { data: right })
//     }

//     type IntoIter = RowViewIterator<'trace, N>;

//     fn into_iter(self) -> Self::IntoIter {
//         RowViewIterator {
//             data: self.data.map(|s| s.iter_mut()),
//         }
//     }
// }

// pub struct ParRowViewIterator<'trace, const N: usize> {
//     data: [&'trace mut [PackedM31]; N],
// }
// impl<'trace, const N: usize> ParallelIterator for ParRowViewIterator<'trace, N> {
//     type Item = RowView<'trace, N>;

//     fn drive_unindexed<C>(self, consumer: C) -> C::Result
//     where
//         C: UnindexedConsumer<Self::Item>,
//     {
//         bridge(self, consumer)
//     }

//     fn opt_len(&self) -> Option<usize> {
//         Some(self.len())
//     }
// }

// impl<'trace, const N: usize> IndexedParallelIterator for ParRowViewIterator<'trace, N> {
//     fn len(&self) -> usize {
//         self.data[0].len()
//     }

//     fn drive<C: Consumer<Self::Item>>(self, consumer: C) -> C::Result {
//         bridge(self, consumer)
//     }

//     fn with_producer<CB: ProducerCallback<Self::Item>>(self, callback: CB) -> CB::Output {
//         callback.callback(ChunkProducer { data: self.data })
//     }
// }

#[cfg(test)]
mod tests {
    use stwo_prover::core::backend::simd::m31::PackedM31;
    use stwo_prover::core::fields::m31::M31;
    use stwo_prover::core::fields::FieldExpOps;

    #[test]
    fn test_row_iter() {
        let mut trace = super::ParallelTrace::<3>::zeroed(5);
        let row_iter = trace.simd_row_chunks_mut(3);
        row_iter.enumerate().for_each(|(row_chunk_idx, row_chunk)| {
            let chunk_size = row_chunk[0].len();
            for row_in_chunk in 0..chunk_size {
                row_chunk[0][row_in_chunk] =
                    PackedM31::broadcast(M31::from(row_chunk_idx + row_in_chunk));
                row_chunk[1][row_in_chunk] =
                    PackedM31::broadcast(M31::from(row_chunk_idx + row_in_chunk + 1));
                row_chunk[2][row_in_chunk] =
                    row_chunk[0][row_in_chunk].square() + row_chunk[1][row_in_chunk].square();
            }
        });

        println!("{}", trace.pretty_print(32));
    }
}
