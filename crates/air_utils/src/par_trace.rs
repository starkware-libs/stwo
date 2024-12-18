use bytemuck::Zeroable;
use stwo_prover::core::backend::simd::column::BaseColumn;
use stwo_prover::core::backend::simd::m31::PackedM31;
use stwo_prover::core::backend::simd::SimdBackend;
use stwo_prover::core::fields::m31::M31;
use stwo_prover::core::poly::circle::{CanonicCoset, CircleEvaluation};
use stwo_prover::core::poly::BitReversedOrder;

/// ```
/// // 2D Matrix of packed [`M31`] values.
/// // Used as the witness for STWO proofs.
/// // Stored in column-major order, and exposes a row-major view and a parallel iterator it.
/// ```
pub struct ParallelTrace<const N: usize> {
    data: [Vec<PackedM31>; N],
    length: usize,
}

impl<const N: usize> ParallelTrace<N> {
    pub fn zeroed(log_size: u32) -> Self {
        let length = 1 << log_size;
        let data = [(); N].map(|_| vec![PackedM31::zeroed(); length]);
        Self { data, length }
    }

    /// # Safety
    /// The caller must ensure that the column is populated before being used.
    pub unsafe fn uninitialized(log_size: u32) -> Self {
        let length = 1 << log_size;
        let data = [(); N].map(|_| {
            let mut v = Vec::with_capacity(length);
            v.set_len(length);
            v
        });
        Self { data, length }
    }

    /// Length over Axis0.
    pub fn len(&self) -> usize {
        self.length
    }

    // pub fn mut_row_iter(&mut self) -> RowViewIterator<N> {
    //     let data: [_; N] = self
    //         .data
    //         .iter_mut()
    //         .map(|v| v.iter_mut())
    //         .collect_vec()
    //         .try_into()
    //         .unwrap();
    //     RowViewIterator { data }
    // }

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
}

/// Iterator over chunks of rows of a [`ParallelTrace`].
pub struct RowChunksMut<'trace, const N: usize, const C: usize> {
    data: [&'trace mut [PackedM31; C]; N],
}
// pub struct ParRowChunksMut<'trace, const N: usize, const C: usize> {
//     data: RowChunksMut<'trace, N, C>,
// }

impl<'trace, const N: usize> Iterator for RowChunksMut<'trace, N> {
    type Item = RowChunk<'trace, N>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.data[0].is_empty() {
            return None;
        }
        Some(std::array::from_fn(|i| (self.data[i].next().unwrap())))
    }
}
impl<'trace, const N: usize> ExactSizeIterator for RowViewIterator<'trace, N> {}
impl<'trace, const N: usize> DoubleEndedIterator for RowViewIterator<'trace, N> {
    fn next_back(&mut self) -> Option<Self::Item> {
        if self.data[0].is_empty() {
            return None;
        }
        Some(std::array::from_fn(|i| (self.data[i].next_back().unwrap())))
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
    #[test]
    fn test_row_iter() {}
}
