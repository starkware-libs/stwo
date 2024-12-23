// TODO(Ohad): write a derive macro for this.
use rayon::iter::plumbing::{bridge, Consumer, Producer, ProducerCallback, UnindexedConsumer};
use rayon::prelude::*;
use stwo_prover::core::backend::simd::m31::{PackedM31, LOG_N_LANES};

/// Lookup data for the example ComponentTrace.
/// field0 and field1 are expected to be of the same length.
pub struct LookupData {
    field0: Vec<PackedM31>,
    field1: Vec<[PackedM31; 2]>,
}
impl LookupData {
    /// # Safety
    /// The caller must ensure that the fields are populated before being used.
    pub unsafe fn uninitialized(log_size: u32) -> Self {
        let n_simd_elems = 1 << (log_size - LOG_N_LANES);
        let mut field0 = Vec::with_capacity(n_simd_elems);
        let mut field1 = Vec::with_capacity(n_simd_elems);
        field0.set_len(n_simd_elems);
        field1.set_len(n_simd_elems);

        Self { field0, field1 }
    }

    pub fn iter_mut(&mut self) -> LookupDataIterMut<'_> {
        LookupDataIterMut::new(&mut self.field0, &mut self.field1)
    }

    pub fn par_iter_mut(&mut self) -> ParLookupDataIterMut<'_> {
        ParLookupDataIterMut {
            field0: &mut self.field0,
            field1: &mut self.field1,
        }
    }
}

pub struct LookupDataMutChunk<'trace> {
    pub field0: &'trace mut PackedM31,
    pub field1: &'trace mut [PackedM31; 2],
}
pub struct LookupDataIterMut<'trace> {
    field0: *mut [PackedM31],
    field1: *mut [[PackedM31; 2]],
    phantom: std::marker::PhantomData<&'trace ()>,
}
impl<'trace> LookupDataIterMut<'trace> {
    pub fn new(slice0: &'trace mut [PackedM31], slice1: &'trace mut [[PackedM31; 2]]) -> Self {
        Self {
            field0: slice0 as *mut _,
            field1: slice1 as *mut _,
            phantom: std::marker::PhantomData,
        }
    }
}
impl<'trace> Iterator for LookupDataIterMut<'trace> {
    type Item = LookupDataMutChunk<'trace>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.field0.is_empty() {
            return None;
        }
        let item = unsafe {
            let (head0, tail0) = self.field0.split_at_mut(1);
            let (head1, tail1) = self.field1.split_at_mut(1);
            self.field0 = tail0;
            self.field1 = tail1;
            LookupDataMutChunk {
                field0: &mut (*head0)[0],
                field1: &mut (*head1)[0],
            }
        };
        Some(item)
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let len = self.field0.len();
        (len, Some(len))
    }
}

impl ExactSizeIterator for LookupDataIterMut<'_> {}
impl DoubleEndedIterator for LookupDataIterMut<'_> {
    fn next_back(&mut self) -> Option<Self::Item> {
        if self.field0.is_empty() {
            return None;
        }
        let item = unsafe {
            let (head0, tail0) = self.field0.split_at_mut(self.field0.len() - 1);
            let (head1, tail1) = self.field1.split_at_mut(self.field1.len() - 1);
            self.field0 = head0;
            self.field1 = head1;
            LookupDataMutChunk {
                field0: &mut (*tail0)[0],
                field1: &mut (*tail1)[0],
            }
        };
        Some(item)
    }
}

struct RowProducer<'trace> {
    field0: &'trace mut [PackedM31],
    field1: &'trace mut [[PackedM31; 2]],
}

impl<'trace> Producer for RowProducer<'trace> {
    type Item = LookupDataMutChunk<'trace>;

    fn split_at(self, index: usize) -> (Self, Self) {
        let (field0, rh0) = self.field0.split_at_mut(index);
        let (field1, rh1) = self.field1.split_at_mut(index);
        (
            RowProducer { field0, field1 },
            RowProducer {
                field0: rh0,
                field1: rh1,
            },
        )
    }

    type IntoIter = LookupDataIterMut<'trace>;

    fn into_iter(self) -> Self::IntoIter {
        LookupDataIterMut::new(self.field0, self.field1)
    }
}

pub struct ParLookupDataIterMut<'trace> {
    field0: &'trace mut [PackedM31],
    field1: &'trace mut [[PackedM31; 2]],
}

impl<'trace> ParLookupDataIterMut<'trace> {
    pub fn new(slice0: &'trace mut [PackedM31], slice1: &'trace mut [[PackedM31; 2]]) -> Self {
        Self {
            field0: slice0,
            field1: slice1,
        }
    }
}

impl<'trace> ParallelIterator for ParLookupDataIterMut<'trace> {
    type Item = LookupDataMutChunk<'trace>;

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

impl IndexedParallelIterator for ParLookupDataIterMut<'_> {
    fn len(&self) -> usize {
        self.field0.len()
    }

    fn drive<D: Consumer<Self::Item>>(self, consumer: D) -> D::Result {
        bridge(self, consumer)
    }

    fn with_producer<CB: ProducerCallback<Self::Item>>(self, callback: CB) -> CB::Output {
        callback.callback(RowProducer {
            field0: self.field0,
            field1: self.field1,
        })
    }
}

#[cfg(test)]
mod tests {
    use itertools::{all, Itertools};
    use rayon::iter::{IndexedParallelIterator, ParallelIterator};
    use rayon::slice::ParallelSlice;
    use stwo_prover::core::backend::simd::m31::{PackedM31, N_LANES};
    use stwo_prover::core::fields::m31::M31;

    use crate::trace::component_trace::ComponentTrace;
    use crate::trace::example_lookup_data::LookupData;

    #[test]
    fn test_lookup_data() {
        const N_COLUMNS: usize = 3;
        const LOG_SIZE: u32 = 8;
        let mut trace = ComponentTrace::<N_COLUMNS>::zeroed(LOG_SIZE);
        let arr = (0..1 << LOG_SIZE).map(M31::from).collect_vec();
        let mut lookup_data = unsafe { LookupData::uninitialized(LOG_SIZE) };
        let expected: (Vec<_>, Vec<_>) = arr
            .array_chunks::<N_LANES>()
            .map(|x| {
                let x = PackedM31::from_array(*x);
                let x1 = x + PackedM31::broadcast(M31(1));
                let x2 = x + x1;
                (x, [x1, x2.double()])
            })
            .unzip();

        trace
            .par_iter_mut()
            .zip(arr.par_chunks(N_LANES))
            .zip(lookup_data.par_iter_mut())
            .chunks(4)
            .for_each(|chunk| {
                chunk.into_iter().for_each(|((row, input), lookup_data)| {
                    *row[0] = PackedM31::from_array(input.try_into().unwrap());
                    *row[1] = *row[0] + PackedM31::broadcast(M31(1));
                    *row[2] = *row[0] + *row[1];
                    *lookup_data.field0 = *row[0];
                    *lookup_data.field1 = [*row[1], row[2].double()];
                })
            });

        assert!(all(
            lookup_data.field0.into_iter().zip(expected.0),
            |(actual, expected)| actual.to_array() == expected.to_array()
                && actual.to_array() == expected.to_array()
        ));
        assert!(all(
            lookup_data.field1.into_iter().zip(expected.1),
            |(actual, expected)| {
                actual[0].to_array() == expected[0].to_array()
                    && actual[1].to_array() == expected[1].to_array()
            }
        ));
    }
}
