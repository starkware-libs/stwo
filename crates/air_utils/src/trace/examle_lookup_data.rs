// TODO(Ohad): write a derive macro for this.
use rayon::iter::plumbing::{bridge, Consumer, Producer, ProducerCallback, UnindexedConsumer};
use rayon::prelude::*;
use stwo_prover::core::backend::simd::m31::{PackedM31, N_LANES};

pub struct LookupData {
    pub lu0: Vec<[PackedM31; 2]>,
    pub lu1: Vec<[PackedM31; 4]>,
}
impl LookupData {
    /// # Safety
    pub unsafe fn uninitialized(log_size: u32) -> Self {
        let length = 1 << log_size;
        let n_simd_elems = length / N_LANES;
        let mut lu0 = Vec::with_capacity(n_simd_elems);
        let mut lu1 = Vec::with_capacity(n_simd_elems);
        lu0.set_len(n_simd_elems);
        lu1.set_len(n_simd_elems);

        Self { lu0, lu1 }
    }

    pub fn iter_mut(&mut self) -> LookupDataIterMut {
        LookupDataIterMut::new(&mut self.lu0, &mut self.lu1)
    }

    pub fn par_iter_mut(&mut self) -> ParLookupDataIterMut {
        ParLookupDataIterMut {
            lu0: &mut self.lu0,
            lu1: &mut self.lu1,
        }
    }
}

pub struct LookupDataMutChunk<'trace> {
    pub lu0: &'trace mut [PackedM31; 2],
    pub lu1: &'trace mut [PackedM31; 4],
}
pub struct LookupDataIterMut<'trace> {
    lu0: *mut [[PackedM31; 2]],
    lu1: *mut [[PackedM31; 4]],
    phantom: std::marker::PhantomData<&'trace ()>,
}
impl<'trace> LookupDataIterMut<'trace> {
    pub fn new(slice0: &'trace mut [[PackedM31; 2]], slice1: &'trace mut [[PackedM31; 4]]) -> Self {
        Self {
            lu0: slice0 as *mut _,
            lu1: slice1 as *mut _,
            phantom: std::marker::PhantomData,
        }
    }
}
impl<'trace> Iterator for LookupDataIterMut<'trace> {
    type Item = LookupDataMutChunk<'trace>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.lu0.is_empty() {
            return None;
        }
        let item = unsafe {
            let (head0, tail0) = self.lu0.split_at_mut(1);
            let (head1, tail1) = self.lu1.split_at_mut(1);
            self.lu0 = tail0;
            self.lu1 = tail1;
            LookupDataMutChunk {
                lu0: &mut (*head0)[0],
                lu1: &mut (*head1)[0],
            }
        };
        Some(item)
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let len = self.lu0.len();
        (len, Some(len))
    }
}

impl ExactSizeIterator for LookupDataIterMut<'_> {}
impl DoubleEndedIterator for LookupDataIterMut<'_> {
    fn next_back(&mut self) -> Option<Self::Item> {
        if self.lu0.is_empty() {
            return None;
        }
        let item = unsafe {
            let (head0, tail0) = self.lu0.split_at_mut(self.lu0.len() - 1);
            let (head1, tail1) = self.lu1.split_at_mut(self.lu1.len() - 1);
            self.lu0 = head0;
            self.lu1 = head1;
            LookupDataMutChunk {
                lu0: &mut (*tail0)[0],
                lu1: &mut (*tail1)[0],
            }
        };
        Some(item)
    }
}

struct RowProducer<'trace> {
    lu0: &'trace mut [[PackedM31; 2]],
    lu1: &'trace mut [[PackedM31; 4]],
}

impl<'trace> Producer for RowProducer<'trace> {
    type Item = LookupDataMutChunk<'trace>;

    fn split_at(self, index: usize) -> (Self, Self) {
        let (lu0, rh0) = self.lu0.split_at_mut(index);
        let (lu1, rh1) = self.lu1.split_at_mut(index);
        (RowProducer { lu0, lu1 }, RowProducer { lu0: rh0, lu1: rh1 })
    }

    type IntoIter = LookupDataIterMut<'trace>;

    fn into_iter(self) -> Self::IntoIter {
        LookupDataIterMut::new(self.lu0, self.lu1)
    }
}

pub struct ParLookupDataIterMut<'trace> {
    lu0: &'trace mut [[PackedM31; 2]],
    lu1: &'trace mut [[PackedM31; 4]],
}

impl<'trace> ParLookupDataIterMut<'trace> {
    pub fn new(slice0: &'trace mut [[PackedM31; 2]], slice1: &'trace mut [[PackedM31; 4]]) -> Self {
        Self {
            lu0: slice0,
            lu1: slice1,
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
        self.lu0.len()
    }

    fn drive<D: Consumer<Self::Item>>(self, consumer: D) -> D::Result {
        bridge(self, consumer)
    }

    fn with_producer<CB: ProducerCallback<Self::Item>>(self, callback: CB) -> CB::Output {
        callback.callback(RowProducer {
            lu0: self.lu0,
            lu1: self.lu1,
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

    use crate::trace::examle_lookup_data::LookupData;
    use crate::trace::iterable_trace::IterableTrace;

    #[test]
    fn test_lookup_data() {
        const N_COLUMNS: usize = 5;
        const LOG_SIZE: u32 = 8;
        let mut trace = IterableTrace::<N_COLUMNS>::zeroed(LOG_SIZE);
        let arr = (0..1 << LOG_SIZE).map(M31::from).collect_vec();
        let mut lookup_data = unsafe { LookupData::uninitialized(LOG_SIZE) };
        let expected: (Vec<_>, Vec<_>) = arr
            .array_chunks::<N_LANES>()
            .map(|x| {
                let x = PackedM31::from_array(*x);
                let x1 = x + PackedM31::broadcast(M31(1));
                let x2 = x + x1;
                let x3 = x + x1 + x2;
                let x4 = x + x1 + x2 + x3;
                ([x, x4], [x1, x1.double(), x2, x2.double()])
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
                    *row[3] = *row[0] + *row[1] + *row[2];
                    *row[4] = *row[0] + *row[1] + *row[2] + *row[3];
                    *lookup_data.lu0 = [*row[0], *row[4]];
                    *lookup_data.lu1 = [*row[1], row[1].double(), *row[2], row[2].double()];
                })
            });

        assert!(all(
            lookup_data.lu0.into_iter().zip(expected.0),
            |(actual, expected)| actual[0].to_array() == expected[0].to_array()
                && actual[1].to_array() == expected[1].to_array()
        ));
        assert!(all(
            lookup_data.lu1.into_iter().zip(expected.1),
            |(actual, expected)| {
                actual[0].to_array() == expected[0].to_array()
                    && actual[1].to_array() == expected[1].to_array()
                    && actual[2].to_array() == expected[2].to_array()
                    && actual[3].to_array() == expected[3].to_array()
            }
        ));
    }
}
