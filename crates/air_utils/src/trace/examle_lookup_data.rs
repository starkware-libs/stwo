// TODO(Ohad): write a derive macro for this.
use stwo_air_utils_derive::StwoIterable;

// #[derive(StwoIterable)]
pub struct LookupData {
    lu3: [Vec<[PackedM31; 16]>; 4],
    lu0: Vec<[PackedM31; 2]>,
    lu1: Vec<[PackedM31; 4]>,
    lu2: Vec<[PackedM31; 8]>,
    lu4: [Vec<[PackedM31; 32]>; 4],
}

use rayon::iter::plumbing::{bridge, Consumer, Producer, ProducerCallback, UnindexedConsumer};
use rayon::prelude::*;
use stwo_prover::core::backend::simd::m31::{PackedM31, N_LANES};
impl LookupData {
    /// # Safety
    /// The caller must ensure that the trace is populated before being used.
    pub unsafe fn uninitialized(log_size: u32) -> Self {
        let length = 1 << log_size;
        let n_simd_elems = length / N_LANES;
        let mut lu0 = Vec::with_capacity(n_simd_elems);
        lu0.set_len(n_simd_elems);
        let mut lu1 = Vec::with_capacity(n_simd_elems);
        lu1.set_len(n_simd_elems);
        let mut lu2 = Vec::with_capacity(n_simd_elems);
        lu2.set_len(n_simd_elems);
        let lu3 = [(); 4].map(|_| {
            let mut vec = Vec::with_capacity(n_simd_elems);
            vec.set_len(n_simd_elems);
            vec
        });
        let lu4 = [(); 4].map(|_| {
            let mut vec = Vec::with_capacity(n_simd_elems);
            vec.set_len(n_simd_elems);
            vec
        });
        Self {
            lu3,
            lu4,
            lu0,
            lu1,
            lu2,
        }
    }
    pub fn iter_mut(&mut self) -> LookupDataIterMut<'_> {
        LookupDataIterMut::new(
            &mut self.lu0,
            &mut self.lu1,
            &mut self.lu2,
            &mut self.lu3,
            &mut self.lu4,
        )
    }
    pub fn par_iter_mut(&mut self) -> ParLookupDataIterMut<'_> {
        ParLookupDataIterMut::new(
            &mut self.lu0,
            &mut self.lu1,
            &mut self.lu2,
            &mut self.lu3,
            &mut self.lu4,
        )
    }
}
pub struct LookupDataMutChunk<'trace> {
    lu0: &'trace mut [PackedM31; 2],
    lu1: &'trace mut [PackedM31; 4],
    lu2: &'trace mut [PackedM31; 8],
    lu3: [&'trace mut [[PackedM31; 16]]; 4],
    lu4: [&'trace mut [[PackedM31; 32]]; 4],
}
pub struct LookupDataIterMut<'trace> {
    lu0: *mut [[PackedM31; 2]],
    lu1: *mut [[PackedM31; 4]],
    lu2: *mut [[PackedM31; 8]],
    lu3: [*mut [[PackedM31; 16]]; 4],
    lu4: [*mut [[PackedM31; 32]]; 4],
    phantom: std::marker::PhantomData<&'trace ()>,
}
impl<'trace> LookupDataIterMut<'trace> {
    pub fn new(
        lu0: &'trace mut [[PackedM31; 2]],
        lu1: &'trace mut [[PackedM31; 4]],
        lu2: &'trace mut [[PackedM31; 8]],
        lu3: [&'trace mut [[PackedM31; 16]]; 4],
        lu4: [&'trace mut [[PackedM31; 32]]; 4],
    ) -> Self {
        Self {
            lu0: lu0.as_mut_ptr(),
            lu1: lu1.as_mut_ptr(),
            lu2: lu2.as_mut_ptr(),
            lu3: lu3.map(|v| v.as_mut_ptr()),
            lu4: lu4.map(|v| v.as_mut_ptr()),
            phantom: std::marker::PhantomData,
        }
    }
}
impl<'trace> Iterator for LookupDataIterMut<'trace> {
    type Item = LookupDataMutChunk<'trace>;
    fn next(&mut self) -> Option<Self::Item> {
        unsafe {
            if self.lu3[0].is_empty() {
                return None;
            }
            let (lu0_head, lu0_tail) = self.lu0.split_at_mut(1);
            self.lu0 = lu0_tail;
            let (lu1_head, lu1_tail) = self.lu1.split_at_mut(1);
            self.lu1 = lu1_tail;
            let (lu2_head, lu2_tail) = self.lu2.split_at_mut(1);
            self.lu2 = lu2_tail;
            let lu3_head = self.lu3.iter_mut().map(|ptr| {
                let (head, tail) = ptr.split_at_mut(1);
                *ptr = tail;
                &mut head[0]
            });
            let lu4_head = self.lu4.iter_mut().map(|ptr| {
                let (head, tail) = ptr.split_at_mut(1);
                *ptr = tail;
                &mut head[0]
            });
            let item = LookupDataMutChunk {
                lu0: lu0_head,
                lu1: lu1_head,
                lu2: lu2_head,
                lu3: lu3_head,
                lu4: lu4_head,
            };
            Some(item)
        }
    }
    fn size_hint(&self) -> (usize, Option<usize>) {
        let len = self.lu3[0].len();
        (len, Some(len))
    }
}
impl ExactSizeIterator for LookupDataIterMut<'_> {}
impl<'trace> DoubleEndedIterator for LookupDataIterMut<'trace> {
    fn next_back(&mut self) -> Option<Self::Item> {
        if self.lu3[0].is_empty() {
            return None;
        }
        let (lu0_head, lu0_tail) = self.lu0.split_at_mut(non_array_field_names.len() - 1);
        self.lu0 = lu0_head;
        let (lu1_head, lu1_tail) = self.lu1.split_at_mut(non_array_field_names.len() - 1);
        self.lu1 = lu1_head;
        let (lu2_head, lu2_tail) = self.lu2.split_at_mut(non_array_field_names.len() - 1);
        self.lu2 = lu2_head;
        let lu3_tail = self.lu3.iter_mut().map(|ptr| {
            let (head, tail) = ptr.split_at_mut(ptr.len() - 1);
            *ptr = head;
            &mut tail[0]
        });
        let lu4_tail = self.lu4.iter_mut().map(|ptr| {
            let (head, tail) = ptr.split_at_mut(ptr.len() - 1);
            *ptr = head;
            &mut tail[0]
        });
        let item = LookupDataMutChunk {
            lu0: lu0_tail,
            lu1: lu1_tail,
            lu2: lu2_tail,
            lu3: lu3_tail,
            lu4: lu4_tail,
        };
        Some(item)
    }
}
pub struct LookupDataRowProducer<'trace> {
    lu0: &'trace mut [[PackedM31; 2]],
    lu1: &'trace mut [[PackedM31; 4]],
    lu2: &'trace mut [[PackedM31; 8]],
}
impl<'trace> Producer for LookupDataRowProducer<'trace> {
    type Item = LookupDataMutChunk<'trace>;
    type IntoIter = LookupDataIterMut<'trace>;
    fn split_at(self, index: usize) -> (Self, Self) {
        let (lu0, lu0_tail) = self.lu0.split_at_mut(index);
        let (lu1, lu1_tail) = self.lu1.split_at_mut(index);
        let (lu2, lu2_tail) = self.lu2.split_at_mut(index);
        let (lu3, lu3_tail) = self.lu3.map(|v| v.as_mut_slice()).split_at_mut(index);
        let (lu4, lu4_tail) = self.lu4.map(|v| v.as_mut_slice()).split_at_mut(index);
        (
            LookupDataRowProducer {
                lu0,
                lu1,
                lu2,
                lu3,
                lu4,
            },
            LookupDataRowProducer {
                lu0_tail,
                lu1_tail,
                lu2_tail,
                lu3_tail,
                lu4_tail,
            },
        )
    }
    fn into_iter(self) -> Self::IntoIter {
        LookupDataIterMut::new(self.lu0, self.lu1, self.lu2, self.lu3, self.lu4)
    }
}
pub struct ParLookupDataIterMut<'trace> {
    lu0: &'trace mut [[PackedM31; 2]],
    lu1: &'trace mut [[PackedM31; 4]],
    lu2: &'trace mut [[PackedM31; 8]],
    lu3: [&'trace mut [[PackedM31; 16]]; 4],
    lu4: [&'trace mut [[PackedM31; 32]]; 4],
}
impl<'trace> ParLookupDataIterMut<'trace> {
    pub fn new(
        lu0: &'trace mut [[PackedM31; 2]],
        lu1: &'trace mut [[PackedM31; 4]],
        lu2: &'trace mut [[PackedM31; 8]],
        lu3: [&'trace mut [[PackedM31; 16]]; 4],
        lu4: [&'trace mut [[PackedM31; 32]]; 4],
    ) -> Self {
        Self {
            lu0,
            lu1,
            lu2,
            lu3,
            lu4,
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
        self.lu3[0].len()
    }
    fn drive<D: Consumer<Self::Item>>(self, consumer: D) -> D::Result {
        bridge(self, consumer)
    }
    fn with_producer<CB: ProducerCallback<Self::Item>>(self, callback: CB) -> CB::Output {
        callback.callback(LookupDataRowProducer {
            lu0: self.lu0,
            lu1: self.lu1,
            lu2: self.lu2,
            lu3: self.lu3,
            lu4: self.lu4,
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
    use crate::trace::examle_lookup_data::LookupData;

    #[test]
    fn test_lookup_data() {
        const N_COLUMNS: usize = 5;
        const LOG_SIZE: u32 = 8;
        let mut trace = ComponentTrace::<N_COLUMNS>::zeroed(LOG_SIZE);
        let arr = (0..1 << LOG_SIZE).map(M31::from).collect_vec();
        let mut lookup_data = unsafe { LookupData::uninitialized(LOG_SIZE) };
        let expected: (Vec<_>, Vec<_>, Vec<_>) = arr
            .array_chunks::<N_LANES>()
            .map(|x| {
                let x = PackedM31::from_array(*x);
                let x1 = x + PackedM31::broadcast(M31(1));
                let x2 = x + x1;
                let x3 = x + x1 + x2;
                let x4 = x + x1 + x2 + x3;
                (
                    [x, x4],
                    [x1, x1.double(), x2, x2.double()],
                    [
                        x3,
                        x3.double(),
                        x4,
                        x4.double(),
                        x,
                        x.double(),
                        x1,
                        x1.double(),
                    ],
                )
            })
            .multiunzip();

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
                    *lookup_data.lu2 = [
                        *row[3],
                        row[3].double(),
                        *row[4],
                        row[4].double(),
                        *row[0],
                        row[0].double(),
                        *row[1],
                        row[1].double(),
                    ];
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
