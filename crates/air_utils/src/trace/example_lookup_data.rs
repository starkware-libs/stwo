#![allow(unused)]
// TODO(Ohad): write a derive macro for this.
use rayon::iter::plumbing::{bridge, Consumer, Producer, ProducerCallback, UnindexedConsumer};
use rayon::prelude::*;
use stwo_air_utils_derive::Iterable;
use stwo_prover::core::backend::simd::m31::{PackedM31, N_LANES};

/// Lookup data for the example ComponentTrace.
/// Vectors are assumed to be of the same length.
#[derive(Iterable)]
pub struct LookupData {
    field0: Vec<PackedM31>,
    field1: Vec<[PackedM31; 2]>,
    field2: [Vec<[PackedM31; 2]>; 2],
}

#[cfg(test)]
mod tests {
    use itertools::{all, Itertools};
    use rayon::iter::{IndexedParallelIterator, ParallelIterator};
    use rayon::slice::ParallelSlice;
    use stwo_prover::core::backend::simd::m31::{PackedM31, LOG_N_LANES, N_LANES};
    use stwo_prover::core::fields::m31::M31;

    use crate::trace::component_trace::ComponentTrace;
    use crate::trace::example_lookup_data::LookupData;

    #[test]
    fn test_derived_par_lookup_data() {
        const N_COLUMNS: usize = 5;
        const LOG_SIZE: u32 = 8;
        let mut trace = ComponentTrace::<N_COLUMNS>::zeroed(LOG_SIZE);
        let arr = (0..1 << LOG_SIZE).map(M31::from).collect_vec();
        let mut lookup_data = unsafe { LookupData::uninitialized(LOG_SIZE - LOG_N_LANES) };
        let expected: (Vec<_>, Vec<_>, Vec<_>) = arr
            .array_chunks::<N_LANES>()
            .map(|x| {
                let x = PackedM31::from_array(*x);
                let x1 = x + PackedM31::broadcast(M31(1));
                let x2 = x + x1;
                let x3 = x + x1 + x2;
                let x4 = x + x1 + x2 + x3;
                (
                    x4,
                    [x1, x1.double()],
                    ([x2, x2.double()], [x3, x3.double()]),
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
                    *lookup_data.field0 = *row[4];
                    *lookup_data.field1 = [*row[1], row[1].double()];
                    *lookup_data.field2[0] = [*row[2], row[2].double()];
                    *lookup_data.field2[1] = [*row[3], row[3].double()];
                });
            });
        let actual = (
            lookup_data.field0,
            lookup_data.field1,
            (lookup_data.field2[0].clone(), lookup_data.field2[1].clone()),
        );

        assert!(
            all(
                expected
                    .0
                    .iter()
                    .zip(actual.0.iter())
                    .map(|(expected, actual)| (expected.to_array(), actual.to_array())),
                |(expected, actual)| expected == actual
            ),
            "Failed on Vec<PackedM31>"
        );
        assert!(
            all(
                expected
                    .1
                    .iter()
                    .zip(actual.1.iter())
                    .map(|(expected, actual)| (
                        expected.iter().flat_map(|v| v.to_array()).collect_vec(),
                        actual.iter().flat_map(|v| v.to_array()).collect_vec()
                    )),
                |(expected, actual)| expected == actual
            ),
            "Failed on Vec<[PackedM31; 2]>"
        );
        assert!(all(
            expected
                .2
                .iter()
                .map(|expected| expected.0)
                .zip(actual.2 .0.into_iter())
                .map(|(expected, actual)| (
                    expected.iter().flat_map(|v| v.to_array()).collect_vec(),
                    actual.iter().flat_map(|v| v.to_array()).collect_vec()
                )),
            |(expected, actual)| expected == actual
        ));
        assert!(
            all(
                expected
                    .2
                    .iter()
                    .map(|e| e.1)
                    .zip(actual.2 .1.into_iter())
                    .map(|(expected, actual)| (
                        expected.iter().flat_map(|v| v.to_array()).collect_vec(),
                        actual.iter().flat_map(|v| v.to_array()).collect_vec()
                    )),
                |(expected, actual)| expected == actual
            ),
            "Failed on [Vec<[PackedM31; 2]>; 2]"
        );
    }
}
