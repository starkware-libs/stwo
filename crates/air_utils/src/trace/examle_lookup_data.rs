// TODO(Ohad): write a derive macro for this.
use stwo_air_utils_derive::StwoIterable;

#[derive(StwoIterable)]
pub struct LookupData {
    lu0: Vec<[PackedM31; 2]>,
    lu1: Vec<[PackedM31; 4]>,
    lu2: Vec<[PackedM31; 8]>,
    lu3: [Vec<[PackedM31; 16]>; 4],
}

#[cfg(test)]
mod tests {
    use itertools::{all, Itertools};
    use rayon::iter::{IndexedParallelIterator, ParallelIterator};
    use rayon::slice::ParallelSlice;
    use stwo_prover::core::backend::simd::m31::{PackedM31, N_LANES};
    use stwo_prover::core::fields::m31::M31;

    use crate::trace::examle_lookup_data::LookupData;
    use crate::trace::component_trace::ComponentTrace;

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
