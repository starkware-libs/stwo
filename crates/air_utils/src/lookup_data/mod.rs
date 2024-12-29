#[cfg(test)]
mod tests {
    use itertools::{all, Itertools};
    use rayon::iter::{IndexedParallelIterator, IntoParallelIterator, ParallelIterator};
    use rayon::prelude::ParallelSlice;
    use stwo_air_utils_derive::{IterMut, ParMutIter, Uninitialized};
    use stwo_prover::core::backend::simd::m31::{PackedM31, LOG_N_LANES, N_LANES};
    use stwo_prover::core::fields::m31::M31;

    use crate::trace::component_trace::ComponentTrace;

    /// Lookup data for the example ComponentTrace.
    /// Vectors are assumed to be of the same length.
    #[derive(Uninitialized, IterMut, ParMutIter)]
    struct LookupData {
        field0: Vec<PackedM31>,
        field1: Vec<[PackedM31; 2]>,
        field2: [Vec<[PackedM31; 2]>; 2],
    }

    #[test]
    fn test_derived_lookup_data() {
        const LOG_SIZE: u32 = 6;
        let LookupData {
            field0,
            field1,
            field2,
        } = unsafe { LookupData::uninitialized(LOG_SIZE) };

        let lengths = [
            [field0.len()].as_slice(),
            [field1.len()].as_slice(),
            field2.map(|v| v.len()).as_slice(),
        ]
        .concat();
        assert!(all(lengths, |len| len == 1 << LOG_SIZE));
    }

    #[test]
    fn test_derived_lookup_data_par_iter() {
        const N_COLUMNS: usize = 5;
        const LOG_N_ROWS: u32 = 8;
        let mut trace = ComponentTrace::<N_COLUMNS>::zeroed(LOG_N_ROWS);
        let arr = (0..1 << LOG_N_ROWS).map(M31::from).collect_vec();
        let mut lookup_data = unsafe { LookupData::uninitialized(LOG_N_ROWS - LOG_N_LANES) };
        let (expected_field0, expected_field1): (Vec<_>, Vec<_>) = arr
            .array_chunks::<N_LANES>()
            .map(|x| {
                let x = PackedM31::from_array(*x);
                let x1 = x + PackedM31::broadcast(M31(1));
                let x2 = x + x1;
                let x3 = x + x1 + x2;
                let x4 = x + x1 + x2 + x3;
                (x4, [x1, x1.double()])
            })
            .multiunzip();

        trace
            .par_iter_mut()
            .zip(arr.par_chunks(N_LANES).into_par_iter())
            .zip(lookup_data.par_iter_mut())
            .for_each(|((row, input), lookup_data)| {
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
        let (actual0, actual1) = (lookup_data.field0, lookup_data.field1);

        assert_eq!(
            format!("{expected_field0:?}"),
            format!("{actual0:?}"),
            "Failed on Vec<PackedM31>"
        );
        assert_eq!(
            format!("{expected_field1:?}"),
            format!("{actual1:?}"),
            "Failed on Vec<[PackedM31; 2]>"
        );
    }
}
