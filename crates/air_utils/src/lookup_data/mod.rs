#![allow(unused)]
// TODO(Ohad): write a derive macro for this.
use rayon::iter::plumbing::{bridge, Consumer, Producer, ProducerCallback, UnindexedConsumer};
use rayon::prelude::*;
use stwo_air_utils_derive::Uninitialized;
use stwo_prover::core::backend::simd::m31::{PackedM31, N_LANES};

/// Lookup data for the example ComponentTrace.
/// Vectors are assumed to be of the same length.
#[derive(Uninitialized)]
struct LookupData {
    field0: Vec<PackedM31>,
    field1: Vec<[PackedM31; 2]>,
    field2: [Vec<[PackedM31; 2]>; 2],
}

#[cfg(test)]
mod tests {
    use itertools::{all, assert_equal, Itertools};
    use rayon::iter::{IndexedParallelIterator, ParallelIterator};
    use rayon::slice::ParallelSlice;
    use stwo_prover::core::backend::simd::m31::{PackedM31, LOG_N_LANES, N_LANES};
    use stwo_prover::core::fields::m31::M31;

    use crate::lookup_data::LookupData;
    use crate::trace::component_trace::ComponentTrace;

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
}
