use super::utils::transmute_col_refs;
use super::WebBackend;
use crate::core::backend::simd::SimdBackend;
use crate::core::backend::{Col, ColumnOps};
use crate::core::fields::m31::BaseField;
use crate::core::vcs::blake2_hash::Blake2sHash;
use crate::core::vcs::blake2_merkle::Blake2sMerkleHasher;
use crate::core::vcs::ops::MerkleOps;

impl ColumnOps<Blake2sHash> for WebBackend {
    type Column = Vec<Blake2sHash>;

    fn bit_reverse_column(_column: &mut Self::Column) {
        unimplemented!()
    }
}

impl MerkleOps<Blake2sMerkleHasher> for WebBackend {
    fn commit_on_layer(
        log_size: u32,
        prev_layer: Option<&Vec<Blake2sHash>>,
        columns: &[&Col<Self, BaseField>],
    ) -> Vec<Blake2sHash> {
        <SimdBackend as MerkleOps<Blake2sMerkleHasher>>::commit_on_layer(
            log_size,
            prev_layer,
            transmute_col_refs(columns),
        )
    }
}

#[cfg(test)]
mod tests {
    use std::array;
    use std::mem::transmute;
    use std::simd::u32x16;

    use aligned::{Aligned, A64};

    use crate::core::backend::simd::blake2s::{compress16, transpose_msgs, untranspose_states};
    use crate::core::vcs::blake2s_ref::compress;

    #[test]
    fn compress16_works() {
        let states: Aligned<A64, [[u32; 8]; 16]> =
            Aligned(array::from_fn(|i| array::from_fn(|j| (i + j) as u32)));
        let msgs: Aligned<A64, [[u32; 16]; 16]> =
            Aligned(array::from_fn(|i| array::from_fn(|j| (i + j + 20) as u32)));
        let count_low = 1;
        let count_high = 2;
        let lastblock = 3;
        let lastnode = 4;
        let res_unvectorized = array::from_fn(|i| {
            compress(
                states[i], msgs[i], count_low, count_high, lastblock, lastnode,
            )
        });

        let res_vectorized: [[u32; 8]; 16] = unsafe {
            transmute(untranspose_states(compress16(
                transpose_states(transmute::<Aligned<A64, [[u32; 8]; 16]>, [u32x16; 8]>(
                    states,
                )),
                transpose_msgs(transmute::<Aligned<A64, [[u32; 16]; 16]>, [u32x16; 16]>(
                    msgs,
                )),
                u32x16::splat(count_low),
                u32x16::splat(count_high),
                u32x16::splat(lastblock),
                u32x16::splat(lastnode),
            )))
        };

        assert_eq!(res_vectorized, res_unvectorized);
    }

    #[test]
    fn untranspose_states_is_transpose_states_inverse() {
        let states = array::from_fn(|i| u32x16::from(array::from_fn(|j| (i + j) as u32)));
        let transposed_states = transpose_states(states);

        let untrasponsed_transposed_states = untranspose_states(transposed_states);

        assert_eq!(untrasponsed_transposed_states, states)
    }

    /// Transposes states, from 8 packed words, to get 16 results, each of size 32B.
    fn transpose_states(mut states: [u32x16; 8]) -> [u32x16; 8] {
        // Index abc:xyzw, refers to a specific word in data as follows:
        //   abc - chunk index (in base 2)
        //   xyzw - word offset (in base 2)
        // Transpose by applying 3 times the index permutation:
        //   abc:xyzw => wab:cxyz
        // In other words, rotate the index to the right by 1.
        for _ in 0..3 {
            let (s0, s4) = states[0].deinterleave(states[1]);
            let (s1, s5) = states[2].deinterleave(states[3]);
            let (s2, s6) = states[4].deinterleave(states[5]);
            let (s3, s7) = states[6].deinterleave(states[7]);
            states = [s0, s1, s2, s3, s4, s5, s6, s7];
        }

        states
    }
}
