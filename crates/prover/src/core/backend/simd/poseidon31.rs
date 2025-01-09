use std::array;

use itertools::Itertools;
use num_traits::Zero;
#[cfg(feature = "parallel")]
use rayon::prelude::*;

use crate::core::backend::simd::m31::{PackedM31, LOG_N_LANES};
use crate::core::backend::simd::SimdBackend;
use crate::core::backend::{Col, Column, ColumnOps};
use crate::core::fields::m31::BaseField;
use crate::core::vcs::ops::{MerkleHasher, MerkleOps};
use crate::core::vcs::poseidon31_hash::Poseidon31Hash;
use crate::core::vcs::poseidon31_merkle::Poseidon31MerkleHasher;
use crate::core::vcs::poseidon31_ref::{
    FIRST_FOUR_ROUND_RC, LAST_FOUR_ROUNDS_RC, MAT_DIAG16_M_1, PARTIAL_ROUNDS_RC,
};
use crate::parallel_iter;

impl ColumnOps<Poseidon31Hash> for SimdBackend {
    type Column = Vec<Poseidon31Hash>;

    fn bit_reverse_column(_column: &mut Self::Column) {
        unimplemented!()
    }
}

impl MerkleOps<Poseidon31MerkleHasher> for SimdBackend {
    fn commit_on_layer(
        log_size: u32,
        prev_layer: Option<&Vec<Poseidon31Hash>>,
        columns: &[&Col<Self, BaseField>],
    ) -> Vec<Poseidon31Hash> {
        if log_size < LOG_N_LANES {
            return parallel_iter!(0..1 << log_size)
                .map(|i| {
                    Poseidon31MerkleHasher::hash_node(
                        prev_layer.map(|prev_layer| (prev_layer[2 * i], prev_layer[2 * i + 1])),
                        &columns.iter().map(|column| column.at(i)).collect_vec(),
                    )
                })
                .collect();
        }

        if let Some(prev_layer) = prev_layer {
            assert_eq!(prev_layer.len(), 1 << (log_size + 1));
        }

        let mut res = vec![Poseidon31Hash::default(); 1 << log_size];
        #[cfg(not(feature = "parallel"))]
        let iter = res.chunks_mut(1 << LOG_N_LANES);

        #[cfg(feature = "parallel")]
        let iter = res.par_chunks_mut(1 << LOG_N_LANES);

        iter.enumerate().for_each(|(i, chunk)| {
            let hash_tree = if let Some(prev_layer) = prev_layer {
                let input = &prev_layer[(i << 5)..((i + 1) << 5)];

                let mut packed_input = [PackedM31::zero(); 16];
                for i in 0..8 {
                    packed_input[i] = PackedM31::from_array(array::from_fn(|k| input[2 * k].0[i]));
                }
                for i in 0..8 {
                    packed_input[8 + i] =
                        PackedM31::from_array(array::from_fn(|k| input[2 * k + 1].0[i]))
                }

                Some(compress16(packed_input))
            } else {
                None
            };

            let hash_column = if !columns.is_empty() {
                let len = columns.len();
                let num_chunk = len.div_ceil(8);

                let mut digest = if num_chunk == 1 {
                    let mut res = [PackedM31::zero(); 8];
                    for j in 0..len {
                        res[j] = columns[j].data[i];
                    }
                    res
                } else {
                    let mut res = [PackedM31::zero(); 16];
                    for j in 0..16 {
                        res[j] = columns[j].data[i];
                    }
                    compress16(res)
                };

                for column_chunk in columns.chunks_exact(8).skip(2) {
                    let mut state = [PackedM31::zero(); 16];
                    for j in 0..8 {
                        state[j] = digest[j];
                    }
                    for j in 0..8 {
                        state[j + 8] = column_chunk[j].data[i];
                    }
                    digest = compress16(state);
                }

                let remain = len % 8;
                if remain != 0 {
                    let mut state = [PackedM31::zero(); 16];
                    for j in 0..8 {
                        state[j] = digest[j];
                    }
                    for j in 0..remain {
                        state[j + 8] = columns[len - remain + j].data[i];
                    }
                    digest = compress16(state);
                }

                Some(digest)
            } else {
                None
            };

            let final_res = match (hash_tree, hash_column) {
                (Some(hash_tree), Some(hash_column)) => {
                    let mut state = [PackedM31::zero(); 16];
                    for j in 0..8 {
                        state[j] = hash_tree[j];
                    }
                    for j in 0..8 {
                        state[j + 8] = hash_column[j];
                    }
                    compress16(state)
                }
                (Some(hash_tree), None) => hash_tree,
                (None, Some(hash_column)) => hash_column,
                _ => unreachable!(),
            };

            for j in 0..8 {
                let unpacked = final_res[j].to_array();
                for k in 0..16 {
                    chunk[k].0[j] = unpacked[k];
                }
            }
        });
        res
    }
}

fn apply_4x4_mds_matrix(
    x0: PackedM31,
    x1: PackedM31,
    x2: PackedM31,
    x3: PackedM31,
) -> [PackedM31; 4] {
    let t0 = x0 + x1;
    let t1 = x2 + x3;
    let t2 = x1.double() + t1;
    let t3 = x3.double() + t0;
    let t4 = t1.double().double() + t3;
    let t5 = t0.double().double() + t2;
    let t6 = t3 + t5;
    let t7 = t2 + t4;

    [t6, t5, t7, t4]
}

fn apply_16x16_mds_matrix(state: &mut [PackedM31; 16]) {
    let p1 = apply_4x4_mds_matrix(state[0], state[1], state[2], state[3]);
    let p2 = apply_4x4_mds_matrix(state[4], state[5], state[6], state[7]);
    let p3 = apply_4x4_mds_matrix(state[8], state[9], state[10], state[11]);
    let p4 = apply_4x4_mds_matrix(state[12], state[13], state[14], state[15]);

    let t = [
        p1[0] + p2[0] + p3[0] + p4[0],
        p1[1] + p2[1] + p3[1] + p4[1],
        p1[2] + p2[2] + p3[2] + p4[2],
        p1[3] + p2[3] + p3[3] + p4[3],
    ];

    for i in 0..4 {
        state[i] = p1[i] + t[i];
        state[i + 4] = p2[i] + t[i];
        state[i + 8] = p3[i] + t[i];
        state[i + 12] = p4[i] + t[i];
    }
}

fn pow5(v: PackedM31) -> PackedM31 {
    let t = v * v;
    t * t * v
}

pub(crate) fn permute(mut state: [PackedM31; 16]) -> [PackedM31; 16] {
    apply_16x16_mds_matrix(&mut state);

    for r in 0..4 {
        for i in 0..16 {
            state[i] += PackedM31::broadcast(FIRST_FOUR_ROUND_RC[r][i]);
        }
        for i in 0..16 {
            state[i] = pow5(state[i]);
        }

        apply_16x16_mds_matrix(&mut state);
    }

    for r in 0..14 {
        state[0] += PackedM31::broadcast(PARTIAL_ROUNDS_RC[r]);
        state[0] = pow5(state[0]);

        let mut sum = state[0];
        for i in 1..16 {
            sum += state[i];
        }

        for i in 0..16 {
            state[i] = sum + state[i] * PackedM31::broadcast(MAT_DIAG16_M_1[i]);
        }
    }

    for r in 0..4 {
        for i in 0..16 {
            state[i] += PackedM31::broadcast(LAST_FOUR_ROUNDS_RC[r][i]);
        }
        for i in 0..16 {
            state[i] = pow5(state[i]);
        }

        apply_16x16_mds_matrix(&mut state);
    }
    state
}

fn compress16(state: [PackedM31; 16]) -> [PackedM31; 8] {
    let permuted_state = permute(state);
    let mut res = permuted_state.first_chunk::<8>().unwrap().clone();
    for i in 0..8 {
        res[i] += state[i];
    }
    res
}

#[cfg(test)]
mod test {
    use num_traits::Zero;
    use rand::rngs::SmallRng;
    use rand::{Rng, SeedableRng};

    use crate::core::backend::simd::m31::PackedM31;
    use crate::core::backend::simd::poseidon31::compress16;
    use crate::core::backend::simd::SimdBackend;
    use crate::core::backend::{Col, CpuBackend};
    use crate::core::fields::m31::{BaseField, M31};
    use crate::core::vcs::ops::MerkleOps;
    use crate::core::vcs::poseidon31_merkle::Poseidon31MerkleHasher;
    use crate::core::vcs::poseidon31_ref::poseidon2_permute;

    #[test]
    fn test_permute_consistency() {
        let mut prng = SmallRng::seed_from_u64(0);
        let test_inputs: [[M31; 16]; 16] = prng.gen();
        let mut test_outputs = [[M31::zero(); 8]; 16];

        for i in 0..16 {
            let mut state = test_inputs[i].clone();
            poseidon2_permute(&mut state);
            for j in 0..8 {
                test_outputs[i][j] = state[j] + test_inputs[i][j];
            }
        }

        let mut packed_inputs = [PackedM31::zero(); 16];
        for i in 0..16 {
            packed_inputs[i] = PackedM31::from_array([
                test_inputs[0][i],
                test_inputs[1][i],
                test_inputs[2][i],
                test_inputs[3][i],
                test_inputs[4][i],
                test_inputs[5][i],
                test_inputs[6][i],
                test_inputs[7][i],
                test_inputs[8][i],
                test_inputs[9][i],
                test_inputs[10][i],
                test_inputs[11][i],
                test_inputs[12][i],
                test_inputs[13][i],
                test_inputs[14][i],
                test_inputs[15][i],
            ]);
        }

        let packed_output = compress16(packed_inputs);
        for i in 0..8 {
            let arr = packed_output[i].to_array();
            for j in 0..16 {
                assert_eq!(arr[j], test_outputs[j][i]);
            }
        }
    }

    #[test]
    fn test_merkle_consistency() {
        let mut prng = SmallRng::seed_from_u64(0);

        let mut data = Vec::with_capacity(1 << 10);
        for _ in 0..(1 << 10) {
            data.push(prng.gen::<M31>())
        }

        let cpu_col: Col<CpuBackend, BaseField> = data.iter().copied().collect();
        let cpu_result = <CpuBackend as MerkleOps<Poseidon31MerkleHasher>>::commit_on_layer(
            10,
            None,
            &[&cpu_col],
        );

        let simd_col: Col<SimdBackend, BaseField> = data.iter().copied().collect();
        let simd_result = <SimdBackend as MerkleOps<Poseidon31MerkleHasher>>::commit_on_layer(
            10,
            None,
            &[&simd_col],
        );
        assert_eq!(cpu_result, simd_result);

        let cpu_result = <CpuBackend as MerkleOps<Poseidon31MerkleHasher>>::commit_on_layer(
            9,
            Some(&cpu_result),
            &[],
        );
        let simd_result = <SimdBackend as MerkleOps<Poseidon31MerkleHasher>>::commit_on_layer(
            9,
            Some(&simd_result),
            &[],
        );
        assert_eq!(cpu_result, simd_result);

        let mut data = Vec::with_capacity(1 << 8);
        for _ in 0..(1 << 8) {
            data.push(prng.gen::<M31>())
        }

        let cpu_col: Col<CpuBackend, BaseField> = data.iter().copied().collect();
        let cpu_result = <CpuBackend as MerkleOps<Poseidon31MerkleHasher>>::commit_on_layer(
            8,
            Some(&cpu_result),
            &[&cpu_col],
        );
        let simd_col: Col<SimdBackend, BaseField> = data.iter().copied().collect();
        let simd_result = <SimdBackend as MerkleOps<Poseidon31MerkleHasher>>::commit_on_layer(
            8,
            Some(&simd_result),
            &[&simd_col],
        );
        assert_eq!(cpu_result, simd_result);
    }
}
