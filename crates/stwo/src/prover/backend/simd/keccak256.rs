//! SIMD-backend bindings for the Keccak256 channel/merkle path.
//!
//! - `MerkleOps<Keccak256MerkleHasher>` is implemented using parallel-permutation
//!   `keccak::parallel`. With `RUSTFLAGS='--cfg keccak_backend="simd256"'` (or `simd128`/`simd512`)
//!   this dispatches to vectorized `f1600x{2,4,8}`. Without any cfg, it falls back gracefully to
//!   the scalar soft backend (parallel width `U1`), which is still correct.
//! - `MerkleOpsLifted` and `GrindOps` here remain naive delegations to the CPU path; optimizing
//!   them is tracked separately.

use core::mem;

use itertools::Itertools;
use keccak::{Backend, BackendClosure, Keccak, ParState1600, State1600};

use super::SimdBackend;
use crate::core::channel::{Channel, Keccak256Channel};
use crate::core::fields::m31::BaseField;
use crate::core::proof_of_work::GrindOps;
use crate::core::vcs::keccak256_hash::Keccak256Hash;
use crate::core::vcs::keccak256_merkle::Keccak256MerkleHasher;
use crate::core::vcs_lifted::keccak256_merkle::Keccak256MerkleHasher as Keccak256MerkleHasherLifted;
use crate::prover::backend::{Col, Column, ColumnOps, CpuBackend};
use crate::prover::vcs::ops::MerkleOps;
use crate::prover::vcs_lifted::ops::MerkleOpsLifted;

/// Keccak256 sponge rate (1088 bits = 17 × 64-bit lanes = 136 bytes).
const KECCAK256_RATE: usize = 136;
const KECCAK256_RATE_WORDS: usize = KECCAK256_RATE / 8;

impl ColumnOps<Keccak256Hash> for SimdBackend {
    type Column = Vec<Keccak256Hash>;

    fn bit_reverse_column(_column: &mut Self::Column) {
        unimplemented!()
    }
}

// TODO: optimize by parallelizing over nonces with `keccak::parallel`.
impl GrindOps<Keccak256Channel> for SimdBackend {
    fn grind(channel: &Keccak256Channel, pow_bits: u32) -> u64 {
        let mut nonce = 0u64;
        loop {
            if channel.verify_pow_nonce(pow_bits, nonce) {
                return nonce;
            }
            nonce += 1;
        }
    }
}

impl MerkleOps<Keccak256MerkleHasher> for SimdBackend {
    fn commit_on_layer(
        log_size: u32,
        prev_layer: Option<&Vec<Keccak256Hash>>,
        columns: &[&Col<Self, BaseField>],
    ) -> Vec<Keccak256Hash> {
        let n_nodes = 1usize << log_size;

        // Each node's input = (children if present) || (column values, BE-packed M31s).
        // All nodes in this layer share the same input length.
        let inputs: Vec<Vec<u8>> = (0..n_nodes)
            .map(|i| {
                let cap = if prev_layer.is_some() { 64 } else { 0 } + 4 * columns.len();
                let mut buf = Vec::with_capacity(cap);
                if let Some(prev) = prev_layer {
                    buf.extend_from_slice(&prev[2 * i].0);
                    buf.extend_from_slice(&prev[2 * i + 1].0);
                }
                for col in columns {
                    buf.extend_from_slice(&col.at(i).0.to_be_bytes());
                }
                buf
            })
            .collect();

        let mut output = vec![Keccak256Hash::default(); n_nodes];
        Keccak::new().with_backend(LayerHasher {
            inputs: &inputs,
            output: &mut output,
        });
        output
    }
}

/// Sponge-construction closure dispatched against the runtime-selected `keccak::Backend`.
/// `call_once` is generic over `B`, so the parallel/scalar permutation function pointers
/// are monomorphized at the right SIMD width.
struct LayerHasher<'a> {
    inputs: &'a [Vec<u8>],
    output: &'a mut [Keccak256Hash],
}

impl BackendClosure for LayerHasher<'_> {
    fn call_once<B: Backend>(self) {
        let par_f1600 = B::get_par_f1600();
        let f1600 = B::get_f1600();

        // Parallel width = how many State1600s fit in one ParState1600<B>.
        // Avoids depending on `hybrid_array::ArraySize` to read the typenum constant.
        let par_width = mem::size_of::<ParState1600<B>>() / mem::size_of::<State1600>();
        debug_assert!(par_width >= 1);
        debug_assert_eq!(
            par_width * mem::size_of::<State1600>(),
            mem::size_of::<ParState1600<B>>(),
            "ParState1600<B> layout is not a packed array of State1600",
        );

        let n_nodes = self.inputs.len();
        if n_nodes == 0 {
            return;
        }
        let mut states: Vec<State1600> = vec![[0u64; 25]; n_nodes];

        let input_len = self.inputs[0].len();
        let n_full_blocks = input_len / KECCAK256_RATE;

        // Absorb full rate blocks.
        for block_idx in 0..n_full_blocks {
            let block_start = block_idx * KECCAK256_RATE;
            for (state, input) in states.iter_mut().zip(self.inputs.iter()) {
                xor_rate_block_into_state(state, &input[block_start..block_start + KECCAK256_RATE]);
            }
            permute_all::<B>(&mut states, par_width, par_f1600, f1600);
        }

        // Final padded block: keccak (not SHA-3) padding `0x01 ... 0x80`.
        // If the data fills the rate exactly, we still emit one extra padding-only block.
        let last_start = n_full_blocks * KECCAK256_RATE;
        let mut padded = [0u8; KECCAK256_RATE];
        for (state, input) in states.iter_mut().zip(self.inputs.iter()) {
            let last = &input[last_start..];
            let last_len = last.len();
            padded[..last_len].copy_from_slice(last);
            padded[last_len] = 0x01;
            padded[KECCAK256_RATE - 1] |= 0x80;
            xor_rate_block_into_state(state, &padded);
            // Reset the bytes we touched for the next iteration.
            padded[..last_len].fill(0);
            padded[last_len] = 0;
            padded[KECCAK256_RATE - 1] = 0;
        }
        permute_all::<B>(&mut states, par_width, par_f1600, f1600);

        // Squeeze: the keccak256 output is the first 32 bytes (4 u64s, LE) of each state.
        for (state, out) in states.iter().zip(self.output.iter_mut()) {
            let mut digest = [0u8; 32];
            for w in 0..4 {
                digest[w * 8..(w + 1) * 8].copy_from_slice(&state[w].to_le_bytes());
            }
            *out = Keccak256Hash(digest);
        }
    }
}

fn xor_rate_block_into_state(state: &mut State1600, block: &[u8]) {
    debug_assert_eq!(block.len(), KECCAK256_RATE);
    for w in 0..KECCAK256_RATE_WORDS {
        let word = u64::from_le_bytes(block[w * 8..(w + 1) * 8].try_into().unwrap());
        state[w] ^= word;
    }
}

/// Apply `f1600` to all states. Uses the parallel function on chunks of `par_width`
/// states and the scalar function on the (shorter than `par_width`) tail.
fn permute_all<B: Backend>(
    states: &mut [State1600],
    par_width: usize,
    par_f1600: keccak::ParFn1600<B>,
    f1600: keccak::Fn1600,
) {
    let mut chunks = states.chunks_exact_mut(par_width);
    for chunk in &mut chunks {
        // SAFETY: `ParState1600<B> = hybrid_array::Array<State1600, B::ParSize1600>` is
        // `#[repr(transparent)]` over `[State1600; par_width]` (where `par_width` matches
        // `B::ParSize1600`'s `USIZE`, asserted above via `size_of`). Casting a
        // `&mut [State1600]` slice of length `par_width` to `&mut ParState1600<B>` therefore
        // preserves layout and aliasing — the slice and the target share the same memory and
        // lifetime, and we only construct one `&mut` at a time.
        let par_state: &mut ParState1600<B> =
            unsafe { &mut *(chunk.as_mut_ptr() as *mut ParState1600<B>) };
        par_f1600(par_state);
    }
    for state in chunks.into_remainder() {
        f1600(state);
    }
}

/// Naive `MerkleOpsLifted` for `SimdBackend`: copies columns to CPU and dispatches to the generic
/// `CpuBackend` lifted impl. Correctness-first; the optimized stage will replace this with a
/// parallel-permutation implementation.
impl MerkleOpsLifted<Keccak256MerkleHasherLifted> for SimdBackend {
    fn build_leaves(
        columns: &[&Col<Self, BaseField>],
        lifting_log_size: u32,
    ) -> Col<Self, Keccak256Hash> {
        let cpu_cols = columns.iter().map(|column| column.to_cpu()).collect_vec();
        <CpuBackend as MerkleOpsLifted<Keccak256MerkleHasherLifted>>::build_leaves(
            &cpu_cols.iter().collect_vec(),
            lifting_log_size,
        )
    }

    fn build_next_layer(prev_layer: &Vec<Keccak256Hash>) -> Vec<Keccak256Hash> {
        <CpuBackend as MerkleOpsLifted<Keccak256MerkleHasherLifted>>::build_next_layer(prev_layer)
    }
}

#[cfg(test)]
mod tests {
    use itertools::Itertools;
    use rand::rngs::SmallRng;
    use rand::{Rng, SeedableRng};

    use crate::core::fields::m31::BaseField;
    use crate::core::vcs::keccak256_hash::Keccak256Hash;
    use crate::core::vcs::keccak256_merkle::Keccak256MerkleHasher;
    use crate::prover::backend::simd::column::BaseColumn;
    use crate::prover::backend::simd::SimdBackend;
    use crate::prover::backend::CpuBackend;
    use crate::prover::vcs::ops::MerkleOps;

    /// Asserts that `CpuBackend::commit_on_layer` and `SimdBackend::commit_on_layer` produce
    /// bit-identical outputs for the given parameters.
    fn assert_simd_eq_cpu(log_size: u32, n_cols: usize, has_prev_layer: bool, seed: u64) {
        let n_nodes = 1usize << log_size;
        let mut rng = SmallRng::seed_from_u64(seed);

        let cpu_cols: Vec<Vec<BaseField>> = (0..n_cols)
            .map(|_| {
                (0..n_nodes)
                    .map(|_| BaseField::from(rng.gen_range(0..(1u32 << 30))))
                    .collect()
            })
            .collect();
        let simd_cols: Vec<BaseColumn> = cpu_cols.iter().map(|c| BaseColumn::from_cpu(c)).collect();

        let cpu_prev: Option<Vec<Keccak256Hash>> = has_prev_layer.then(|| {
            (0..2 * n_nodes)
                .map(|_| Keccak256Hash(core::array::from_fn(|_| rng.gen())))
                .collect()
        });

        let cpu_root = <CpuBackend as MerkleOps<Keccak256MerkleHasher>>::commit_on_layer(
            log_size,
            cpu_prev.as_ref(),
            &cpu_cols.iter().collect_vec(),
        );
        let simd_root = <SimdBackend as MerkleOps<Keccak256MerkleHasher>>::commit_on_layer(
            log_size,
            cpu_prev.as_ref(),
            &simd_cols.iter().collect_vec(),
        );

        assert_eq!(
            cpu_root, simd_root,
            "SIMD/CPU mismatch: log_size={log_size}, n_cols={n_cols}, has_prev_layer={has_prev_layer}",
        );
    }

    /// Sweeps across input-length boundaries (1 rate = 136 bytes, M31 = 4 bytes, children = 64
    /// bytes): below 1 rate, exactly 1 rate (triggers extra padding-only block), above 1 rate,
    /// exactly 2 rates, and above 2 rates. `n_cols` is chosen so `input_len % 136` lands on 0,
    /// near the rate boundary, and well below.
    #[test]
    fn test_simd_matches_cpu_keccak256_commit_on_layer() {
        // Leaf layer (no children).
        assert_simd_eq_cpu(4, 0, false, 1); //   0 bytes (pure padding block)
        assert_simd_eq_cpu(4, 1, false, 2); //   4 bytes
        assert_simd_eq_cpu(5, 7, false, 3); //  28 bytes
        assert_simd_eq_cpu(6, 16, false, 4); //  64 bytes
        assert_simd_eq_cpu(4, 33, false, 5); // 132 bytes (just below rate)
        assert_simd_eq_cpu(5, 34, false, 6); // 136 bytes (== rate, extra padding-only block)
        assert_simd_eq_cpu(4, 35, false, 7); // 140 bytes (> rate, two blocks)
        assert_simd_eq_cpu(4, 64, false, 8); // 256 bytes
        assert_simd_eq_cpu(4, 68, false, 9); // 272 bytes (== 2 * rate)

        // Inner layer (with children: +64 bytes).
        assert_simd_eq_cpu(4, 0, true, 10); //  64 bytes
        assert_simd_eq_cpu(4, 1, true, 11); //  68 bytes
        assert_simd_eq_cpu(5, 16, true, 12); // 128 bytes (just below rate)
        assert_simd_eq_cpu(6, 18, true, 13); // 136 bytes (== rate boundary)
        assert_simd_eq_cpu(4, 33, true, 14); // 196 bytes
        assert_simd_eq_cpu(4, 50, true, 15); // 264 bytes
        assert_simd_eq_cpu(4, 52, true, 16); // 272 bytes (== 2 * rate)
        assert_simd_eq_cpu(4, 64, true, 17); // 320 bytes
    }
}
