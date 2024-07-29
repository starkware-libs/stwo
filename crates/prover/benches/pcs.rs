use core::arch::x86_64::_rdtsc;
use std::iter;

use criterion::{black_box, criterion_group, criterion_main, BatchSize, Criterion};
use stwo_prover::core::backend::simd::SimdBackend;
use stwo_prover::core::backend::{Backend, CpuBackend};
use stwo_prover::core::channel::{Blake2sChannel, Channel};
use stwo_prover::core::fields::m31::BaseField;
use stwo_prover::core::pcs::CommitmentTreeProver;
use stwo_prover::core::poly::circle::{CanonicCoset, CircleEvaluation};
use stwo_prover::core::poly::twiddles::TwiddleTree;
use stwo_prover::core::poly::BitReversedOrder;
use stwo_prover::core::vcs::blake2_hash::Blake2sHash;
use stwo_prover::core::vcs::blake2_merkle::Blake2sMerkleHasher;
use stwo_prover::core::vcs::ops::MerkleOps;

const LOG_COSET_SIZE: u32 = 20;
const LOG_BLOWUP_FACTOR: u32 = 1;
const N_POLYS: usize = 16;

fn measure_cycles<F>(f: F) -> u64
where
    F: FnOnce(),
{
    unsafe {
        let start = _rdtsc();
        f();
        let end = _rdtsc();
        end - start
    }
}

fn benched_fn<B: Backend + MerkleOps<Blake2sMerkleHasher>>(
    evals: Vec<CircleEvaluation<B, BaseField, BitReversedOrder>>,
    channel: &mut Blake2sChannel,
    twiddles: &TwiddleTree<B>,
) {
    let polys = evals
        .into_iter()
        .map(|eval| eval.interpolate_with_twiddles(twiddles))
        .collect();

    CommitmentTreeProver::<B>::new(polys, LOG_BLOWUP_FACTOR, channel, twiddles);
}

fn bench_pcs<B: Backend + MerkleOps<Blake2sMerkleHasher>>(c: &mut Criterion, id: &str) {
    let big_domain = CanonicCoset::new(LOG_COSET_SIZE + LOG_BLOWUP_FACTOR);
    let small_domain = CanonicCoset::new(LOG_COSET_SIZE);

    let twiddles = B::precompute_twiddles(big_domain.half_coset());

    let initial_digest = Blake2sHash::from(vec![0; 32]);
    let mut channel = Blake2sChannel::new(initial_digest);

    let evals: Vec<CircleEvaluation<B, BaseField, BitReversedOrder>> = iter::repeat_with(|| {
        CircleEvaluation::new(
            small_domain.circle_domain(),
            (0..1 << LOG_COSET_SIZE).map(BaseField::from).collect(),
        )
    })
    .take(N_POLYS)
    .collect();

    c.bench_function(
        &format!("{id} polynomial commitment 2^{LOG_COSET_SIZE}"),
        |b| {
            b.iter_batched(
                || evals.clone(),
                |evals| {
                    let cycles = measure_cycles(|| {
                        benched_fn::<B>(
                            black_box(evals),
                            black_box(&mut channel),
                            black_box(&twiddles),
                        )
                    });
                    println!("Clock cycles for this batch: {}", cycles);
                    black_box(cycles)
                },
                BatchSize::LargeInput,
            );
        },
    );
}

fn pcs_benches(c: &mut Criterion) {
    bench_pcs::<SimdBackend>(c, "simd");
    bench_pcs::<CpuBackend>(c, "cpu");
}

criterion_group!(
    name = benches;
    config = Criterion::default().sample_size(10);
    targets = pcs_benches);
criterion_main!(benches);
