use std::iter;

use criterion::{black_box, criterion_group, criterion_main, BatchSize, Criterion};
use rand::rngs::SmallRng;
use rand::{Rng, SeedableRng};
use stwo_prover::core::backend::simd::SimdBackend;
use stwo_prover::core::backend::{BackendForChannel, CpuBackend};
use stwo_prover::core::channel::Blake2sChannel;
use stwo_prover::core::fields::m31::BaseField;
use stwo_prover::core::pcs::CommitmentTreeProver;
use stwo_prover::core::poly::circle::{CanonicCoset, CircleEvaluation};
use stwo_prover::core::poly::twiddles::TwiddleTree;
use stwo_prover::core::poly::BitReversedOrder;
use stwo_prover::core::vcs::blake2_merkle::Blake2sMerkleChannel;

const LOG_COSET_SIZE: u32 = 20;
const LOG_BLOWUP_FACTOR: u32 = 1;
const N_POLYS: usize = 16;

fn benched_fn<B: BackendForChannel<Blake2sMerkleChannel>>(
    evals: Vec<CircleEvaluation<B, BaseField, BitReversedOrder>>,
    channel: &mut Blake2sChannel,
    twiddles: &TwiddleTree<B>,
) {
    let polys = evals
        .into_iter()
        .map(|eval| eval.interpolate_with_twiddles(twiddles))
        .collect();

    CommitmentTreeProver::<B, Blake2sMerkleChannel>::new(
        polys,
        LOG_BLOWUP_FACTOR,
        channel,
        twiddles,
    );
}

fn bench_pcs<B: BackendForChannel<Blake2sMerkleChannel>>(c: &mut Criterion, id: &str) {
    let small_domain = CanonicCoset::new(LOG_COSET_SIZE);
    let big_domain = CanonicCoset::new(LOG_COSET_SIZE + LOG_BLOWUP_FACTOR);
    let twiddles = B::precompute_twiddles(big_domain.half_coset());
    let mut channel = Blake2sChannel::default();
    let mut rng = SmallRng::seed_from_u64(0);

    let evals: Vec<CircleEvaluation<B, BaseField, BitReversedOrder>> = iter::repeat_with(|| {
        CircleEvaluation::new(
            small_domain.circle_domain(),
            (0..1 << LOG_COSET_SIZE).map(|_| rng.gen()).collect(),
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
                    benched_fn::<B>(
                        black_box(evals),
                        black_box(&mut channel),
                        black_box(&twiddles),
                    )
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
