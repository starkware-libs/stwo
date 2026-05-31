use criterion::{black_box, criterion_group, criterion_main, BatchSize, Criterion};
use rand::rngs::SmallRng;
use rand::{CryptoRng, RngCore, SeedableRng};
use stwo::core::fields::qm31::SecureField;
use stwo::core::poly::circle::CanonicCoset;
use stwo::core::vcs_lifted::blake2_merkle::Blake2sMerkleChannel;
use stwo::prover::backend::{BackendForChannel, CpuBackend};
use stwo::prover::poly::circle::{PolyOps, SecureEvaluation};
use stwo::prover::poly::twiddles::TwiddleTree;
use stwo::prover::poly::BitReversedOrder;
use stwo::prover::zk::{
    add_fri_batch_mask, sample_fri_batch_mask_evaluation, ZkFriBatchMaskOracleProver,
};

const LOG_SIZE: u32 = 16;
const LOG_BLOWUP_FACTOR: u32 = 1;
const LOG_DEGREE_BOUND: u32 = LOG_SIZE - LOG_BLOWUP_FACTOR;
const QUERY_POSITIONS: [usize; 6] = [0, 3, 17, 1024, 32768, (1 << LOG_SIZE) - 1];

#[derive(Clone)]
struct DeterministicBenchCryptoRng(SmallRng);

impl DeterministicBenchCryptoRng {
    fn seed_from_u64(seed: u64) -> Self {
        Self(SmallRng::seed_from_u64(seed))
    }
}

impl RngCore for DeterministicBenchCryptoRng {
    fn next_u32(&mut self) -> u32 {
        self.0.next_u32()
    }

    fn next_u64(&mut self) -> u64 {
        self.0.next_u64()
    }

    fn fill_bytes(&mut self, dest: &mut [u8]) {
        self.0.fill_bytes(dest);
    }

    fn try_fill_bytes(&mut self, dest: &mut [u8]) -> Result<(), rand::Error> {
        self.0.try_fill_bytes(dest)
    }
}

impl CryptoRng for DeterministicBenchCryptoRng {}

fn sample_mask<B: BackendForChannel<Blake2sMerkleChannel>>(
    twiddles: &TwiddleTree<B>,
    seed: u64,
) -> SecureEvaluation<B, BitReversedOrder> {
    let domain = CanonicCoset::new(LOG_SIZE).circle_domain();
    let mut rng = DeterministicBenchCryptoRng::seed_from_u64(seed);
    sample_fri_batch_mask_evaluation(domain, LOG_DEGREE_BOUND, twiddles, &mut rng)
}

fn bench_zk_phase1_r_sampling(c: &mut Criterion) {
    let domain = CanonicCoset::new(LOG_SIZE).circle_domain();
    let twiddles = CpuBackend::precompute_twiddles(domain.half_coset);

    c.bench_function("zk phase1 r sampling and evaluation cpu 2^16", |b| {
        b.iter_batched(
            || DeterministicBenchCryptoRng::seed_from_u64(1),
            |mut rng| {
                black_box(sample_fri_batch_mask_evaluation::<CpuBackend, _>(
                    domain,
                    LOG_DEGREE_BOUND,
                    black_box(&twiddles),
                    black_box(&mut rng),
                ));
            },
            BatchSize::LargeInput,
        );
    });
}

fn bench_zk_phase1_r_commitment(c: &mut Criterion) {
    let domain = CanonicCoset::new(LOG_SIZE).circle_domain();
    let twiddles = CpuBackend::precompute_twiddles(domain.half_coset);
    let mask = sample_mask::<CpuBackend>(&twiddles, 2);

    c.bench_function("zk phase1 r commitment cpu 2^16", |b| {
        b.iter_batched(
            || mask.clone(),
            |mask| {
                black_box(ZkFriBatchMaskOracleProver::<
                    CpuBackend,
                    <Blake2sMerkleChannel as stwo::core::channel::MerkleChannel>::H,
                >::new(black_box(mask)));
            },
            BatchSize::LargeInput,
        );
    });
}

fn bench_zk_phase1_h_batch_addition(c: &mut Criterion) {
    let domain = CanonicCoset::new(LOG_SIZE).circle_domain();
    let twiddles = CpuBackend::precompute_twiddles(domain.half_coset);
    let raw_quotient = sample_mask::<CpuBackend>(&twiddles, 3);
    let mask = sample_mask::<CpuBackend>(&twiddles, 4);

    c.bench_function("zk phase1 h_batch addition cpu 2^16", |b| {
        b.iter_batched(
            || (raw_quotient.clone(), mask.clone()),
            |(raw_quotient, mask)| {
                black_box(add_fri_batch_mask(
                    black_box(raw_quotient),
                    black_box(&mask),
                ));
            },
            BatchSize::LargeInput,
        );
    });
}

fn bench_zk_phase1_r_opening(c: &mut Criterion) {
    let domain = CanonicCoset::new(LOG_SIZE).circle_domain();
    let twiddles = CpuBackend::precompute_twiddles(domain.half_coset);
    let mask = sample_mask::<CpuBackend>(&twiddles, 5);

    c.bench_function("zk phase1 r opening construction cpu 2^16", |b| {
        b.iter_batched(
            || {
                ZkFriBatchMaskOracleProver::<
                    CpuBackend,
                    <Blake2sMerkleChannel as stwo::core::channel::MerkleChannel>::H,
                >::new(mask.clone())
            },
            |oracle| {
                black_box(oracle.decommit(black_box(&QUERY_POSITIONS)));
            },
            BatchSize::LargeInput,
        );
    });
}

fn bench_zk_phase1_r_verification(c: &mut Criterion) {
    let domain = CanonicCoset::new(LOG_SIZE).circle_domain();
    let twiddles = CpuBackend::precompute_twiddles(domain.half_coset);
    let mask = sample_mask::<CpuBackend>(&twiddles, 6);

    c.bench_function("zk phase1 r opening verification cpu 2^16", |b| {
        b.iter_batched(
            || {
                let oracle = ZkFriBatchMaskOracleProver::<
                    CpuBackend,
                    <Blake2sMerkleChannel as stwo::core::channel::MerkleChannel>::H,
                >::new(mask.clone());
                oracle.decommit(&QUERY_POSITIONS).0
            },
            |proof| {
                black_box(
                    proof
                        .verify_openings(black_box(&QUERY_POSITIONS), LOG_SIZE)
                        .unwrap(),
                );
            },
            BatchSize::LargeInput,
        );
    });
}

fn bench_zk_phase1_answer_addition(c: &mut Criterion) {
    let mut answers = vec![SecureField::from_u32_unchecked(1, 2, 3, 4); QUERY_POSITIONS.len()];
    let masks = vec![SecureField::from_u32_unchecked(5, 6, 7, 8); QUERY_POSITIONS.len()];

    c.bench_function("zk phase1 r answer addition 6 queries", |b| {
        b.iter_batched(
            || (answers.clone(), masks.clone()),
            |(mut answers, masks)| {
                for (answer, mask) in answers.iter_mut().zip(masks) {
                    *answer += mask;
                }
                black_box(answers);
            },
            BatchSize::SmallInput,
        );
    });

    answers.clear();
}

criterion_group!(
    name = benches;
    config = Criterion::default().sample_size(10);
    targets =
        bench_zk_phase1_r_sampling,
        bench_zk_phase1_r_commitment,
        bench_zk_phase1_h_batch_addition,
        bench_zk_phase1_r_opening,
        bench_zk_phase1_r_verification,
        bench_zk_phase1_answer_addition
);
criterion_main!(benches);
