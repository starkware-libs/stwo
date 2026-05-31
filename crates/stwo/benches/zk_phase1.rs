use criterion::{black_box, criterion_group, criterion_main, BatchSize, Criterion};
use rand::rngs::SmallRng;
use rand::{CryptoRng, RngCore, SeedableRng};
use stwo::core::channel::Blake2sChannel;
use stwo::core::fields::m31::{BaseField, P as M31_MODULUS};
use stwo::core::fields::qm31::SecureField;
use stwo::core::fri::{CirclePolyDegreeBound, FriConfig, FriVerifier};
use stwo::core::poly::circle::CanonicCoset;
use stwo::core::queries::Queries;
use stwo::core::vcs_lifted::blake2_merkle::Blake2sMerkleChannel;
use stwo::core::zk::zk_fri_batch_mask_query_positions;
use stwo::prover::backend::{BackendForChannel, Col, CpuBackend};
use stwo::prover::fri::FriProver;
use stwo::prover::poly::circle::{CircleCoefficients, PolyOps, SecureCirclePoly, SecureEvaluation};
use stwo::prover::poly::twiddles::TwiddleTree;
use stwo::prover::poly::BitReversedOrder;
use stwo::prover::zk::{
    add_fri_batch_mask, sample_fri_batch_mask_evaluation, ZkFriBatchMaskOracleProver,
};

const LOG_SIZE: u32 = 16;
const LOG_BLOWUP_FACTOR: u32 = 1;
const LOG_SIZES: [u32; 3] = [12, 16, 20];

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

fn sample_base_field<R: RngCore + ?Sized>(rng: &mut R) -> BaseField {
    loop {
        let candidate = rng.next_u32() & 0x7fff_ffff;
        if candidate < M31_MODULUS {
            return BaseField::from_u32_unchecked(candidate);
        }
    }
}

fn sample_mask_polynomial<B: BackendForChannel<Blake2sMerkleChannel>>(
    log_size: u32,
    seed: u64,
) -> SecureCirclePoly<B> {
    let coefficient_count = 1usize << (log_size - LOG_BLOWUP_FACTOR);
    let mut rng = DeterministicBenchCryptoRng::seed_from_u64(seed);
    SecureCirclePoly(core::array::from_fn(|_| {
        let coeffs: Col<B, BaseField> = (0..coefficient_count)
            .map(|_| sample_base_field(&mut rng))
            .collect();
        CircleCoefficients::<B>::new(coeffs)
    }))
}

fn sample_mask<B: BackendForChannel<Blake2sMerkleChannel>>(
    log_size: u32,
    twiddles: &TwiddleTree<B>,
    seed: u64,
) -> SecureEvaluation<B, BitReversedOrder> {
    let domain = CanonicCoset::new(log_size).circle_domain();
    let mut rng = DeterministicBenchCryptoRng::seed_from_u64(seed);
    sample_fri_batch_mask_evaluation(domain, log_size - LOG_BLOWUP_FACTOR, twiddles, &mut rng)
}

fn query_positions(log_size: u32) -> Vec<usize> {
    let domain_size = 1usize << log_size;
    vec![0, 3, 17, domain_size / 4, domain_size / 2, domain_size - 1]
}

fn bench_zk_r_sampling(c: &mut Criterion) {
    for log_size in LOG_SIZES {
        c.bench_function(
            &format!("zk phase1 r coefficient sampling cpu 2^{log_size}"),
            |b| {
                b.iter_batched(
                    || DeterministicBenchCryptoRng::seed_from_u64(1),
                    |mut rng| {
                        let coefficient_count = 1usize << (log_size - LOG_BLOWUP_FACTOR);
                        let coordinates: [Col<CpuBackend, BaseField>; 4] =
                            core::array::from_fn(|_| {
                                (0..coefficient_count)
                                    .map(|_| sample_base_field(&mut rng))
                                    .collect()
                            });
                        black_box(coordinates);
                    },
                    BatchSize::LargeInput,
                );
            },
        );

        let domain = CanonicCoset::new(log_size).circle_domain();
        let twiddles = CpuBackend::precompute_twiddles(domain.half_coset);

        c.bench_function(&format!("zk phase1 r evaluation cpu 2^{log_size}"), |b| {
            b.iter_batched(
                || sample_mask_polynomial::<CpuBackend>(log_size, 101),
                |poly| {
                    black_box(poly.evaluate_with_twiddles(domain, black_box(&twiddles)));
                },
                BatchSize::LargeInput,
            );
        });

        c.bench_function(
            &format!("zk phase1 r sampling and evaluation cpu 2^{log_size}"),
            |b| {
                b.iter_batched(
                    || DeterministicBenchCryptoRng::seed_from_u64(1),
                    |mut rng| {
                        black_box(sample_fri_batch_mask_evaluation::<CpuBackend, _>(
                            domain,
                            log_size - LOG_BLOWUP_FACTOR,
                            black_box(&twiddles),
                            black_box(&mut rng),
                        ));
                    },
                    BatchSize::LargeInput,
                );
            },
        );
    }
}

fn bench_zk_r_commitment(c: &mut Criterion) {
    for log_size in LOG_SIZES {
        let domain = CanonicCoset::new(log_size).circle_domain();
        let twiddles = CpuBackend::precompute_twiddles(domain.half_coset);
        let mask = sample_mask::<CpuBackend>(log_size, &twiddles, 2);

        c.bench_function(&format!("zk phase1 r commitment cpu 2^{log_size}"), |b| {
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
}

fn bench_zk_h_batch_addition(c: &mut Criterion) {
    for log_size in LOG_SIZES {
        let domain = CanonicCoset::new(log_size).circle_domain();
        let twiddles = CpuBackend::precompute_twiddles(domain.half_coset);
        let raw_quotient = sample_mask::<CpuBackend>(log_size, &twiddles, 3);
        let mask = sample_mask::<CpuBackend>(log_size, &twiddles, 4);

        c.bench_function(
            &format!("zk phase1 h_batch addition cpu 2^{log_size}"),
            |b| {
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
            },
        );
    }
}

fn bench_zk_r_opening(c: &mut Criterion) {
    for log_size in LOG_SIZES {
        let domain = CanonicCoset::new(log_size).circle_domain();
        let twiddles = CpuBackend::precompute_twiddles(domain.half_coset);
        let mask = sample_mask::<CpuBackend>(log_size, &twiddles, 5);
        let queries = query_positions(log_size);
        let fri_config = FriConfig::new(0, LOG_BLOWUP_FACTOR, queries.len(), 1);

        c.bench_function(
            &format!("zk phase1 r opening construction cpu 2^{log_size}"),
            |b| {
                b.iter_batched(
                    || {
                        ZkFriBatchMaskOracleProver::<
                            CpuBackend,
                            <Blake2sMerkleChannel as stwo::core::channel::MerkleChannel>::H,
                        >::new(mask.clone())
                    },
                    |oracle| {
                        let mut channel = Blake2sChannel::default();
                        let fri_queries = zk_fri_batch_mask_query_positions(
                            &mut channel,
                            log_size,
                            fri_config.n_queries,
                            &queries,
                            fri_config,
                        )
                        .unwrap();
                        let fri_prover = FriProver::<CpuBackend, Blake2sMerkleChannel>::commit(
                            &mut channel,
                            fri_config,
                            oracle.evaluation(),
                            &twiddles,
                        );
                        let fri_proof =
                            fri_prover.decommit_on_queries(&Queries::new(&fri_queries, log_size));
                        black_box(oracle.decommit(
                            black_box(queries.as_slice()),
                            black_box(fri_queries.as_slice()),
                            fri_proof.proof,
                        ));
                    },
                    BatchSize::LargeInput,
                );
            },
        );

        c.bench_function(
            &format!("zk phase1 r independent query sampling cpu 2^{log_size}"),
            |b| {
                b.iter_batched(
                    Blake2sChannel::default,
                    |mut channel| {
                        black_box(
                            zk_fri_batch_mask_query_positions(
                                black_box(&mut channel),
                                log_size,
                                fri_config.n_queries,
                                black_box(&queries),
                                fri_config,
                            )
                            .unwrap(),
                        );
                    },
                    BatchSize::SmallInput,
                );
            },
        );
    }
}

fn bench_zk_r_verification(c: &mut Criterion) {
    for log_size in LOG_SIZES {
        let domain = CanonicCoset::new(log_size).circle_domain();
        let twiddles = CpuBackend::precompute_twiddles(domain.half_coset);
        let mask = sample_mask::<CpuBackend>(log_size, &twiddles, 6);
        let queries = query_positions(log_size);
        let fri_config = FriConfig::new(0, LOG_BLOWUP_FACTOR, queries.len(), 1);
        let mut query_channel = Blake2sChannel::default();
        let fri_queries = zk_fri_batch_mask_query_positions(
            &mut query_channel,
            log_size,
            fri_config.n_queries,
            &queries,
            fri_config,
        )
        .unwrap();

        c.bench_function(
            &format!("zk phase1 r opening verification cpu 2^{log_size}"),
            |b| {
                b.iter_batched(
                    || {
                        let oracle = ZkFriBatchMaskOracleProver::<
                            CpuBackend,
                            <Blake2sMerkleChannel as stwo::core::channel::MerkleChannel>::H,
                        >::new(mask.clone());
                        let mut channel = Blake2sChannel::default();
                        let fri_prover = FriProver::<CpuBackend, Blake2sMerkleChannel>::commit(
                            &mut channel,
                            fri_config,
                            oracle.evaluation(),
                            &twiddles,
                        );
                        let fri_proof =
                            fri_prover.decommit_on_queries(&Queries::new(&fri_queries, log_size));
                        oracle.decommit(&queries, &fri_queries, fri_proof.proof).0
                    },
                    |proof| {
                        let fri_proof = proof.fri_proof.clone();
                        let fri_values = proof.fri_queried_values.to_secure_values();
                        let mut channel = Blake2sChannel::default();
                        let fri_verifier = FriVerifier::<Blake2sMerkleChannel>::commit(
                            &mut channel,
                            fri_config,
                            fri_proof,
                            CirclePolyDegreeBound::new(log_size - LOG_BLOWUP_FACTOR),
                        )
                        .unwrap();
                        fri_verifier
                            .decommit_on_query_positions(&fri_queries, fri_values)
                            .unwrap();
                        black_box(
                            proof
                                .verify_openings(black_box(queries.as_slice()), log_size)
                                .unwrap(),
                        );
                    },
                    BatchSize::LargeInput,
                );
            },
        );
    }
}

fn bench_zk_answer_addition(c: &mut Criterion) {
    let queries = query_positions(LOG_SIZE);
    let mut answers = vec![SecureField::from_u32_unchecked(1, 2, 3, 4); queries.len()];
    let masks = vec![SecureField::from_u32_unchecked(5, 6, 7, 8); queries.len()];

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
        bench_zk_r_sampling,
        bench_zk_r_commitment,
        bench_zk_h_batch_addition,
        bench_zk_r_opening,
        bench_zk_r_verification,
        bench_zk_answer_addition
);
criterion_main!(benches);
