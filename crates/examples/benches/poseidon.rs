use std::fs;
use std::time::{Duration, Instant};

use criterion::{criterion_group, criterion_main, Criterion, Throughput};
use rand::rngs::StdRng;
use rand::SeedableRng;
use stwo::core::air::Component;
use stwo::core::channel::Blake2sChannel;
use stwo::core::fri::FriConfig;
use stwo::core::pcs::{CommitmentSchemeVerifier, PcsConfig};
use stwo::core::poly::circle::CanonicCoset;
use stwo::core::proof::StarkProof;
use stwo::core::vcs_lifted::blake2_merkle::{Blake2sMerkleChannel, Blake2sMerkleHasher};
use stwo::core::verifier::{verify, verify_zk_with_witness_randomization_audit};
use stwo::core::zk::ZkStarkProof;
use stwo::prover::backend::simd::SimdBackend;
use stwo::prover::poly::circle::PolyOps;
use stwo::prover::zk::{ZkDerivationGate, ZkDerivationReview};
use stwo::prover::{prove_zk, CommitmentSchemeProver};
use stwo_constraint_framework::TraceLocationAllocator;
use stwo_examples::poseidon::zk::{
    mix_poseidon_zk_prover_metadata_before_lookup, mix_poseidon_zk_verifier_metadata_before_lookup,
    poseidon_zk_configs, poseidon_zk_log_n_rows, poseidon_zk_metadata_component,
    poseidon_zk_pcs_config, PoseidonZkMetadataError,
};
use stwo_examples::poseidon::{
    gen_interaction_trace, gen_trace, prove_poseidon, PoseidonComponent, PoseidonElements,
    PoseidonEval,
};

struct PublicPoseidonReport {
    prove_time: Duration,
    verify_time: Duration,
    proof_size_estimate: usize,
    sampled_value_count: usize,
    queried_value_count: usize,
}

struct ZkPoseidonReport {
    witness_randomization_time: Duration,
    interaction_randomization_time: Duration,
    quotient_composition_and_fri_time: Duration,
    verify_time: Duration,
    proof_size_estimate: usize,
    sampled_value_count: usize,
    queried_value_count: usize,
    original_private_range_count: usize,
    interaction_private_range_count: usize,
    trace_domain_log_size: u32,
    randomized_witness_log_degree: u32,
    fri_first_layer_log_size: u32,
    quotient_log_degree_bound: u32,
    trace_tree_scope_hash: [u8; 32],
}

fn report_hash(seed: u8) -> [u8; 32] {
    [seed; 32]
}

fn derivation_reviews() -> Vec<ZkDerivationReview> {
    [
        ZkDerivationGate::StwoSplitQueryExpansion,
        ZkDerivationGate::CircleRandomizerSpace,
        ZkDerivationGate::OodsDomainExclusion,
        ZkDerivationGate::ZkAwareDegreeMetadata,
        ZkDerivationGate::FriBatchMaskDegree,
        ZkDerivationGate::PrivateLookupPermutationExclusion,
        ZkDerivationGate::ProofDataSecrecy,
        ZkDerivationGate::ZkPerformanceControls,
    ]
    .into_iter()
    .enumerate()
    .map(|(index, gate)| ZkDerivationReview {
        gate,
        review_hash: report_hash(index as u8 + 110),
    })
    .collect()
}

fn verify_public_poseidon(component: &PoseidonComponent, proof: StarkProof<Blake2sMerkleHasher>) {
    let channel = &mut Blake2sChannel::default();
    let commitment_scheme =
        &mut CommitmentSchemeVerifier::<Blake2sMerkleChannel>::new(proof.config);
    let sizes = component.trace_log_degree_bounds();

    commitment_scheme.commit(proof.commitments[0], &sizes[0], channel);
    commitment_scheme.commit(proof.commitments[1], &sizes[1], channel);
    let lookup_elements = PoseidonElements::draw(channel);
    assert_eq!(lookup_elements, component.lookup_elements);
    commitment_scheme.commit(proof.commitments[2], &sizes[2], channel);

    verify(&[component], channel, commitment_scheme, proof).unwrap();
}

fn public_poseidon_report(log_n_instances: u32) -> PublicPoseidonReport {
    let config = PcsConfig::default();
    let prove_start = Instant::now();
    let (component, proof) = prove_poseidon(log_n_instances, config);
    let prove_time = prove_start.elapsed();
    let proof_size_estimate = proof.size_estimate();
    let sampled_value_count = tree_value_count(&proof.0.sampled_values.0);
    let queried_value_count = tree_value_count(&proof.0.queried_values.0);

    let verify_start = Instant::now();
    verify_public_poseidon(&component, proof);
    let verify_time = verify_start.elapsed();

    PublicPoseidonReport {
        prove_time,
        verify_time,
        proof_size_estimate,
        sampled_value_count,
        queried_value_count,
    }
}

fn value_count<T>(values: &[Vec<T>]) -> usize {
    values.iter().map(Vec::len).sum()
}

fn tree_value_count<T>(values: &[Vec<Vec<T>>]) -> usize {
    values.iter().map(|tree| value_count(tree)).sum()
}

fn verify_zk_poseidon(
    config: PcsConfig,
    component: &PoseidonComponent,
    proof: ZkStarkProof<Blake2sMerkleHasher>,
    zk_verifier_config: &stwo::core::zk::ZkVerificationConfig,
    zk_verifier_audit: &stwo::core::zk::ZkWitnessRandomizationVerifierAudit,
) {
    let metadata_component = poseidon_zk_metadata_component(component.log_n_rows);
    let (_, _, _, canonical_metadata) = poseidon_zk_configs(
        &metadata_component,
        config.fri_config.log_blowup_factor,
        derivation_reviews(),
    )
    .unwrap();
    let channel = &mut Blake2sChannel::default();
    let commitment_scheme = &mut CommitmentSchemeVerifier::<Blake2sMerkleChannel>::new(config);

    commitment_scheme.commit(
        proof.0.randomized_pcs_proof.commitments[0],
        &canonical_metadata.component_column_log_sizes[0],
        channel,
    );
    commitment_scheme.commit(
        proof.0.randomized_pcs_proof.commitments[1],
        &vec![
            canonical_metadata.randomized_witness_log_degree;
            canonical_metadata.component_column_log_sizes[1].len()
        ],
        channel,
    );
    mix_poseidon_zk_verifier_metadata_before_lookup(channel, zk_verifier_config);
    let lookup_elements = PoseidonElements::draw(channel);
    assert_eq!(lookup_elements, component.lookup_elements);
    commitment_scheme.commit(
        proof.0.randomized_pcs_proof.commitments[2],
        &vec![
            canonical_metadata.randomized_witness_log_degree;
            canonical_metadata.component_column_log_sizes[2].len()
        ],
        channel,
    );

    verify_zk_with_witness_randomization_audit(
        &[component],
        channel,
        commitment_scheme,
        proof,
        zk_verifier_config,
        zk_verifier_audit,
    )
    .unwrap();
}

fn zk_poseidon_report(log_n_instances: u32) -> Result<ZkPoseidonReport, PoseidonZkMetadataError> {
    let log_n_rows = poseidon_zk_log_n_rows(log_n_instances).unwrap();
    let config = poseidon_zk_pcs_config(log_n_rows, FriConfig::new(1, 1, 3, 1))?;
    let twiddles = SimdBackend::precompute_twiddles(
        CanonicCoset::new(config.lifting_log_size.unwrap()).half_coset(),
    );
    let metadata_component = poseidon_zk_metadata_component(log_n_rows);
    let (zk_prover_config, zk_verifier_config, zk_verifier_audit, canonical_metadata) =
        poseidon_zk_configs(
            &metadata_component,
            config.fri_config.log_blowup_factor,
            derivation_reviews(),
        )?;
    let prover_channel = &mut Blake2sChannel::default();
    let mut commitment_scheme =
        CommitmentSchemeProver::<SimdBackend, Blake2sMerkleChannel>::new(config, &twiddles);
    commitment_scheme.set_store_polynomials_coefficients();

    let mut tree_builder = commitment_scheme.tree_builder();
    tree_builder.extend_evals(vec![]);
    tree_builder.commit(prover_channel);

    let (trace, lookup_data) = gen_trace(log_n_rows);
    let mut witness_rng = StdRng::seed_from_u64(1);
    let mut tree_builder = commitment_scheme.tree_builder();
    tree_builder.extend_evals(trace);
    let witness_randomization_start = Instant::now();
    tree_builder
        .commit_zk_witness_randomized(&zk_prover_config, &mut witness_rng, prover_channel)
        .unwrap();
    let witness_randomization_time = witness_randomization_start.elapsed();

    mix_poseidon_zk_prover_metadata_before_lookup(prover_channel, &zk_prover_config);
    let lookup_elements = PoseidonElements::draw(prover_channel);
    let (trace, claimed_sum) = gen_interaction_trace(log_n_rows, lookup_data, &lookup_elements);
    let mut interaction_rng = StdRng::seed_from_u64(3);
    let mut tree_builder = commitment_scheme.tree_builder();
    tree_builder.extend_evals(trace);
    let interaction_randomization_start = Instant::now();
    tree_builder
        .commit_zk_witness_randomized(&zk_prover_config, &mut interaction_rng, prover_channel)
        .unwrap();
    let interaction_randomization_time = interaction_randomization_start.elapsed();

    let component = PoseidonComponent::new(
        &mut TraceLocationAllocator::default(),
        PoseidonEval {
            log_n_rows,
            lookup_elements,
            claimed_sum,
        },
        claimed_sum,
    );
    assert_eq!(
        &component.trace_log_degree_bounds().0,
        &metadata_component.trace_log_degree_bounds().0
    );
    assert_eq!(
        canonical_metadata.private_ranges.len(),
        canonical_metadata.original_trace_private_ranges.len()
            + canonical_metadata.interaction_trace_private_ranges.len()
    );

    let mut proof_rng = StdRng::seed_from_u64(2);
    let prove_zk_start = Instant::now();
    let proof = prove_zk::<SimdBackend, Blake2sMerkleChannel, _>(
        &[&component],
        prover_channel,
        commitment_scheme,
        &zk_prover_config,
        &mut proof_rng,
    )
    .unwrap();
    let quotient_composition_and_fri_time = prove_zk_start.elapsed();
    let proof_size_estimate = proof.size_estimate();
    let sampled_value_count = tree_value_count(&proof.0.randomized_pcs_proof.sampled_values.0);
    let queried_value_count = tree_value_count(&proof.0.randomized_pcs_proof.queried_values.0);

    let verify_start = Instant::now();
    verify_zk_poseidon(
        config,
        &component,
        proof,
        &zk_verifier_config,
        &zk_verifier_audit,
    );
    let verify_time = verify_start.elapsed();

    Ok(ZkPoseidonReport {
        witness_randomization_time,
        interaction_randomization_time,
        quotient_composition_and_fri_time,
        verify_time,
        proof_size_estimate,
        sampled_value_count,
        queried_value_count,
        original_private_range_count: canonical_metadata.original_trace_private_ranges.len(),
        interaction_private_range_count: canonical_metadata.interaction_trace_private_ranges.len(),
        trace_domain_log_size: canonical_metadata.trace_domain_log_size,
        randomized_witness_log_degree: canonical_metadata.randomized_witness_log_degree,
        fri_first_layer_log_size: canonical_metadata.fri_first_layer_log_size,
        quotient_log_degree_bound: canonical_metadata.quotient_degree_bound.log_degree_bound,
        trace_tree_scope_hash: canonical_metadata.trace_tree_scope_hash,
    })
}

fn maybe_write_poseidon_zk_report() {
    let Some(path) = std::env::var_os("STWO_POSEIDON_ZK_BENCH_REPORT") else {
        return;
    };
    const LOG_N_INSTANCES: u32 = 10;
    let public = public_poseidon_report(LOG_N_INSTANCES);
    let Ok(zk) = zk_poseidon_report(LOG_N_INSTANCES) else {
        let report = format!(
            "# Poseidon public vs ZK benchmark report\n\n\
             `log_n_instances`: {LOG_N_INSTANCES}\n\n\
             ZK mode: blocked. Private LogUp interaction columns currently expose \
             witness-derived `claimed_sum` scalars unless a reviewed private-claim protocol \
             replaces public scalar claims. The ZK metadata builder fails closed before proof \
             construction.\n\n\
             | bucket | value |\n\
             |---|---:|\n\
             | public Poseidon prove | {:?} |\n\
             | public verify | {:?} |\n\
             | public proof size estimate bytes | {} |\n\
             | public sampled value count | {} |\n\
             | public queried/opened value count | {} |\n",
            public.prove_time,
            public.verify_time,
            public.proof_size_estimate,
            public.sampled_value_count,
            public.queried_value_count,
        );
        fs::write(path, report).unwrap();
        return;
    };
    let proof_size_delta = zk.proof_size_estimate as i128 - public.proof_size_estimate as i128;
    let sampled_value_delta = zk.sampled_value_count as i128 - public.sampled_value_count as i128;
    let queried_value_delta = zk.queried_value_count as i128 - public.queried_value_count as i128;
    let report = format!(
        "# Poseidon public vs ZK benchmark report\n\n\
         `log_n_instances`: {LOG_N_INSTANCES}\n\n\
         ZK mode: original trace and LogUp interaction running-sum columns private with a reviewed \
         private LogUp scalar-claim policy.\n\n\
         Metadata builder: `stwo::core::zk::build_stwo_zk_air_metadata`; Poseidon supplies \
         AIR-specific tree policy and reviewed degree expansion.\n\n\
         Public metadata leakage: private column positions/counts, degree bounds, tree scopes, \
         and stable metadata hashes are public circuit metadata and must not encode secrets.\n\n\
         | bucket | value |\n\
         |---|---:|\n\
         | public Poseidon prove | {:?} |\n\
         | ZK original witness randomization | {:?} |\n\
         | ZK LogUp interaction randomization | {:?} |\n\
         | ZK quotient/composition masking + ZK FRI/proof generation | {:?} |\n\
         | public verify | {:?} |\n\
         | ZK verify | {:?} |\n\
         | ZK trace domain log size | {} |\n\
         | ZK randomized witness log degree | {} |\n\
         | ZK FRI first layer log size | {} |\n\
         | ZK quotient log degree bound | {} |\n\
         | ZK trace tree scope hash | {:?} |\n\
         | ZK original private range count | {} |\n\
         | ZK LogUp interaction private range count | {} |\n\
         | public proof size estimate bytes | {} |\n\
         | ZK proof size estimate bytes | {} |\n\
         | proof size delta bytes | {} |\n\
         | public sampled value count | {} |\n\
         | ZK sampled value count | {} |\n\
         | sampled value count delta | {} |\n\
         | public queried/opened value count | {} |\n\
         | ZK queried/opened value count | {} |\n\
         | queried/opened value count delta | {} |\n",
        public.prove_time,
        zk.witness_randomization_time,
        zk.interaction_randomization_time,
        zk.quotient_composition_and_fri_time,
        public.verify_time,
        zk.verify_time,
        zk.trace_domain_log_size,
        zk.randomized_witness_log_degree,
        zk.fri_first_layer_log_size,
        zk.quotient_log_degree_bound,
        zk.trace_tree_scope_hash,
        zk.original_private_range_count,
        zk.interaction_private_range_count,
        public.proof_size_estimate,
        zk.proof_size_estimate,
        proof_size_delta,
        public.sampled_value_count,
        zk.sampled_value_count,
        sampled_value_delta,
        public.queried_value_count,
        zk.queried_value_count,
        queried_value_delta,
    );
    fs::write(path, report).unwrap();
}

pub fn simd_poseidon(c: &mut Criterion) {
    maybe_write_poseidon_zk_report();

    if std::env::var_os("STWO_RUN_POSEIDON_PROOF_BENCH").is_none() {
        eprintln!(
            "skipping Poseidon proof bench by default; set \
             STWO_RUN_POSEIDON_PROOF_BENCH=1 to run it explicitly"
        );
        return;
    }

    const LOG_N_INSTANCES: u32 = 18;
    let mut group = c.benchmark_group("poseidon2");
    group.throughput(Throughput::Elements(1u64 << LOG_N_INSTANCES));
    group.bench_function(format!("poseidon2 2^{LOG_N_INSTANCES} instances"), |b| {
        b.iter(|| prove_poseidon(LOG_N_INSTANCES, PcsConfig::default()));
    });
}

criterion_group!(
    name = bit_rev;
    config = Criterion::default().sample_size(10);
    targets = simd_poseidon);
criterion_main!(bit_rev);
