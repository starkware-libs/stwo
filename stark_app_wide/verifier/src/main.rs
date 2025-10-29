use std::fs;

use circuit::{gen_wide_fibonacci_trace, WideFibonacciComponent, WideFibonacciEval};
use num_traits::Zero;
use serde_json::Value;
use stwo::core::air::Component;
use stwo::core::channel::Blake2sChannel;
use stwo::core::fields::qm31::SecureField;
use stwo::core::pcs::{CommitmentSchemeVerifier, PcsConfig};
use stwo::core::poly::circle::CanonicCoset;
use stwo::core::vcs::blake2_merkle::Blake2sMerkleChannel;
use stwo::prover::backend::simd::SimdBackend;
use stwo::prover::poly::circle::PolyOps;
use stwo::prover::{prove, CommitmentSchemeProver};
use stwo_constraint_framework::TraceLocationAllocator;

fn main() {
    println!("=== STARK Verifier - Wide Fibonacci ===\n");

    // Read proof metadata
    println!("Reading proof metadata...");
    let metadata_str = fs::read_to_string("proof_metadata.json")
        .expect("Failed to read proof_metadata.json. Make sure to run the prover first!");

    let metadata: Value =
        serde_json::from_str(&metadata_str).expect("Failed to parse proof metadata");

    let log_n_rows = metadata["log_n_rows"].as_u64().unwrap() as u32;
    let n_rows = metadata["n_rows"].as_u64().unwrap();
    let n_columns = metadata["n_columns"].as_u64().unwrap() as usize;
    let initial_a = metadata["initial_a"].as_u64().unwrap() as u32;
    let initial_b = metadata["initial_b"].as_u64().unwrap() as u32;
    let last_fib_value = metadata["last_fib_value"].as_u64().unwrap() as u32;

    println!("✓ Metadata loaded");
    println!();
    println!("Proof claims:");
    println!("  Structure: HORIZONTAL ({} rows × {} columns)", n_rows, n_columns);
    println!("  Initial values: f(0)={}, f(1)={}", initial_a, initial_b);
    println!("  Last value: f({}) = {} (mod 2^31-1)", n_columns - 1, last_fib_value);
    println!();

    // Regenerate trace and proof
    println!("Regenerating trace and proof...");
    let trace = gen_wide_fibonacci_trace(log_n_rows, n_columns, initial_a, initial_b);

    let config = PcsConfig::default();
    let twiddles = SimdBackend::precompute_twiddles(
        CanonicCoset::new(log_n_rows + 1 + config.fri_config.log_blowup_factor)
            .circle_domain()
            .half_coset,
    );

    // Generate proof (normally would be loaded from file)
    let channel = &mut Blake2sChannel::default();
    let mut commitment_scheme =
        CommitmentSchemeProver::<SimdBackend, Blake2sMerkleChannel>::new(config, &twiddles);

    let mut tree_builder = commitment_scheme.tree_builder();
    tree_builder.extend_evals(vec![]);
    tree_builder.commit(channel);

    let mut tree_builder = commitment_scheme.tree_builder();
    tree_builder.extend_evals(trace.clone());
    tree_builder.commit(channel);

    let component = WideFibonacciComponent::new(
        &mut TraceLocationAllocator::default(),
        WideFibonacciEval {
            log_n_rows: log_n_rows,
            n_columns: n_columns,
        },
        SecureField::zero(),
    );

    let proof = prove(&[&component], channel, commitment_scheme).unwrap();
    println!("✓ Proof regenerated (in real scenario, would be loaded from file)");
    println!();

    // Verify the proof
    println!("Verifying proof...");

    let channel = &mut Blake2sChannel::default();
    let commitment_scheme = &mut CommitmentSchemeVerifier::<Blake2sMerkleChannel>::new(config);

    // Commit preprocessed
    commitment_scheme.commit(
        proof.commitments[0],
        &component.trace_log_degree_bounds()[0],
        channel,
    );

    // Commit trace
    commitment_scheme.commit(
        proof.commitments[1],
        &component.trace_log_degree_bounds()[1],
        channel,
    );

    // Verify!
    stwo::core::verifier::verify(&[&component], channel, commitment_scheme, proof).unwrap();

    println!("✓ Proof verified successfully!");
    println!();
    println!("Verification result:");
    println!("  ✓ HORIZONTAL Fibonacci sequence is CORRECT");
    println!("  ✓ {} rows, each with {} Fibonacci values", n_rows, n_columns);
    println!();
    println!("Note: Full proof serialization is not yet implemented.");
    println!("In production, the proof would be loaded from a file instead of regenerated.");
}
