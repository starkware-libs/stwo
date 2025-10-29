use std::fs;

use circuit::{gen_fibonacci_trace, SimpleFibonacciComponent, SimpleFibonacciEval};
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
    println!("=== STARK Verifier v3 ===\n");

    // Read proof metadata
    println!("Reading proof metadata...");
    let metadata_str = fs::read_to_string("proof_metadata.json")
        .expect("Failed to read proof_metadata.json. Make sure to run the prover first!");

    let metadata: Value =
        serde_json::from_str(&metadata_str).expect("Failed to parse proof metadata");

    let target_n = metadata["target_n"].as_u64().unwrap() as usize;
    let target_value = metadata["target_value"].as_u64().unwrap() as u32;
    let initial_a = metadata["initial_a"].as_u64().unwrap() as u32;
    let initial_b = metadata["initial_b"].as_u64().unwrap() as u32;
    let log_size = metadata["log_size"].as_u64().unwrap() as u32;
    let n_rows = metadata["n_rows"].as_u64().unwrap();
    let computed_rows = metadata["computed_rows"].as_u64().unwrap();
    let padding_rows = metadata["padding_rows"].as_u64().unwrap();

    println!("✓ Metadata loaded");
    println!();
    println!("Proof claims:");
    println!("  f({}) = {} (mod 2^31-1)", target_n, target_value);
    println!("  Initial values: f(0)={}, f(1)={}", initial_a, initial_b);
    println!("  Trace size: {} rows ({} computed, {} padding)", n_rows, computed_rows, padding_rows);
    println!();

    // Note: Since we can't serialize/deserialize the full proof yet,
    // we'll regenerate it here to demonstrate verification
    println!("Regenerating trace and proof...");
    let (trace, regenerated_value, _) = gen_fibonacci_trace(target_n, initial_a, initial_b);

    // Verify the computed value matches metadata
    if regenerated_value.0 != target_value {
        println!("❌ Error: Regenerated value doesn't match metadata!");
        println!("   Expected: {}", target_value);
        println!("   Got: {}", regenerated_value.0);
        return;
    }

    let config = PcsConfig::default();
    let twiddles = SimdBackend::precompute_twiddles(
        CanonicCoset::new(log_size + 1 + config.fri_config.log_blowup_factor)
            .circle_domain()
            .half_coset,
    );

    // Generate proof (normally this would be loaded from file)
    let channel = &mut Blake2sChannel::default();
    let mut commitment_scheme =
        CommitmentSchemeProver::<SimdBackend, Blake2sMerkleChannel>::new(config, &twiddles);

    let mut tree_builder = commitment_scheme.tree_builder();
    tree_builder.extend_evals(vec![]);
    tree_builder.commit(channel);

    let mut tree_builder = commitment_scheme.tree_builder();
    tree_builder.extend_evals(trace.clone());
    tree_builder.commit(channel);

    let component = SimpleFibonacciComponent::new(
        &mut TraceLocationAllocator::default(),
        SimpleFibonacciEval {
            log_n_rows: log_size,
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
    println!("  ✓ f({}) = {} is CORRECT", target_n, target_value);
    println!();
    println!("Note: Full proof serialization is not yet implemented.");
    println!("In production, the proof would be loaded from a file instead of regenerated.");
}
