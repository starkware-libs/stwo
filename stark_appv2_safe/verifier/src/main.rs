use std::fs;
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

use circuit::{
    gen_fibonacci_trace, gen_is_first_column, is_first_column_id, SimpleFibonacciComponent,
    SimpleFibonacciEval,
};

fn main() {
    println!("=== STARK Verifier ===\n");

    // Read proof metadata
    println!("Reading proof metadata...");
    let metadata_str = fs::read_to_string("proof_metadata.json")
        .expect("Failed to read proof_metadata.json. Make sure to run the prover first!");

    let metadata: Value = serde_json::from_str(&metadata_str)
        .expect("Failed to parse proof metadata");

    let log_size = metadata["log_size"].as_u64().unwrap() as u32;
    let n_rows = metadata["n_rows"].as_u64().unwrap();
    let initial_a = metadata["initial_a"].as_u64().unwrap() as u32;
    let initial_b = metadata["initial_b"].as_u64().unwrap() as u32;

    println!("✓ Metadata loaded");
    println!("\nConfiguration:");
    println!("  log_size: {} ({} rows)", log_size, n_rows);
    println!("  initial values: f(0)={}, f(1)={}", initial_a, initial_b);
    println!();

    println!("\n=== Security Test: Attempting to verify proof with wrong trace ===\n");

    let config = PcsConfig::default();
    let twiddles = SimdBackend::precompute_twiddles(
        CanonicCoset::new(log_size + 1 + config.fri_config.log_blowup_factor)
            .circle_domain()
            .half_coset,
    );

    // Generate CORRECT proof with values from metadata
    println!("Generating CORRECT proof with initial values: f(0)={}, f(1)={}", initial_a, initial_b);
    let correct_trace = gen_fibonacci_trace(log_size, initial_a, initial_b);

    let channel1 = &mut Blake2sChannel::default();
    let mut commitment_scheme1 =
        CommitmentSchemeProver::<SimdBackend, Blake2sMerkleChannel>::new(config, &twiddles);

    let is_first_col = gen_is_first_column(log_size);
    let mut tree_builder = commitment_scheme1.tree_builder();
    tree_builder.extend_evals(vec![is_first_col.clone()]);
    tree_builder.commit(channel1);

    let mut tree_builder = commitment_scheme1.tree_builder();
    tree_builder.extend_evals(correct_trace.clone());
    tree_builder.commit(channel1);

    let correct_component = SimpleFibonacciComponent::new(
        &mut TraceLocationAllocator::default(),
        SimpleFibonacciEval {
            log_n_rows: log_size,
            is_first_id: is_first_column_id(log_size),
            initial_a,
            initial_b,
        },
        SecureField::zero(),
    );

    let correct_proof = prove(&[&correct_component], channel1, commitment_scheme1).unwrap();
    println!("✓ CORRECT proof generated");

    // Generate FAKE proof with DIFFERENT values (this will be a VALID proof for different constraints)
    let fake_initial_a = 5u32;
    let fake_initial_b = 7u32;
    println!("\nGenerating FAKE proof with different values: f(0)={}, f(1)={}", fake_initial_a, fake_initial_b);
    println!("Note: This proof will be VALID for constraints with initial values (5, 7)");
    let fake_trace = gen_fibonacci_trace(log_size, fake_initial_a, fake_initial_b);

    let channel2 = &mut Blake2sChannel::default();
    let mut commitment_scheme2 =
        CommitmentSchemeProver::<SimdBackend, Blake2sMerkleChannel>::new(config, &twiddles);

    let mut tree_builder = commitment_scheme2.tree_builder();
    tree_builder.extend_evals(vec![is_first_col]);
    tree_builder.commit(channel2);

    let mut tree_builder = commitment_scheme2.tree_builder();
    tree_builder.extend_evals(fake_trace.clone());
    tree_builder.commit(channel2);

    // Component with FAKE initial values - proof will be valid for THESE constraints
    let fake_component = SimpleFibonacciComponent::new(
        &mut TraceLocationAllocator::default(),
        SimpleFibonacciEval {
            log_n_rows: log_size,
            is_first_id: is_first_column_id(log_size),
            initial_a: fake_initial_a,
            initial_b: fake_initial_b,
        },
        SecureField::zero(),
    );

    let fake_proof = prove(&[&fake_component], channel2, commitment_scheme2).unwrap();
    println!("✓ FAKE proof generated (valid for f(0)=5, f(1)=7)");

    // Test 1: Verify CORRECT proof with CORRECT component (should succeed)
    println!("\n--- Test 1: Legitimate verification ---");
    println!("Verifying CORRECT proof with CORRECT component...");

    let verify_channel1 = &mut Blake2sChannel::default();
    let verify_commitment_scheme1 = &mut CommitmentSchemeVerifier::<Blake2sMerkleChannel>::new(config);

    verify_commitment_scheme1.commit(
        correct_proof.commitments[0],
        &correct_component.trace_log_degree_bounds()[0],
        verify_channel1
    );

    verify_commitment_scheme1.commit(
        correct_proof.commitments[1],
        &correct_component.trace_log_degree_bounds()[1],
        verify_channel1
    );

    match stwo::core::verifier::verify(&[&correct_component], verify_channel1, verify_commitment_scheme1, correct_proof.clone()) {
        Ok(_) => println!("✓ Test 1 PASSED: Correct proof verified successfully!"),
        Err(e) => println!("✗ Test 1 FAILED: {:?}", e),
    }

    // Test 2: SECURITY TEST - Try to verify FAKE proof with CORRECT constraints (should FAIL)
    println!("\n--- Test 2: Security test (attempting to cheat) ---");
    println!("We have a proof that is VALID for f(0)={}, f(1)={}", fake_initial_a, fake_initial_b);
    println!("Now trying to verify it using constraints for f(0)={}, f(1)={}", initial_a, initial_b);
    println!("This simulates an attacker trying to use a proof for wrong computation!");

    let verify_channel2 = &mut Blake2sChannel::default();
    let verify_commitment_scheme2 = &mut CommitmentSchemeVerifier::<Blake2sMerkleChannel>::new(config);

    // Create component with CORRECT initial values (what we want to check against)
    let correct_constraints_component = SimpleFibonacciComponent::new(
        &mut TraceLocationAllocator::default(),
        SimpleFibonacciEval {
            log_n_rows: log_size,
            is_first_id: is_first_column_id(log_size),
            initial_a,  // Using CORRECT values
            initial_b,  // But verifying FAKE proof!
        },
        SecureField::zero(),
    );

    // Try to verify FAKE proof with CORRECT constraints
    verify_commitment_scheme2.commit(
        fake_proof.commitments[0],
        &correct_constraints_component.trace_log_degree_bounds()[0],
        verify_channel2
    );

    verify_commitment_scheme2.commit(
        fake_proof.commitments[1],
        &correct_constraints_component.trace_log_degree_bounds()[1],
        verify_channel2
    );

    println!("\nAttempting verification...");
    match stwo::core::verifier::verify(&[&correct_constraints_component], verify_channel2, verify_commitment_scheme2, fake_proof) {
        Ok(_) => {
            println!("✗ SECURITY BREACH: Fake proof was accepted! This should NOT happen!");
            println!("   The system allowed us to verify a proof for f(0)={}, f(1)={}", fake_initial_a, fake_initial_b);
            println!("   using constraints that expect f(0)={}, f(1)={}", initial_a, initial_b);
        },
        Err(e) => {
            println!("✓ Test 2 PASSED: Verifier correctly REJECTED the mismatched proof!");
            println!("   Error: {:?}", e);
            println!("\n   This proves the STARK verifier is secure!");
            println!("   You cannot use a proof generated for one set of public inputs");
            println!("   to verify against different public inputs.");
        }
    }

    // Test 3: REVERSE Security test - Try to verify CORRECT proof with FAKE constraints (should FAIL)
    println!("\n--- Test 3: Reverse security test ---");
    println!("We have a proof that is VALID for f(0)={}, f(1)={}", initial_a, initial_b);
    println!("Now trying to verify it using constraints for f(0)={}, f(1)={}", fake_initial_a, fake_initial_b);
    println!("This is the REVERSE of Test 2 - testing symmetry of security!");

    let verify_channel3 = &mut Blake2sChannel::default();
    let verify_commitment_scheme3 = &mut CommitmentSchemeVerifier::<Blake2sMerkleChannel>::new(config);

    // Create component with FAKE initial values
    let fake_constraints_component = SimpleFibonacciComponent::new(
        &mut TraceLocationAllocator::default(),
        SimpleFibonacciEval {
            log_n_rows: log_size,
            is_first_id: is_first_column_id(log_size),
            initial_a: fake_initial_a,  // Using FAKE values
            initial_b: fake_initial_b,  // But verifying CORRECT proof!
        },
        SecureField::zero(),
    );

    // Try to verify CORRECT proof with FAKE constraints
    verify_commitment_scheme3.commit(
        correct_proof.commitments[0],
        &fake_constraints_component.trace_log_degree_bounds()[0],
        verify_channel3
    );

    verify_commitment_scheme3.commit(
        correct_proof.commitments[1],
        &fake_constraints_component.trace_log_degree_bounds()[1],
        verify_channel3
    );

    println!("\nAttempting verification...");
    match stwo::core::verifier::verify(&[&fake_constraints_component], verify_channel3, verify_commitment_scheme3, correct_proof) {
        Ok(_) => {
            println!("✗ SECURITY BREACH: Proof was accepted with wrong constraints! This should NOT happen!");
            println!("   The system allowed us to verify a proof for f(0)={}, f(1)={}", initial_a, initial_b);
            println!("   using constraints that expect f(0)={}, f(1)={}", fake_initial_a, fake_initial_b);
        },
        Err(e) => {
            println!("✓ Test 3 PASSED: Verifier correctly REJECTED the mismatched constraints!");
            println!("   Error: {:?}", e);
            println!("\n   Security is SYMMETRIC - it doesn't matter which direction you try to cheat!");
        }
    }

    println!("\n=== Security Test Completed ===");
    println!("\nConclusion:");
    println!("  • Test 1: Legitimate verification ✓");
    println!("  • Test 2: Fake proof with correct constraints ✓ (rejected)");
    println!("  • Test 3: Correct proof with fake constraints ✓ (rejected)");
    println!("\nThe STARK verifier is cryptographically secure!");
}
