mod fibonacci;

use fibonacci::{gen_fibonacci_trace, SimpleFibonacciComponent, SimpleFibonacciEval};
use num_traits::Zero;
use stwo::core::air::Component;
use stwo::core::channel::Blake2sChannel;
use stwo::core::fields::m31::M31;
use stwo::core::fields::qm31::SecureField;
use stwo::core::pcs::{CommitmentSchemeVerifier, PcsConfig};
use stwo::core::poly::circle::CanonicCoset;
use stwo::core::vcs::blake2_merkle::Blake2sMerkleChannel;
use stwo::prover::backend::simd::SimdBackend;
use stwo::prover::backend::Column;
use stwo::prover::poly::circle::PolyOps;
use stwo::prover::{prove, CommitmentSchemeProver};
use stwo_constraint_framework::TraceLocationAllocator;

fn main() {
    println!("=== My STARK App ===");
    println!("Using stwo from GitHub!\n");

    // Test M31 field
    let a = M31::from(42);
    let b = M31::from(100);
    let c = a + b;

    println!("M31 field operations:");
    println!("  {} + {} = {}", a, b, c);

    // Test SecureField
    let x = SecureField::from_u32_unchecked(1, 2, 3, 4);
    println!("\nSecureField example:");
    println!("  {:?}", x);

    // Test Fibonacci
    println!("\n=== Fibonacci Sequence ===");
    let log_size = 8; // 256 rows
    let trace = gen_fibonacci_trace(log_size, 1, 1);

    println!("First 10 values of Fibonacci sequence (starting with 1, 1):");
    for i in 0..10 {
        let val = trace[2].values.at(i); // Column 2 is f(n)
        println!("  f({}) = {}", i + 2, val);
    }

    // STARK Proof!
    println!("\n=== STARK Proof ===");
    println!("Generating proof for {} Fibonacci steps...", 1 << log_size);

    let config = PcsConfig::default();

    // Precompute twiddles
    let twiddles = SimdBackend::precompute_twiddles(
        CanonicCoset::new(log_size + 1 + config.fri_config.log_blowup_factor)
            .circle_domain()
            .half_coset,
    );

    // Create channel and commitment scheme
    let channel = &mut Blake2sChannel::default();
    let mut commitment_scheme =
        CommitmentSchemeProver::<SimdBackend, Blake2sMerkleChannel>::new(config, &twiddles);

    // Commit preprocessed (empty)
    let mut tree_builder = commitment_scheme.tree_builder();
    tree_builder.extend_evals(vec![]);
    tree_builder.commit(channel);

    // Commit trace
    let mut tree_builder = commitment_scheme.tree_builder();
    tree_builder.extend_evals(trace.clone());
    tree_builder.commit(channel);

    // Create component
    let component = SimpleFibonacciComponent::new(
        &mut TraceLocationAllocator::default(),
        SimpleFibonacciEval {
            log_n_rows: log_size,
        },
        SecureField::zero(),
    );

    // Generate proof
    let proof = prove(&[&component], channel, commitment_scheme).unwrap();

    println!("✓ Proof generated!");
    println!("  Proof commitments: {}", proof.commitments.len());

    // Verify proof
    println!("\nVerifying proof...");

    let channel = &mut Blake2sChannel::default();
    let commitment_scheme = &mut CommitmentSchemeVerifier::<Blake2sMerkleChannel>::new(config);

    // Preprocessed
    commitment_scheme.commit(
        proof.commitments[0],
        &component.trace_log_degree_bounds()[0],
        channel,
    );

    // Trace
    commitment_scheme.commit(
        proof.commitments[1],
        &component.trace_log_degree_bounds()[1],
        channel,
    );

    stwo::core::verifier::verify(&[&component], channel, commitment_scheme, proof).unwrap();

    println!("✓ Proof verified successfully!");

    println!("\n✓ Everything works!");
}
