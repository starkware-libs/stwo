use std::{env, fs};

use circuit::{gen_fibonacci_trace, SimpleFibonacciComponent, SimpleFibonacciEval};
use num_traits::Zero;
use stwo::core::channel::Blake2sChannel;
use stwo::core::fields::qm31::SecureField;
use stwo::core::pcs::PcsConfig;
use stwo::core::poly::circle::CanonicCoset;
use stwo::core::vcs::blake2_merkle::Blake2sMerkleChannel;
use stwo::prover::backend::simd::SimdBackend;
use stwo::prover::backend::Column;
use stwo::prover::poly::circle::PolyOps;
use stwo::prover::{prove, CommitmentSchemeProver};
use stwo_constraint_framework::TraceLocationAllocator;

fn main() {
    println!("=== STARK Prover ===\n");

    let args: Vec<String> = env::args().collect();
    let log_size = if args.len() > 1 {
        args[1]
            .parse::<u32>()
            .expect("First argument must be log_size (e.g., 8 for 256 rows)")
    } else {
        8
    };

    let initial_a = if args.len() > 2 {
        args[2]
            .parse::<u32>()
            .expect("Second argument must be initial_a")
    } else {
        0
    };

    let initial_b = if args.len() > 3 {
        args[3]
            .parse::<u32>()
            .expect("Third argument must be initial_b")
    } else {
        1
    };

    let n_rows = 1 << log_size;
    println!("Configuration:");
    println!("  log_size: {} ({} rows)", log_size, n_rows);
    println!("  initial values: f(0)={}, f(1)={}", initial_a, initial_b);
    println!();

    // Generate trace
    println!("Generating Fibonacci trace...");
    let trace = gen_fibonacci_trace(log_size, initial_a, initial_b);
    println!("✓ Trace generated with {} rows", n_rows);

    // Get the last computed Fibonacci value
    // Column 2 (index 2) contains f(n) values
    // Last row contains f(n_rows + 1)
    let last_fib_value = trace[2].values.at(n_rows - 1);
    let fib_index = n_rows + 1;
    println!(
        "  Last computed value: f({}) = {}",
        fib_index, last_fib_value
    );
    println!();

    // Setup proving configuration
    let config = PcsConfig::default();
    let twiddles = SimdBackend::precompute_twiddles(
        CanonicCoset::new(log_size + 1 + config.fri_config.log_blowup_factor)
            .circle_domain()
            .half_coset,
    );

    println!("\nGenerating STARK proof...");
    let channel = &mut Blake2sChannel::default();
    let mut commitment_scheme =
        CommitmentSchemeProver::<SimdBackend, Blake2sMerkleChannel>::new(config, &twiddles);

    // Commit preprocessed (empty for this simple circuit)
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
    println!("  Commitments: {}", proof.commitments.len());

    // Save proof to file
    // Note: StarkProof doesn't implement Serialize by default
    // We'll save it in a binary format for now
    println!("\nSaving proof to file...");

    // For now, we'll create a simple metadata file with the parameters
    // The actual proof serialization requires custom implementation
    let metadata = serde_json::json!({
        "log_size": log_size,
        "n_rows": n_rows,
        "initial_a": initial_a,
        "initial_b": initial_b,
        "commitments_count": proof.commitments.len(),
        "note": "Full proof serialization requires custom implementation"
    });

    fs::write(
        "proof_metadata.json",
        serde_json::to_string_pretty(&metadata).unwrap(),
    )
    .expect("Failed to write proof metadata");

    println!("✓ Proof metadata saved to proof_metadata.json");
    println!("\nNote: Full proof serialization is not yet implemented.");
    println!("The verifier will need to regenerate the proof for now.");
    println!("\n✓ Prover completed successfully!");
}
