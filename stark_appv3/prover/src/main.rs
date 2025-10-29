use std::{env, fs};

use circuit::{gen_fibonacci_trace, SimpleFibonacciComponent, SimpleFibonacciEval};
use num_traits::Zero;
use stwo::core::channel::Blake2sChannel;
use stwo::core::fields::m31::BaseField;
use stwo::core::fields::qm31::SecureField;
use stwo::core::pcs::PcsConfig;
use stwo::core::poly::circle::CanonicCoset;
use stwo::core::vcs::blake2_merkle::Blake2sMerkleChannel;
use stwo::core::ColumnVec;
use stwo::prover::backend::simd::SimdBackend;
use stwo::prover::backend::Column;
use stwo::prover::poly::circle::{CircleEvaluation, PolyOps};
use stwo::prover::poly::BitReversedOrder;
use stwo::prover::{prove, CommitmentSchemeProver};
use stwo_constraint_framework::TraceLocationAllocator;

fn dump_trace_to_file(
    trace: &ColumnVec<CircleEvaluation<SimdBackend, BaseField, BitReversedOrder>>,
    n_rows: usize,
    target_n: usize,
    filename: &str,
) {
    let mut output = String::new();

    output.push_str("=== Fibonacci Trace Dump ===\n\n");
    output.push_str(&format!("Target: f({})\n", target_n));
    output.push_str(&format!("Total rows: {}\n", n_rows));
    output.push_str(&format!("Computed rows: {} (f(0) to f({}))\n", target_n - 1, target_n));
    output.push_str(&format!("Padding rows: {}\n\n", n_rows - (target_n - 1)));

    output.push_str("Structure:\n");
    output.push_str("  Column A: f(n-2)\n");
    output.push_str("  Column B: f(n-1)\n");
    output.push_str("  Column C: f(n)\n\n");

    output.push_str(&format!("{:<8} {:<15} {:<15} {:<15}\n", "Row", "Col A (f(n-2))", "Col B (f(n-1))", "Col C (f(n))"));
    output.push_str(&format!("{}\n", "-".repeat(60)));

    for row in 0..n_rows {
        let col_a = trace[0].values.at(row);
        let col_b = trace[1].values.at(row);
        let col_c = trace[2].values.at(row);

        // Mark padding rows
        let marker = if row >= target_n - 1 { " (padding)" } else { "" };

        output.push_str(&format!(
            "{:<8} {:<15} {:<15} {:<15}{}\n",
            row,
            col_a.0,
            col_b.0,
            col_c.0,
            marker
        ));
    }

    fs::write(filename, output).expect("Failed to write trace dump");
}

fn main() {
    println!("=== STARK Prover v3 ===");
    println!("Compute and prove specific Fibonacci number\n");

    // Parse command line arguments
    let args: Vec<String> = env::args().collect();

    // Check for --dump-trace flag
    let dump_trace = args.iter().any(|arg| arg == "--dump-trace");
    let args_without_flags: Vec<String> = args.iter()
        .filter(|arg| !arg.starts_with("--"))
        .cloned()
        .collect();

    let target_n = if args_without_flags.len() > 1 {
        args_without_flags[1]
            .parse::<usize>()
            .expect("First argument must be target index (e.g., 50 for f(50))")
    } else {
        50 // Default: f(50)
    };

    let initial_a = if args_without_flags.len() > 2 {
        args_without_flags[2]
            .parse::<u32>()
            .expect("Second argument must be initial_a (f(0))")
    } else {
        0 // Standard Fibonacci: f(0) = 0
    };

    let initial_b = if args_without_flags.len() > 3 {
        args_without_flags[3]
            .parse::<u32>()
            .expect("Third argument must be initial_b (f(1))")
    } else {
        1 // Standard Fibonacci: f(1) = 1
    };

    if target_n < 2 {
        println!("Error: target_n must be at least 2");
        return;
    }

    println!("Target: Compute f({})", target_n);
    println!("Initial values: f(0)={}, f(1)={}", initial_a, initial_b);
    println!();

    // Generate trace
    println!("Generating Fibonacci trace...");
    let (trace, target_value, log_size) = gen_fibonacci_trace(target_n, initial_a, initial_b);
    let n_rows = 1 << log_size;
    let computed_rows = target_n - 1;
    let padding_rows = n_rows - computed_rows;

    println!("✓ Trace generated");
    println!("  Total rows: {} (2^{})", n_rows, log_size);
    println!("  Computed rows: {} (f(0) to f({}))", computed_rows, target_n);
    println!("  Padding rows: {} (filled with zeros)", padding_rows);
    println!();
    println!("✓ Target value: f({}) = {} (mod 2^31-1)", target_n, target_value);
    println!();

    // Dump trace if requested
    if dump_trace {
        println!("Dumping trace to file...");
        dump_trace_to_file(&trace, n_rows, target_n, "trace_dump.txt");
        println!("✓ Trace saved to trace_dump.txt");
        println!();
    }

    // Setup proving configuration
    let config = PcsConfig::default();
    let twiddles = SimdBackend::precompute_twiddles(
        CanonicCoset::new(log_size + 1 + config.fri_config.log_blowup_factor)
            .circle_domain()
            .half_coset,
    );

    println!("Generating STARK proof...");
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
    println!();

    // Save proof metadata
    println!("Saving proof metadata...");
    let metadata = serde_json::json!({
        "target_n": target_n,
        "target_value": target_value.0,
        "initial_a": initial_a,
        "initial_b": initial_b,
        "log_size": log_size,
        "n_rows": n_rows,
        "computed_rows": computed_rows,
        "padding_rows": padding_rows,
        "commitments_count": proof.commitments.len(),
        "note": "Full proof serialization requires custom implementation"
    });

    fs::write(
        "proof_metadata.json",
        serde_json::to_string_pretty(&metadata).unwrap(),
    )
    .expect("Failed to write proof metadata");

    println!("✓ Proof metadata saved to proof_metadata.json");
    println!();
    println!("Summary:");
    println!("  Proved: f({}) = {}", target_n, target_value);
    println!("  Trace size: {} rows (with {} padding)", n_rows, padding_rows);
    println!();
    println!("✓ Prover completed successfully!");
}
