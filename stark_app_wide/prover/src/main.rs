use std::{env, fs};

use circuit::{gen_wide_fibonacci_trace, WideFibonacciComponent, WideFibonacciEval};
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
    n_columns: usize,
    filename: &str,
) {
    let mut output = String::new();

    output.push_str("=== Wide Fibonacci Trace Dump ===\n\n");
    output.push_str(&format!("Structure: {} columns (HORIZONTAL)\n", n_columns));
    output.push_str(&format!("Each row contains: f(0) to f({})\n", n_columns - 1));
    output.push_str(&format!("Total rows: {}\n\n", n_rows));

    // Header
    output.push_str(&format!("{:<8}", "Row"));
    for i in 0..n_columns.min(10) {
        output.push_str(&format!("{:<12}", format!("f({})", i)));
    }
    if n_columns > 10 {
        output.push_str("  ...");
    }
    output.push_str("\n");
    output.push_str(&format!("{}\n", "-".repeat(100)));

    // Data rows
    for row in 0..n_rows {
        output.push_str(&format!("{:<8}", row));
        for col in 0..n_columns.min(10) {
            let val = trace[col].values.at(row);
            output.push_str(&format!("{:<12}", val.0));
        }
        if n_columns > 10 {
            output.push_str(&format!("  ...  {}", trace[n_columns - 1].values.at(row).0));
        }
        output.push_str("\n");
    }

    fs::write(filename, output).expect("Failed to write trace dump");
}

fn main() {
    println!("=== STARK Prover - Wide Fibonacci ===\n");

    // Parse command line arguments
    let args: Vec<String> = env::args().collect();

    let dump_trace = args.iter().any(|arg| arg == "--dump-trace");
    let args_without_flags: Vec<String> = args
        .iter()
        .filter(|arg| !arg.starts_with("--"))
        .cloned()
        .collect();

    let log_n_rows = if args_without_flags.len() > 1 {
        args_without_flags[1]
            .parse::<u32>()
            .expect("First argument must be log_n_rows (e.g., 3 for 8 rows)")
    } else {
        3 // Default: 8 rows
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

    let n_columns = if args_without_flags.len() > 4 {
        args_without_flags[4]
            .parse::<usize>()
            .expect("Fourth argument must be n_columns (number of Fibonacci values per row)")
    } else {
        50 // Default: 50 columns
    };

    let n_rows = 1 << log_n_rows;

    println!("Configuration:");
    println!("  Rows: {} (2^{})", n_rows, log_n_rows);
    println!("  Columns: {} (each row = complete Fibonacci sequence)", n_columns);
    println!("  Initial values: f(0)={}, f(1)={}", initial_a, initial_b);
    println!();

    // Generate trace
    println!("Generating wide Fibonacci trace...");
    let trace = gen_wide_fibonacci_trace(log_n_rows, n_columns, initial_a, initial_b);
    println!("✓ Trace generated");
    println!("  {} rows × {} columns = {} total values", n_rows, n_columns, n_rows * n_columns);

    // Show last Fibonacci value
    let last_fib_value = trace[n_columns - 1].values.at(0);
    println!("  Last value: f({}) = {} (mod 2^31-1)", n_columns - 1, last_fib_value);
    println!();

    // Dump trace if requested
    if dump_trace {
        println!("Dumping trace to file...");
        dump_trace_to_file(&trace, n_rows, n_columns, "trace_dump.txt");
        println!("✓ Trace saved to trace_dump.txt");
        println!();
    }

    // Setup proving configuration
    let config = PcsConfig::default();
    let twiddles = SimdBackend::precompute_twiddles(
        CanonicCoset::new(log_n_rows + 1 + config.fri_config.log_blowup_factor)
            .circle_domain()
            .half_coset,
    );

    println!("Generating STARK proof...");
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
    let component = WideFibonacciComponent::new(
        &mut TraceLocationAllocator::default(),
        WideFibonacciEval {
            log_n_rows: log_n_rows,
            n_columns: n_columns,
        },
        SecureField::zero(),
    );

    // Generate proof
    let proof = prove(&[&component], channel, commitment_scheme).unwrap();
    println!("✓ Proof generated!");
    println!("  Commitments: {}", proof.commitments.len());
    println!("  Proof size estimate: {} bytes", proof.size_estimate());
    println!();

    // Save proof to JSON
    println!("Saving proof to file...");
    let proof_json = serde_json::to_string_pretty(&proof).expect("Failed to serialize proof");
    fs::write("proof.json", proof_json).expect("Failed to write proof.json");
    println!("✓ Proof saved to proof.json");
    println!("  File size: {} bytes", fs::metadata("proof.json").unwrap().len());
    println!();

    // Save proof metadata
    println!("Saving proof metadata...");
    let metadata = serde_json::json!({
        "log_n_rows": log_n_rows,
        "n_rows": n_rows,
        "n_columns": n_columns,
        "initial_a": initial_a,
        "initial_b": initial_b,
        "last_fib_value": last_fib_value.0,
        "commitments_count": proof.commitments.len(),
        "proof_size_bytes": proof.size_estimate(),
        "structure": "horizontal"
    });

    fs::write(
        "proof_metadata.json",
        serde_json::to_string_pretty(&metadata).unwrap(),
    )
    .expect("Failed to write proof metadata");

    println!("✓ Proof metadata saved to proof_metadata.json");
    println!();
    println!("Summary:");
    println!("  Structure: HORIZONTAL ({} rows × {} columns)", n_rows, n_columns);
    println!("  Each row = complete Fibonacci f(0) to f({})", n_columns - 1);
    println!("  Last value: f({}) = {}", n_columns - 1, last_fib_value);
    println!();
    println!("✓ Prover completed successfully!");
}
