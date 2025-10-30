use poseidon_circuit::{
    gen_interaction_trace, gen_trace, PoseidonComponent, PoseidonElements, PoseidonEval,
    N_INSTANCES_PER_ROW, N_LOG_INSTANCES_PER_ROW, N_STATE, N_COLUMNS,
};
use std::{env, fs};
use stwo::core::channel::Blake2sChannel;
use stwo::core::fields::m31::BaseField;
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
use tracing::{info, span, Level};

const LOG_EXPAND: u32 = 2;

fn dump_trace_to_file(
    trace: &ColumnVec<CircleEvaluation<SimdBackend, BaseField, BitReversedOrder>>,
    n_rows: usize,
    filename: &str,
) {
    let mut output = String::new();

    output.push_str("=== Poseidon2 Hash Trace Dump ===\n\n");
    output.push_str(&format!("Hash function: Poseidon2 (16-element state)\n"));
    output.push_str(&format!("Instances per row: {}\n", N_INSTANCES_PER_ROW));
    output.push_str(&format!("Total columns: {}\n", N_COLUMNS));
    output.push_str(&format!("Total rows: {}\n\n", n_rows));

    output.push_str("Structure per instance (158 columns):\n");
    output.push_str("  Cols 0-15:    Initial state (16 elements)\n");
    output.push_str("  Cols 16-31:   After Full Round 1\n");
    output.push_str("  Cols 32-47:   After Full Round 2\n");
    output.push_str("  Cols 48-63:   After Full Round 3\n");
    output.push_str("  Cols 64-79:   After Full Round 4\n");
    output.push_str("  Cols 80-93:   After Partial Rounds 1-14 (first element only)\n");
    output.push_str("  Cols 94-109:  After Full Round 5\n");
    output.push_str("  Cols 110-125: After Full Round 6\n");
    output.push_str("  Cols 126-141: After Full Round 7\n");
    output.push_str("  Cols 142-157: After Full Round 8 (Final state)\n\n");

    // Show only first instance for readability (columns 0-93)
    let max_rows_to_show = n_rows.min(10);

    output.push_str("=== INSTANCE 0 (First Hash) ===\n\n");

    // Initial state
    output.push_str("Initial State (cols 0-15):\n");
    output.push_str(&format!("{:<6}", "Row"));
    for i in 0..N_STATE {
        output.push_str(&format!("s{:<2}        ", i));
    }
    output.push_str("\n");
    output.push_str(&format!("{}\n", "-".repeat(150)));

    for row in 0..max_rows_to_show {
        output.push_str(&format!("{:<6}", row));
        for col in 0..N_STATE {
            let val = trace[col].values.at(row);
            output.push_str(&format!("{:<11}", val.0));
        }
        output.push_str("\n");
    }

    // After Full Rounds
    for full_round in 0..4 {
        let start_col = 16 + full_round * 16;
        output.push_str(&format!("\nAfter Full Round {} (cols {}-{}):\n",
            full_round + 1, start_col, start_col + 15));
        output.push_str(&format!("{:<6}", "Row"));
        for i in 0..N_STATE {
            output.push_str(&format!("s{:<2}        ", i));
        }
        output.push_str("\n");
        output.push_str(&format!("{}\n", "-".repeat(150)));

        for row in 0..max_rows_to_show {
            output.push_str(&format!("{:<6}", row));
            for col in start_col..(start_col + N_STATE) {
                let val = trace[col].values.at(row);
                output.push_str(&format!("{:<11}", val.0));
            }
            output.push_str("\n");
        }
    }

    // Partial rounds (only first element)
    output.push_str("\nPartial Rounds 1-14 (cols 80-93, first element only):\n");
    output.push_str(&format!("{:<6}", "Row"));
    for i in 1..=14 {
        output.push_str(&format!("PR{:<2}       ", i));
    }
    output.push_str("\n");
    output.push_str(&format!("{}\n", "-".repeat(150)));

    for row in 0..max_rows_to_show {
        output.push_str(&format!("{:<6}", row));
        for col in 80..94 {
            let val = trace[col].values.at(row);
            output.push_str(&format!("{:<11}", val.0));
        }
        output.push_str("\n");
    }

    // Last 4 full rounds (after partial rounds)
    for full_round in 0..4 {
        let start_col = 94 + full_round * 16;
        output.push_str(&format!("\nAfter Full Round {} (cols {}-{}):\n",
            5 + full_round, start_col, start_col + 15));
        output.push_str(&format!("{:<6}", "Row"));
        for i in 0..N_STATE {
            output.push_str(&format!("s{:<2}        ", i));
        }
        output.push_str("\n");
        output.push_str(&format!("{}\n", "-".repeat(150)));

        for row in 0..max_rows_to_show {
            output.push_str(&format!("{:<6}", row));
            for col in start_col..(start_col + N_STATE) {
                let val = trace[col].values.at(row);
                output.push_str(&format!("{:<11}", val.0));
            }
            output.push_str("\n");
        }
    }

    if n_rows > max_rows_to_show {
        output.push_str(&format!("\n... ({} more rows not shown)\n", n_rows - max_rows_to_show));
    }

    output.push_str(&format!("\n\n=== Summary for all {} instances ===\n", N_INSTANCES_PER_ROW));
    output.push_str("(Each instance follows the same structure, offset by 158 columns)\n\n");

    // Show first and last values for each instance
    output.push_str(&format!("{:<10} {:<30} {:<30}\n", "Instance", "Initial[0]", "Final[0]"));
    output.push_str(&format!("{}\n", "-".repeat(80)));
    for inst in 0..N_INSTANCES_PER_ROW {
        let initial_col = inst * 158;
        let final_col = inst * 158 + 142; // After Full Round 8 (first element)
        let row = 0;

        let initial_val = trace[initial_col].values.at(row).0;
        let final_val = trace[final_col].values.at(row).0;

        output.push_str(&format!("{:<10} {:<30} {:<30}\n", inst, initial_val, final_val));
    }

    fs::write(filename, output).expect("Failed to write trace dump");
}

fn main() {
    println!("=== STARK Prover - Poseidon2 Hash Function ===\n");

    let args: Vec<String> = env::args().collect();

    // Parse arguments
    let mut log_n_instances = 10; // Default: 1024 instances = 2^10
    let mut dump_trace = false;

    for arg in args.iter().skip(1) {
        if arg == "--dump-trace" {
            dump_trace = true;
        } else if let Ok(val) = arg.parse::<u32>() {
            log_n_instances = val;
        }
    }

    assert!(
        log_n_instances >= N_LOG_INSTANCES_PER_ROW as u32,
        "log_n_instances must be >= {} (got {})",
        N_LOG_INSTANCES_PER_ROW,
        log_n_instances
    );

    let log_n_rows = log_n_instances - N_LOG_INSTANCES_PER_ROW as u32;
    let n_rows = 1 << log_n_rows;
    let n_instances = 1 << log_n_instances;

    println!("Configuration:");
    println!("  Log instances: {}", log_n_instances);
    println!("  Total instances: {}", n_instances);
    println!("  Instances per row: {}", N_INSTANCES_PER_ROW);
    println!("  Log rows: {}", log_n_rows);
    println!("  Rows: {}", n_rows);
    if dump_trace {
        println!("  Trace dump: ENABLED");
    }
    println!();

    let config = PcsConfig::default();

    // Precompute twiddles
    let span = span!(Level::INFO, "Precompute twiddles").entered();
    let twiddles = SimdBackend::precompute_twiddles(
        CanonicCoset::new(log_n_rows + LOG_EXPAND + config.fri_config.log_blowup_factor)
            .circle_domain()
            .half_coset,
    );
    span.exit();

    let channel = &mut Blake2sChannel::default();
    let mut commitment_scheme =
        CommitmentSchemeProver::<SimdBackend, Blake2sMerkleChannel>::new(config, &twiddles);

    // Step 1: Preprocessed (empty)
    println!("Step 1: Committing preprocessed (empty)...");
    let span = span!(Level::INFO, "Constant").entered();
    let mut tree_builder = commitment_scheme.tree_builder();
    tree_builder.extend_evals(vec![]);
    tree_builder.commit(channel);
    span.exit();

    // Step 2: Generate trace
    println!("Step 2: Generating Poseidon trace...");
    let span = span!(Level::INFO, "Trace").entered();
    let (trace, lookup_data) = gen_trace(log_n_rows);
    println!("  ✓ Trace generated: {} rows", n_rows);

    // Dump trace if requested
    if dump_trace {
        println!("  Dumping trace to file...");
        dump_trace_to_file(&trace, n_rows, "trace_dump.txt");
        println!("  ✓ Trace saved to trace_dump.txt");
    }

    let mut tree_builder = commitment_scheme.tree_builder();
    tree_builder.extend_evals(trace);
    tree_builder.commit(channel);
    span.exit();

    // Step 3: Draw lookup elements
    println!("Step 3: Drawing lookup elements...");
    let lookup_elements = PoseidonElements::draw(channel);

    // Step 4: Generate interaction trace
    println!("Step 4: Generating LogUp interaction trace...");
    let span = span!(Level::INFO, "Interaction").entered();
    let (trace, claimed_sum) = gen_interaction_trace(log_n_rows, lookup_data, &lookup_elements);
    println!("  Claimed sum: {:?}", claimed_sum);
    let mut tree_builder = commitment_scheme.tree_builder();
    tree_builder.extend_evals(trace);
    tree_builder.commit(channel);
    span.exit();

    // Step 5: Create component and prove
    println!("Step 5: Creating Poseidon component...");
    let component = PoseidonComponent::new(
        &mut TraceLocationAllocator::default(),
        PoseidonEval {
            log_n_rows,
            lookup_elements: lookup_elements.clone(),
            claimed_sum,
        },
        claimed_sum,
    );
    info!("Poseidon component info:\n{}", component);

    println!("Step 6: Generating STARK proof...");
    let proof = prove(&[&component], channel, commitment_scheme).unwrap();
    println!("✓ Proof generated!");
    println!("  Commitments: {}", proof.commitments.len());
    println!("  Proof size: {} bytes", proof.size_estimate());
    println!();

    // Save proof to JSON
    println!("Saving proof to file...");
    let proof_json = serde_json::to_string_pretty(&proof).expect("Failed to serialize proof");
    fs::write("proof.json", proof_json).expect("Failed to write proof.json");
    println!("✓ Proof saved to proof.json");
    println!("  File size: {} bytes", fs::metadata("proof.json").unwrap().len());
    println!();

    // Save metadata
    println!("Saving proof metadata...");
    let metadata = serde_json::json!({
        "log_n_instances": log_n_instances,
        "n_instances": n_instances,
        "log_n_rows": log_n_rows,
        "n_rows": n_rows,
        "instances_per_row": N_INSTANCES_PER_ROW,
        "claimed_sum": {
            "a": [claimed_sum.0.0.0, claimed_sum.0.1.0],
            "b": [claimed_sum.1.0.0, claimed_sum.1.1.0]
        },
        "commitments_count": proof.commitments.len(),
        "proof_size_bytes": proof.size_estimate(),
        "hash_function": "Poseidon2"
    });

    fs::write(
        "proof_metadata.json",
        serde_json::to_string_pretty(&metadata).unwrap(),
    )
    .expect("Failed to write metadata");

    println!("✓ Proof metadata saved to proof_metadata.json");
    println!();
    println!("Summary:");
    println!("  Hash function: Poseidon2 (16-element state)");
    println!("  Instances proven: {}", n_instances);
    println!("  Rows in trace: {}", n_rows);
    println!("  Proof size: {} bytes", proof.size_estimate());
    println!();
    println!("✓ Prover completed successfully!");
}
