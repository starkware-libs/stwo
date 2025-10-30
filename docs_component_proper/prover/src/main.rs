use dual_component_circuit::{
    gen_computing_logup_trace, gen_computing_trace, gen_scheduling_logup_trace,
    gen_scheduling_trace, Components, ComponentsStatement0, ComponentsStatement1,
    ComputationLookupElements,
};
use num_traits::Zero;
use std::{env, fs};
use stwo::core::channel::Blake2sChannel;
use stwo::core::pcs::PcsConfig;
use stwo::core::poly::circle::CanonicCoset;
use stwo::prover::backend::simd::m31::LOG_N_LANES;
use stwo::prover::backend::simd::SimdBackend;
use stwo::prover::poly::circle::PolyOps;
use stwo::prover::{prove, CommitmentSchemeProver};
use stwo::core::vcs::blake2_merkle::Blake2sMerkleChannel;

const CONSTRAINT_EVAL_BLOWUP_FACTOR: u32 = 1;

fn main() {
    println!("=== STARK Prover - Dual Component (Computing + Scheduling) ===\n");

    let args: Vec<String> = env::args().collect();

    let log_size = if args.len() > 1 {
        args[1]
            .parse::<u32>()
            .expect("Argument must be log_size (e.g., 5 for 32 rows)")
    } else {
        LOG_N_LANES // Default
    };

    let n_rows = 1 << log_size;

    println!("Configuration:");
    println!("  Log size: {}", log_size);
    println!("  Rows: {}", n_rows);
    println!();

    let config = PcsConfig::default();
    let twiddles = SimdBackend::precompute_twiddles(
        CanonicCoset::new(
            log_size + CONSTRAINT_EVAL_BLOWUP_FACTOR + config.fri_config.log_blowup_factor,
        )
        .circle_domain()
        .half_coset,
    );

    let channel = &mut Blake2sChannel::default();
    let mut commitment_scheme =
        CommitmentSchemeProver::<SimdBackend, Blake2sMerkleChannel>::new(config, &twiddles);

    println!("Step 1: Committing preprocessed (empty)...");
    let mut tree_builder = commitment_scheme.tree_builder();
    tree_builder.extend_evals(vec![]);
    tree_builder.commit(channel);

    println!("Step 2: Computing generates inputs and computes x^5+1...");
    let (computing_trace, computation_lookup_data) = gen_computing_trace(log_size);
    println!("  ✓ Computing generated {} rows", n_rows);

    println!("Step 3: Scheduling copies data from Computing...");
    let scheduling_trace = gen_scheduling_trace(log_size, &computation_lookup_data);
    println!("  ✓ Scheduling copied data");
    println!();

    let statement0 = ComponentsStatement0 { log_size };
    statement0.mix_into(channel);

    println!("Step 4: Committing trace columns...");
    let mut tree_builder = commitment_scheme.tree_builder();
    tree_builder.extend_evals([scheduling_trace.clone(), computing_trace.clone()].concat());
    tree_builder.commit(channel);

    println!("Step 5: Drawing lookup elements...");
    let lookup_elements = ComputationLookupElements::draw(channel);

    println!("Step 6: Generating LogUp interaction columns...");
    let (scheduling_logup_cols, scheduling_claimed_sum) = gen_scheduling_logup_trace(
        log_size,
        &scheduling_trace[0],
        &scheduling_trace[1],
        &lookup_elements,
    );
    let (computing_logup_cols, computing_claimed_sum) = gen_computing_logup_trace(
        log_size,
        &computing_trace[0],
        &computing_trace[2],
        &lookup_elements,
    );

    println!("  Scheduling claimed sum: {:?}", scheduling_claimed_sum);
    println!("  Computing claimed sum: {:?}", computing_claimed_sum);
    println!(
        "  Sum (should be zero): {:?}",
        scheduling_claimed_sum + computing_claimed_sum
    );
    assert_eq!(
        scheduling_claimed_sum + computing_claimed_sum,
        stwo::core::fields::qm31::SecureField::zero(),
        "LogUp sums must cancel!"
    );
    println!();

    let statement1 = ComponentsStatement1 {
        scheduling_claimed_sum,
        computing_claimed_sum,
    };
    statement1.mix_into(channel);

    println!("Step 7: Committing LogUp columns...");
    let mut tree_builder = commitment_scheme.tree_builder();
    tree_builder.extend_evals([scheduling_logup_cols, computing_logup_cols].concat());
    tree_builder.commit(channel);

    println!("Step 8: Creating components...");
    let components = Components::new(&statement0, &lookup_elements, &statement1);

    println!("Step 9: Generating STARK proof...");
    let proof = prove(&components.component_provers(), channel, commitment_scheme).unwrap();
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
        "log_size": log_size,
        "n_rows": n_rows,
        "scheduling_claimed_sum": {
            "a": [scheduling_claimed_sum.0.0.0, scheduling_claimed_sum.0.1.0],
            "b": [scheduling_claimed_sum.1.0.0, scheduling_claimed_sum.1.1.0]
        },
        "computing_claimed_sum": {
            "a": [computing_claimed_sum.0.0.0, computing_claimed_sum.0.1.0],
            "b": [computing_claimed_sum.1.0.0, computing_claimed_sum.1.1.0]
        },
        "commitments_count": proof.commitments.len(),
        "proof_size_bytes": proof.size_estimate(),
        "components": "Computing (x^5+1) + Scheduling (copy via LogUp)"
    });

    fs::write(
        "proof_metadata.json",
        serde_json::to_string_pretty(&metadata).unwrap(),
    )
    .expect("Failed to write metadata");

    println!("✓ Proof metadata saved to proof_metadata.json");
    println!();
    println!("Summary:");
    println!("  Computing: Generates {} rows of x^5+1", n_rows);
    println!("  Scheduling: Copies data via LogUp");
    println!("  LogUp verification: ✓ Sums cancel (proof valid)");
    println!();
    println!("✓ Prover completed successfully!");
}
