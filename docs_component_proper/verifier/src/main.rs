use dual_component_circuit::{
    Components, ComponentsStatement0, ComponentsStatement1, ComputationLookupElements,
};
use num_traits::Zero;
use std::fs;
use stwo::core::channel::Blake2sChannel;
use stwo::core::fields::m31::M31;
use stwo::core::fields::cm31::CM31;
use stwo::core::fields::qm31::QM31;
use stwo::core::pcs::{CommitmentSchemeVerifier, PcsConfig};
use stwo::core::proof::StarkProof;
use stwo::core::vcs::blake2_merkle::{Blake2sMerkleChannel, Blake2sMerkleHasher};
use stwo::core::verifier::verify;

fn main() {
    println!("=== STARK Verifier - Dual Component ===\n");

    // Read proof metadata
    println!("Reading proof metadata...");
    let metadata_str = fs::read_to_string("proof_metadata.json")
        .expect("Failed to read proof_metadata.json. Run prover first!");

    let metadata: serde_json::Value =
        serde_json::from_str(&metadata_str).expect("Failed to parse metadata");

    let log_size = metadata["log_size"].as_u64().unwrap() as u32;
    let n_rows = metadata["n_rows"].as_u64().unwrap();

    // Parse claimed sums: QM31(CM31(a0, a1), CM31(b0, b1))
    let sched_a = &metadata["scheduling_claimed_sum"]["a"];
    let sched_b = &metadata["scheduling_claimed_sum"]["b"];
    let scheduling_claimed_sum = QM31(
        CM31::from_m31(
            M31::from(sched_a[0].as_u64().unwrap() as u32),
            M31::from(sched_a[1].as_u64().unwrap() as u32),
        ),
        CM31::from_m31(
            M31::from(sched_b[0].as_u64().unwrap() as u32),
            M31::from(sched_b[1].as_u64().unwrap() as u32),
        ),
    );

    let comp_a = &metadata["computing_claimed_sum"]["a"];
    let comp_b = &metadata["computing_claimed_sum"]["b"];
    let computing_claimed_sum = QM31(
        CM31::from_m31(
            M31::from(comp_a[0].as_u64().unwrap() as u32),
            M31::from(comp_a[1].as_u64().unwrap() as u32),
        ),
        CM31::from_m31(
            M31::from(comp_b[0].as_u64().unwrap() as u32),
            M31::from(comp_b[1].as_u64().unwrap() as u32),
        ),
    );

    println!("✓ Metadata loaded");
    println!();
    println!("Proof claims:");
    println!("  Rows: {}", n_rows);
    println!("  Scheduling claimed sum: {:?}", scheduling_claimed_sum);
    println!("  Computing claimed sum: {:?}", computing_claimed_sum);
    println!(
        "  Sum check: {:?}",
        scheduling_claimed_sum + computing_claimed_sum
    );
    println!();

    // Verify sums cancel
    assert_eq!(
        scheduling_claimed_sum + computing_claimed_sum,
        QM31::zero(),
        "LogUp sums must cancel!"
    );
    println!("✓ LogUp sums cancel (valid)");
    println!();

    // Load proof from file
    println!("Loading proof from file...");
    let proof_json =
        fs::read_to_string("proof.json").expect("Failed to read proof.json. Run prover first!");

    let proof: StarkProof<Blake2sMerkleHasher> =
        serde_json::from_str(&proof_json).expect("Failed to deserialize proof");

    println!("✓ Proof loaded from proof.json");
    println!("  Proof size: {} bytes", proof.size_estimate());
    println!();

    // Create statements
    let statement0 = ComponentsStatement0 { log_size };
    let statement1 = ComponentsStatement1 {
        scheduling_claimed_sum,
        computing_claimed_sum,
    };

    // Verify proof
    println!("Verifying proof...");

    let config = PcsConfig::default();
    let channel = &mut Blake2sChannel::default();
    let commitment_scheme = &mut CommitmentSchemeVerifier::<Blake2sMerkleChannel>::new(config);
    let log_sizes = statement0.log_sizes();

    // Commit preprocessed
    commitment_scheme.commit(proof.commitments[0], &log_sizes[0], channel);

    // Mix statement0
    statement0.mix_into(channel);

    // Commit trace
    commitment_scheme.commit(proof.commitments[1], &log_sizes[1], channel);

    // Draw lookup elements (deterministic from channel)
    let lookup_elements = ComputationLookupElements::draw(channel);

    // Mix statement1
    statement1.mix_into(channel);

    // Commit LogUp
    commitment_scheme.commit(proof.commitments[2], &log_sizes[2], channel);

    // Create components
    let components = Components::new(&statement0, &lookup_elements, &statement1);

    // Verify!
    verify(
        &components.components(),
        channel,
        commitment_scheme,
        proof,
    )
    .unwrap();

    println!("✓ Proof verified successfully!");
    println!();
    println!("Verification result:");
    println!("  ✓ Computing component verified (x^5+1 constraints)");
    println!("  ✓ Scheduling component verified (LogUp copy)");
    println!("  ✓ LogUp sums cancel → data matches");
    println!("  ✓ Proof loaded from file and verified");
}
