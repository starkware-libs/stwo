use poseidon_circuit::{PoseidonComponent, PoseidonElements, PoseidonEval};
use std::fs;
use stwo::core::air::Component;
use stwo::core::channel::Blake2sChannel;
use stwo::core::fields::cm31::CM31;
use stwo::core::fields::m31::M31;
use stwo::core::fields::qm31::QM31;
use stwo::core::pcs::{CommitmentSchemeVerifier, PcsConfig};
use stwo::core::proof::StarkProof;
use stwo::core::vcs::blake2_merkle::{Blake2sMerkleChannel, Blake2sMerkleHasher};
use stwo::core::verifier::verify;
use stwo_constraint_framework::TraceLocationAllocator;

fn main() {
    println!("=== STARK Verifier - Poseidon2 Hash Function ===\n");

    // Load metadata
    println!("Loading metadata from proof_metadata.json...");
    let metadata_json =
        fs::read_to_string("proof_metadata.json").expect("Failed to read proof_metadata.json");
    let metadata: serde_json::Value =
        serde_json::from_str(&metadata_json).expect("Failed to parse metadata");

    let log_n_rows = metadata["log_n_rows"].as_u64().unwrap() as u32;
    let n_instances = metadata["n_instances"].as_u64().unwrap();

    // Reconstruct claimed_sum from metadata
    let claimed_sum_obj = &metadata["claimed_sum"];
    let a = claimed_sum_obj["a"].as_array().unwrap();
    let b = claimed_sum_obj["b"].as_array().unwrap();
    let claimed_sum = QM31(
        CM31::from_m31(
            M31::from(a[0].as_u64().unwrap() as u32),
            M31::from(a[1].as_u64().unwrap() as u32),
        ),
        CM31::from_m31(
            M31::from(b[0].as_u64().unwrap() as u32),
            M31::from(b[1].as_u64().unwrap() as u32),
        ),
    );

    println!("✓ Metadata loaded");
    println!("  Hash function: {}", metadata["hash_function"].as_str().unwrap());
    println!("  Instances: {}", n_instances);
    println!("  Log rows: {}", log_n_rows);
    println!("  Claimed sum: {:?}", claimed_sum);
    println!();

    // Load proof
    println!("Loading proof from proof.json...");
    let proof_json = fs::read_to_string("proof.json").expect("Failed to read proof.json");
    let proof: StarkProof<Blake2sMerkleHasher> =
        serde_json::from_str(&proof_json).expect("Failed to deserialize proof");
    println!("✓ Proof loaded");
    println!("  Proof size: {} bytes", proof.size_estimate());
    println!();

    // Setup verifier and replay channel
    println!("Creating Poseidon component for verification...");
    let config = PcsConfig::default();
    let channel = &mut Blake2sChannel::default();
    let commitment_scheme = &mut CommitmentSchemeVerifier::<Blake2sMerkleChannel>::new(config);

    // Create temporary component to get trace log degree bounds
    // We need to create it once to get the sizes, then recreate with correct lookup elements
    let temp_component = PoseidonComponent::new(
        &mut TraceLocationAllocator::default(),
        PoseidonEval {
            log_n_rows,
            lookup_elements: PoseidonElements::draw(&mut Blake2sChannel::default()),
            claimed_sum,
        },
        claimed_sum,
    );

    // Get the expected column sizes from the component
    let sizes = temp_component.trace_log_degree_bounds();

    // Replay the channel to match prover's state
    // Commit preprocessed (empty)
    commitment_scheme.commit(proof.commitments[0], &sizes[0], channel);

    // Commit trace
    commitment_scheme.commit(proof.commitments[1], &sizes[1], channel);

    // Draw lookup elements (deterministically from channel, matching prover)
    let lookup_elements = PoseidonElements::draw(channel);

    // Commit interaction trace
    commitment_scheme.commit(proof.commitments[2], &sizes[2], channel);

    // Create final component with correct lookup elements
    let component = PoseidonComponent::new(
        &mut TraceLocationAllocator::default(),
        PoseidonEval {
            log_n_rows,
            lookup_elements,
            claimed_sum,
        },
        claimed_sum,
    );
    println!("✓ Component created");
    println!();

    // Verify proof
    println!("Verifying STARK proof...");
    match verify(&[&component], channel, commitment_scheme, proof) {
        Ok(_) => {
            println!("✓✓✓ PROOF VERIFICATION SUCCESSFUL! ✓✓✓");
            println!();
            println!("The proof is cryptographically valid.");
            println!("Poseidon2 hash computations verified: {} instances", n_instances);
        }
        Err(e) => {
            println!("✗✗✗ PROOF VERIFICATION FAILED! ✗✗✗");
            println!("Error: {:?}", e);
            std::process::exit(1);
        }
    }
}
