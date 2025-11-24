use stwo::core::air::Component;
use stwo::core::channel::Blake2sM31Channel;
use stwo::core::pcs::{CommitmentSchemeVerifier, PcsConfig};
use stwo::core::vcs_lifted::blake2_merkle::{Blake2sM31MerkleChannel, Blake2sMerkleChannelGeneric};
use stwo::core::verifier::verify;

use crate::wide_fibonacci_seq::simple_air::{LOG_N_INSTANCES, create_proof};

#[test]
fn verify_simple_proof() {
    let config = PcsConfig::default();
    let (component, proof) = create_proof();

    // Verify.
    let verifier_channel = &mut Blake2sM31Channel::default();
    let commitment_scheme = &mut CommitmentSchemeVerifier::<Blake2sM31MerkleChannel>::new(config);

    // Retrieve the expected column sizes in each commitment interaction, from the AIR.
    let sizes = component.trace_log_degree_bounds();
    println!("sizes in 0: {:?}", sizes[0]);
    commitment_scheme.commit(proof.proof.commitments[0], &[LOG_N_INSTANCES, LOG_N_INSTANCES + 3], verifier_channel);
    commitment_scheme.commit(proof.proof.commitments[1], &sizes[1], verifier_channel);
    commitment_scheme.commit(proof.proof.commitments[2], &sizes[2], verifier_channel);
    verify::<Blake2sMerkleChannelGeneric<true>>(
        &[&component],
        verifier_channel,
        commitment_scheme,
        proof.proof,
    )
    .unwrap();
}
