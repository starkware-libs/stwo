use thiserror::Error;
use tracing::{span, Level};

use super::channel::Poseidon252Channel;
use crate::core::channel::Channel;
use crate::core::vcs::blake2_hash::{Blake2sHash, Blake2sHasher};
use crate::core::vcs::hasher::Hasher;

// TODO(ShaharS): generalize to more channels and create a from function in the hash traits.
pub struct ProofOfWork {
    // Proof of work difficulty.
    pub n_bits: u32,
}

#[derive(Clone, Debug)]
pub struct ProofOfWorkProof {
    pub nonce: u64,
}

impl ProofOfWork {
    pub fn new(n_bits: u32) -> Self {
        Self { n_bits }
    }

    pub fn prove(&self, channel: &mut Poseidon252Channel) -> ProofOfWorkProof {
        let _span = span!(Level::INFO, "Proof of work").entered();
        let seed = channel.get_digest().as_ref().to_bytes_be().to_vec();
        let proof = self.grind(seed);
        channel.mix_nonce(proof.nonce);
        proof
    }

    pub fn verify(
        &self,
        channel: &mut Poseidon252Channel,
        proof: &ProofOfWorkProof,
    ) -> Result<(), ProofOfWorkVerificationError> {
        let seed = channel.get_digest().as_ref().to_bytes_be().to_vec();
        let verified = check_leading_zeros(
            self.hash_with_nonce(&seed, proof.nonce).as_ref(),
            self.n_bits,
        );

        if !verified {
            return Err(ProofOfWorkVerificationError::ProofOfWorkVerificationFailed);
        }

        channel.mix_nonce(proof.nonce);
        Ok(())
    }

    fn grind(&self, seed: Vec<u8>) -> ProofOfWorkProof {
        let mut nonce = 0u64;
        // TODO(ShaharS): naive implementation, should be replaced with a parallel one.
        loop {
            let hash = self.hash_with_nonce(&seed, nonce);
            if check_leading_zeros(hash.as_ref(), self.n_bits) {
                return ProofOfWorkProof { nonce };
            }
            nonce += 1;
        }
    }

    fn hash_with_nonce(&self, seed: &[u8], nonce: u64) -> Blake2sHash {
        let hash_input = seed
            .iter()
            .chain(nonce.to_le_bytes().iter())
            .cloned()
            .collect::<Vec<_>>();
        Blake2sHasher::hash(&hash_input)
    }
}

/// Check that the prefix leading zeros is greater than `bound_bits`.
fn check_leading_zeros(bytes: &[u8], bound_bits: u32) -> bool {
    let mut n_bits = 0;
    // bytes are in little endian order.
    for byte in bytes.iter().rev() {
        if *byte == 0 {
            n_bits += 8;
        } else {
            n_bits += byte.leading_zeros();
            break;
        }
    }
    n_bits >= bound_bits
}

#[derive(Clone, Copy, Debug, Error)]
pub enum ProofOfWorkVerificationError {
    #[error("Proof of work verification failed.")]
    ProofOfWorkVerificationFailed,
}

#[cfg(test)]
mod tests {
    use starknet_ff::FieldElement as FieldElement252;

    use crate::core::channel::{Channel, Poseidon252Channel};
    use crate::core::proof_of_work::{ProofOfWork, ProofOfWorkProof};

    #[test]
    fn test_verify_proof_of_work_success() {
        let mut channel = Poseidon252Channel::new(FieldElement252::default());
        let proof_of_work_prover = ProofOfWork { n_bits: 11 };
        let proof = ProofOfWorkProof { nonce: 133 };

        proof_of_work_prover.verify(&mut channel, &proof).unwrap();
    }

    #[test]
    fn test_verify_proof_of_work_fail() {
        let mut channel = Poseidon252Channel::new(FieldElement252::default());
        let proof_of_work_prover = ProofOfWork { n_bits: 1 };
        let invalid_proof = ProofOfWorkProof { nonce: 0 };

        proof_of_work_prover
            .verify(&mut channel, &invalid_proof)
            .unwrap_err();
    }

    #[test]
    fn test_proof_of_work() {
        let n_bits = 12;
        let mut prover_channel = Poseidon252Channel::new(FieldElement252::default());
        let mut verifier_channel = Poseidon252Channel::new(FieldElement252::default());
        let prover = ProofOfWork::new(n_bits);
        let verifier = ProofOfWork::new(n_bits);

        let proof = prover.prove(&mut prover_channel);
        verifier.verify(&mut verifier_channel, &proof).unwrap();

        assert_eq!(prover_channel.get_digest(), verifier_channel.get_digest());
    }
}
