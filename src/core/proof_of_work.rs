use crate::commitment_scheme::blake2_hash::{Blake2sHash, Blake2sHasher};
use crate::commitment_scheme::hasher::Hasher;
use crate::core::channel::{Blake2sChannel, Channel};

// TODO(ShaharS): generalize to more channels and create a from function in the hash traits.
pub struct ProofOfWork {
    channel: Blake2sChannel,
    config: ProofOfWorkConfig,
}

#[derive(Clone, Debug)]
pub struct ProofOfWorkConfig {
    // Proof of work difficulty.
    pub n_bits: u32,
}

#[derive(Clone, Debug)]
pub struct ProofOfWorkProof {
    pub nonce: u64,
}

impl ProofOfWorkProof {
    // TODO(ShaharS): Support multiple digests.
    pub fn to_digest(&self, size: usize) -> Vec<u8> {
        let mut padded = vec![0; size];
        // Copy the elements from the original array to the new array
        padded[..8].copy_from_slice(&self.nonce.to_le_bytes());
        padded
    }
}

impl ProofOfWork {
    pub fn new(config: ProofOfWorkConfig, channel: Blake2sChannel) -> Self {
        Self { channel, config }
    }

    pub fn prove(&mut self) -> ProofOfWorkProof {
        let proof = self.grind();
        self.channel.mix_with_seed(Blake2sHash::from(
            proof.to_digest(Blake2sHasher::OUTPUT_SIZE).as_ref(),
        ));
        proof
    }

    pub fn verify(&mut self, proof: &ProofOfWorkProof) -> bool {
        let verified = check_leading_zeros(
            self.hash_with_nonce(proof.nonce).as_ref(),
            self.config.n_bits,
        );

        if verified {
            self.channel.mix_with_seed(Blake2sHash::from(
                proof.to_digest(Blake2sHasher::OUTPUT_SIZE).as_ref(),
            ));
        }
        verified
    }

    fn grind(&self) -> ProofOfWorkProof {
        let mut nonce = 0u64;
        // TODO(ShaharS): naive implementation, should be replaced with a parallel one.
        loop {
            let hash = self.hash_with_nonce(nonce);
            if check_leading_zeros(hash.as_ref(), self.config.n_bits) {
                return ProofOfWorkProof { nonce };
            }
            nonce += 1;
        }
    }

    fn hash_with_nonce(&self, nonce: u64) -> Blake2sHash {
        let hash_input = self
            .channel
            .get_digest()
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

#[cfg(test)]
mod tests {
    use crate::commitment_scheme::blake2_hash::Blake2sHash;
    use crate::core::channel::{Blake2sChannel, Channel};
    use crate::core::proof_of_work::{ProofOfWork, ProofOfWorkConfig, ProofOfWorkProof};

    #[test]
    fn test_verify_proof_of_work_success() {
        let mut proof_of_work_prover = ProofOfWork {
            channel: Blake2sChannel::new(Blake2sHash::from(vec![0; 32])),
            config: ProofOfWorkConfig { n_bits: 11 },
        };
        let proof = ProofOfWorkProof { nonce: 133 };

        assert!(proof_of_work_prover.verify(&proof));
    }

    #[test]
    fn test_verify_proof_of_work_fail() {
        let mut proof_of_work_prover = ProofOfWork {
            channel: Blake2sChannel::new(Blake2sHash::from(vec![0; 32])),
            config: ProofOfWorkConfig { n_bits: 1 },
        };
        let invalid_proof = ProofOfWorkProof { nonce: 0 };

        assert!(!proof_of_work_prover.verify(&invalid_proof));
    }

    #[test]
    fn test_proof_of_work() {
        let proof_of_work_config = ProofOfWorkConfig { n_bits: 12 };
        let prover_channel = Blake2sChannel::new(Blake2sHash::default());
        let verifier_channel = Blake2sChannel::new(Blake2sHash::default());
        let mut prover = ProofOfWork::new(proof_of_work_config.clone(), prover_channel);
        let mut verifier = ProofOfWork::new(proof_of_work_config, verifier_channel);

        let proof = prover.prove();
        let verified = verifier.verify(&proof);

        assert!(verified);
        assert_eq!(prover.channel.get_digest(), verifier.channel.get_digest());
    }
}
