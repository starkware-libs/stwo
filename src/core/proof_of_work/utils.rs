use crate::commitment_scheme::hasher::Hasher;

#[derive(Clone, Debug)]
pub struct ProofOfWorkConfig {
    // Proof of work difficulty.
    pub n_bits: u32,
}

#[derive(Clone, Debug)]
pub struct ProofOfWorkProof {
    pub nonce: u64,
}

pub struct ProofOfWorkData<H: Hasher<NativeType = u8>> {
    config: ProofOfWorkConfig,
    seed: H::Hash,
}

// TODO(ShaharS): Consider to split to prover and verifier and create traits for them.
impl<H: Hasher<NativeType = u8>> ProofOfWorkData<H> {
    pub fn new(seed: H::Hash, config: ProofOfWorkConfig) -> Self {
        Self { config, seed }
    }

    pub fn prove(&self) -> ProofOfWorkProof {
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

    pub fn verify(&self, proof: &ProofOfWorkProof) -> bool {
        let hash = self.hash_with_nonce(proof.nonce);
        check_leading_zeros(hash.as_ref(), self.config.n_bits)
    }

    fn hash_with_nonce(&self, nonce: u64) -> H::Hash {
        let hash_input = self
            .seed
            .as_ref()
            .iter()
            .chain(nonce.to_le_bytes().iter())
            .cloned()
            .collect::<Vec<_>>();
        H::hash(&hash_input)
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
    use super::ProofOfWorkData;
    use crate::commitment_scheme::blake2_hash::{Blake2sHash, Blake2sHasher};
    use crate::core::proof_of_work::utils::{ProofOfWorkConfig, ProofOfWorkProof};

    #[test]
    fn test_verify_proof_of_work_success() {
        let proof_of_work_prover = ProofOfWorkData::<Blake2sHasher> {
            seed: Blake2sHash::from(vec![0; 32]),
            config: ProofOfWorkConfig { n_bits: 11 },
        };
        let proof = ProofOfWorkProof { nonce: 133 };

        assert!(proof_of_work_prover.verify(&proof));
    }

    #[test]
    fn test_verify_proof_of_work_fail() {
        let proof_of_work_prover = ProofOfWorkData::<Blake2sHasher> {
            seed: Blake2sHash::from(vec![0; 32]),
            config: ProofOfWorkConfig { n_bits: 1 },
        };
        let invalid_proof = ProofOfWorkProof { nonce: 0 };

        assert!(!proof_of_work_prover.verify(&invalid_proof));
    }

    #[test]
    fn test_proof_of_work() {
        let proof_of_work_prover = ProofOfWorkData::<Blake2sHasher> {
            seed: Blake2sHash::from(vec![0; 32]),
            config: ProofOfWorkConfig { n_bits: 12 },
        };

        let proof = proof_of_work_prover.prove();

        assert!(proof_of_work_prover.verify(&proof));
    }
}
