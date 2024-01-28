use num_bigint::BigUint;

use crate::commitment_scheme::hasher::Hasher;

pub struct ProofOfWork<H: Hasher<NativeType = u8>> {
    seed: H::Hash,
    work_bits: u32,
}

// TODO(ShaharS): Consider to split to prover and verifier and create traits for them.
impl<H: Hasher<NativeType = u8>> ProofOfWork<H> {
    pub fn new(seed: H::Hash, work_bits: u32) -> Self {
        Self { seed, work_bits }
    }

    pub fn prove(&self) -> u64 {
        let hash_total_output_bits = H::OUTPUT_SIZE as u32 * H::NativeType::BITS;
        let mut nonce = 0u64;
        // TODO(ShaharS): naive implementation, should be replaced with a parallel one.
        loop {
            let hash = BigUint::from_bytes_le(self.hash_with_nonce(nonce).as_ref());
            if hash.bits() <= hash_total_output_bits as u64 - self.work_bits as u64 {
                return nonce;
            }
            nonce += 1;
        }
    }

    pub fn verify(&self, nonce: u64) -> bool {
        let hash_total_output_bits = H::OUTPUT_SIZE as u32 * H::NativeType::BITS;
        let hash = BigUint::from_bytes_le(self.hash_with_nonce(nonce).as_ref());
        hash.bits() <= hash_total_output_bits as u64 - self.work_bits as u64
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

#[cfg(test)]
mod tests {
    use super::ProofOfWork;
    use crate::commitment_scheme::blake2_hash::{Blake2sHash, Blake2sHasher};

    #[test]
    fn test_verify_proof_of_work_success() {
        let proof_of_work_prover = ProofOfWork::<Blake2sHasher> {
            seed: Blake2sHash::from(vec![0; 32]),
            work_bits: 10,
        };
        let valid_salt = 133;

        assert!(proof_of_work_prover.verify(valid_salt));
    }

    #[test]
    fn test_verify_proof_of_work_fail() {
        let proof_of_work_prover = ProofOfWork::<Blake2sHasher> {
            seed: Blake2sHash::from(vec![0; 32]),
            work_bits: 1,
        };
        let invalid_salt = 0;

        assert!(!proof_of_work_prover.verify(invalid_salt));
    }

    #[test]
    fn test_proof_of_work() {
        let proof_of_work_prover = ProofOfWork::<Blake2sHasher> {
            seed: Blake2sHash::from(vec![0; 32]),
            work_bits: 12,
        };

        let salt = proof_of_work_prover.prove();

        assert!(proof_of_work_prover.verify(salt));
    }
}
