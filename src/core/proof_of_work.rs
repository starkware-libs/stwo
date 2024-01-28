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

    pub fn prove(&self) -> Vec<H::NativeType> {
        let hash_total_output_bits = H::OUTPUT_SIZE as u32 * H::NativeType::BITS;
        let mut nonce = 0u64;
        // TODO(ShaharS): naive implementation, should be replaced with a parallel one.
        loop {
            let nonce_bytes = nonce.to_le_bytes();
            let hash = BigUint::from_bytes_le(self.hash_with_nonce(&nonce_bytes).as_ref());
            if hash.bits() <= hash_total_output_bits as u64 - self.work_bits as u64 {
                return nonce_bytes.to_vec();
            }
            nonce += 1;
        }
    }

    pub fn verify(&self, nonce: &[H::NativeType]) -> bool {
        let hash_total_output_bits = H::OUTPUT_SIZE as u32 * H::NativeType::BITS;
        let hash = BigUint::from_bytes_le(self.hash_with_nonce(nonce).as_ref());
        hash.bits() <= hash_total_output_bits as u64 - self.work_bits as u64
    }

    fn hash_with_nonce(&self, nonce: &[H::NativeType]) -> H::Hash {
        let hash_input = self
            .seed
            .as_ref()
            .iter()
            .chain(nonce.iter())
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
            work_bits: 7,
        };
        let valid_nonce = 3u8.to_le_bytes();

        assert!(proof_of_work_prover.verify(&valid_nonce));
    }

    #[test]
    fn test_verify_proof_of_work_fail() {
        let proof_of_work_prover = ProofOfWork::<Blake2sHasher> {
            seed: Blake2sHash::from(vec![0; 32]),
            work_bits: 3,
        };
        let invalid_nonce = 0u8.to_le_bytes();

        assert!(!proof_of_work_prover.verify(&invalid_nonce));
    }

    #[test]
    fn test_proof_of_work() {
        let proof_of_work_prover = ProofOfWork::<Blake2sHasher> {
            seed: Blake2sHash::from(vec![0; 32]),
            work_bits: 12,
        };

        let nonce_bytes = proof_of_work_prover.prove();

        assert!(proof_of_work_prover.verify(&nonce_bytes));
    }
}
