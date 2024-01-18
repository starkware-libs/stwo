use num_bigint::BigUint;

use crate::commitment_scheme::hasher::Hasher;

pub struct ProofOfWork<H: Hasher<NativeType = u8>> {
    seed: H::Hash,
    work_bits: u32,
}

// TODO (ShaharS): Implement ProofOfWorkProver for hashers with other native type.
impl<H: Hasher<NativeType = u8>> ProofOfWork<H> {
    pub fn prove(&self) -> &[H::NativeType] {
        unimplemented!("ProofOfWorkProver::prove")
    }

    pub fn verify(&self, nonce: &[H::NativeType]) -> bool {
        let hash_total_output_bits = H::OUTPUT_SIZE as u32 * H::NativeType::BITS;
        let hash_input = self
            .seed
            .as_ref()
            .iter()
            .chain(nonce.iter())
            .cloned()
            .collect::<Vec<_>>();
        let hash_int = BigUint::from_bytes_le(H::hash(&hash_input).as_ref());
        hash_int.bits() <= hash_total_output_bits as u64 - self.work_bits as u64
    }
}

#[cfg(test)]
mod tests {
    use super::ProofOfWork;
    use crate::commitment_scheme::blake2_hash::{Blake2sHash, Blake2sHasher};

    #[test]
    fn test_proof_of_work() {
        let proof_of_work_prover = ProofOfWork::<Blake2sHasher> {
            seed: Blake2sHash::from(vec![0; 32]),
            work_bits: 7,
        };
        let valid_nonce = 3u8.to_le_bytes();
        let invalid_nonce = 0u8.to_le_bytes();

        assert!(proof_of_work_prover.verify(&valid_nonce));
        assert!(!proof_of_work_prover.verify(&invalid_nonce));
    }
}
