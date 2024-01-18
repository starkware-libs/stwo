use num_bigint::BigUint;

use crate::commitment_scheme::hasher::Hasher;

pub struct ProofOfWork<H: Hasher<NativeType = u8>> {
    pub seed: H::Hash,
    pub work_bits: u32,
}

// TODO (ShaharS): Implement ProofOfWorkProver for hashers with other native type.
impl<H: Hasher<NativeType = u8>> ProofOfWork<H> {
    #[allow(dead_code)]
    fn prove(&self) -> &[H::NativeType] {
        unimplemented!("ProofOfWorkProver::prove")
    }

    #[allow(dead_code)]
    fn verify(&self, nonce: &[H::NativeType]) {
        let hash_input = self
            .seed
            .as_ref()
            .iter()
            .chain(nonce.iter())
            .cloned()
            .collect::<Vec<_>>();
        let hash_uint = BigUint::from_bytes_le(H::hash(&hash_input).as_ref());
        assert!(hash_uint < BigUint::from(2u32).pow(256 - self.work_bits));
    }
}

#[cfg(test)]
mod tests {
    use super::ProofOfWork;
    use crate::commitment_scheme::blake2_hash::{Blake2sHash, Blake2sHasher};
    use crate::core::channel::{Blake2sChannel, Channel};

    #[test]
    fn test_proof_of_work() {
        let channel = Blake2sChannel::new(Blake2sHash::from(vec![0; 32]));
        let proof_of_work_prover = ProofOfWork::<Blake2sHasher> {
            seed: channel.get_digest(),
            work_bits: 2,
        };
        let nonce = 0u8.to_le_bytes();

        proof_of_work_prover.verify(&nonce);
    }
}
