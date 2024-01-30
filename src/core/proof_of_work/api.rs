use super::utils::{ProofOfWorkConfig, ProofOfWorkData, ProofOfWorkProof};
use crate::commitment_scheme::blake2_hash::{Blake2sHash, Blake2sHasher};
use crate::commitment_scheme::hasher::Hasher;
use crate::core::channel::{Blake2sChannel, Channel};

// TODO(ShaharS): generalize to more channels and create a from function in the hash traits.
pub struct ProofOfWork<H: Hasher<NativeType = u8>> {
    channel: Blake2sChannel,
    data: ProofOfWorkData<H>,
}

impl ProofOfWork<Blake2sHasher> {
    pub fn new(config: ProofOfWorkConfig, channel: Blake2sChannel) -> Self {
        let digest = channel.get_digest();
        Self {
            channel,
            data: ProofOfWorkData::new(Blake2sHash::from(digest), config),
        }
    }

    pub fn prove(&mut self) -> ProofOfWorkProof {
        let proof = self.data.prove();
        self.channel.mix_with_seed(Blake2sHash::from(
            proof.to_digest(Blake2sHasher::OUTPUT_SIZE).as_ref(),
        ));
        proof
    }

    pub fn verify(&mut self, proof: &ProofOfWorkProof) -> bool {
        let verified = self.data.verify(proof);
        if verified {
            self.channel.mix_with_seed(Blake2sHash::from(
                proof.to_digest(Blake2sHasher::OUTPUT_SIZE).as_ref(),
            ));
        }
        verified
    }
}

#[cfg(test)]
mod tests {
    use crate::commitment_scheme::blake2_hash::{Blake2sHash, Blake2sHasher};
    use crate::core::channel::{Blake2sChannel, Channel};
    use crate::core::proof_of_work::api::ProofOfWork;
    use crate::core::proof_of_work::utils::ProofOfWorkConfig;

    #[test]
    fn test_proof_of_work() {
        let proof_of_work_config = ProofOfWorkConfig { n_bits: 7 };
        let prover_channel = Blake2sChannel::new(Blake2sHash::default());
        let verifier_channel = Blake2sChannel::new(Blake2sHash::default());
        let mut prover =
            ProofOfWork::<Blake2sHasher>::new(proof_of_work_config.clone(), prover_channel);
        let mut verifier =
            ProofOfWork::<Blake2sHasher>::new(proof_of_work_config, verifier_channel);

        let proof = prover.prove();
        let verified = verifier.verify(&proof);

        assert!(verified);
        assert_eq!(prover.channel.get_digest(), verifier.channel.get_digest());
    }
}
