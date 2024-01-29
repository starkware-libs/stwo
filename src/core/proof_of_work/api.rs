use super::utils::{ProofOfWorkConfig, ProofOfWorkData, ProofOfWorkProof};
use crate::commitment_scheme::blake2_hash::{Blake2sHash, Blake2sHasher};
use crate::commitment_scheme::hasher::Hasher;
use crate::core::channel::{Blake2sChannel, Channel};

// TODO(ShaharS): generalize to more channels and create a from function in the hash traits.
pub struct ProofOfWork<H: Hasher<NativeType = u8>> {
    _channel: Blake2sChannel,
    _data: ProofOfWorkData<H>,
}

impl ProofOfWork<Blake2sHasher> {
    pub fn new(config: ProofOfWorkConfig, channel: Blake2sChannel) -> Self {
        let digest = channel.get_digest();
        Self {
            _channel: channel,
            _data: ProofOfWorkData::new(Blake2sHash::from(digest), config),
        }
    }

    pub fn prove(&mut self) -> ProofOfWorkProof {
        unimplemented!("Not implemented yet")
    }

    pub fn verify(&mut self, _proof: &ProofOfWorkProof) -> bool {
        unimplemented!("Not implemented yet")
    }
}
