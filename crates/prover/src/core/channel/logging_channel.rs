use std::fmt::Debug;

use tracing::{debug, debug_span};

use crate::core::backend::simd::SimdBackend;
use crate::core::backend::BackendForChannel;
use crate::core::channel::{Channel, MerkleChannel};
use crate::core::fields::qm31::SecureField;
use crate::core::proof_of_work::GrindOps;
use crate::core::vcs::ops::{MerkleHasher, MerkleOps};

#[derive(Debug, Clone, Default)]
pub struct LoggingChannel<C: Channel> {
    pub channel: C,
}

impl<C: Channel> Channel for LoggingChannel<C> {
    const BYTES_PER_HASH: usize = C::BYTES_PER_HASH;

    fn trailing_zeros(&self) -> u32 {
        self.channel.trailing_zeros()
    }

    fn mix_felts(&mut self, felts: &[SecureField]) {
        let _ = debug_span!("Channel mix_felts");
        log_invocation(&mut self.channel, C::mix_felts, felts)
    }

    fn mix_u64(&mut self, value: u64) {
        let _ = debug_span!("Channel mix_u64");
        log_invocation(&mut self.channel, C::mix_u64, value)
    }

    fn draw_felt(&mut self) -> SecureField {
        let _ = debug_span!("Channel draw_felt");
        log_invocation(&mut self.channel, |channel, _| channel.draw_felt(), ())
    }

    fn draw_felts(&mut self, n_felts: usize) -> Vec<SecureField> {
        let _ = debug_span!("Channel draw_felts");
        log_invocation(&mut self.channel, C::draw_felts, n_felts)
    }

    fn draw_random_bytes(&mut self) -> Vec<u8> {
        let _ = debug_span!("Channel draw_random_bytes");
        log_invocation(
            &mut self.channel,
            |channel, _| channel.draw_random_bytes(),
            (),
        )
    }
}

#[derive(Default)]
pub struct LoggingMerkleChannel<MC: MerkleChannel> {
    phantom: std::marker::PhantomData<MC>,
}

impl<MC: MerkleChannel> MerkleChannel for LoggingMerkleChannel<MC> {
    type C = LoggingChannel<MC::C>;

    type H = MC::H;

    fn mix_root(channel: &mut Self::C, root: <Self::H as MerkleHasher>::Hash) {
        let _ = debug_span!("Channel mix_root");
        log_invocation(&mut channel.channel, MC::mix_root, root)
    }
}

impl<C: Channel> GrindOps<LoggingChannel<C>> for SimdBackend
where
    SimdBackend: GrindOps<C>,
{
    fn grind(channel: &LoggingChannel<C>, pow_bits: u32) -> u64 {
        let _ = debug_span!("Channel grind");
        let res = <SimdBackend as GrindOps<C>>::grind(&channel.channel, pow_bits);
        debug!("Grind result: {}", res);
        res
    }
}

impl<B, MC> BackendForChannel<LoggingMerkleChannel<MC>> for B
where
    B: BackendForChannel<MC> + GrindOps<LoggingChannel<MC::C>> + MerkleOps<MC::H>,
    MC: MerkleChannel,
{
}

fn log_invocation<F: FnOnce(&mut C, I) -> O, I: Debug, O: Debug, C: Channel>(
    channel: &mut C,
    f: F,
    input: I,
) -> O {
    debug!("State: {:?}", channel);

    let input_dbg = format!("{:?}", input);
    if input_dbg != "()" {
        debug!("Input: {}", input_dbg);
    }
    let output = f(channel, input);
    let output_dbg = format!("{:?}", output);
    if output_dbg != "()" {
        debug!("Output: {}", output_dbg);
    }
    debug!("State: {:?}", channel);
    output
}
#[cfg(test)]
mod tests {
    use rand::rngs::SmallRng;
    use rand::{Rng, SeedableRng};

    use super::*;
    use crate::core::channel::Blake2sChannel;

    /// To view the output, run:
    /// `RUST_LOG_SPAN_EVENTS=new RUST_LOG=debug
    ///   cargo t test_logging_channel -- --nocapture`
    #[test_log::test]
    fn test_logging_channel() {
        let mut rng = SmallRng::seed_from_u64(0);

        // Create both channels
        let mut logging_channel = LoggingChannel::<Blake2sChannel>::default();
        let mut regular_channel = Blake2sChannel::default();

        let felts = vec![
            rng.gen::<SecureField>(),
            rng.gen::<SecureField>(),
            rng.gen::<SecureField>(),
        ];
        logging_channel.mix_felts(&felts);
        regular_channel.mix_felts(&felts);

        let value = rng.gen::<u64>();
        logging_channel.mix_u64(value);
        regular_channel.mix_u64(value);

        let felt1 = logging_channel.draw_felt();
        let felt2 = regular_channel.draw_felt();
        assert_eq!(felt1, felt2);

        let n_felts = rng.gen_range(1..10);
        let felts1 = logging_channel.draw_felts(n_felts);
        let felts2 = regular_channel.draw_felts(n_felts);
        assert_eq!(felts1, felts2);

        let bytes1 = logging_channel.draw_random_bytes();
        let bytes2 = regular_channel.draw_random_bytes();
        assert_eq!(bytes1, bytes2);

        let pow_bits = 8;
        let logging_result = SimdBackend::grind(&logging_channel, pow_bits);
        let regular_result = SimdBackend::grind(&regular_channel, pow_bits);
        assert_eq!(logging_result, regular_result);

        assert_eq!(logging_channel.channel.digest(), regular_channel.digest());
    }
}
