use crate::core::channel::Channel;

pub trait GrindOps<C: Channel> {
    /// Searches for a nonce s.t. mixing it to the channel makes the digest have `pow_bits` leading
    /// zero bits.
    fn grind(channel: &C, pow_bits: u32) -> u64;
}
