use crate::core::channel::Channel;

/// Number of proof-of-work bits ground into the channel before the OODS point is drawn.
///
/// This grind raises the cost for a malicious prover to re-roll the OODS (DEEP) challenge in
/// search of a favorable point. It is independent of the FRI query-phase proof of work
/// (`PcsConfig::pow_bits`).
pub const OODS_POW_BITS: u32 = 10;

pub trait GrindOps<C: Channel> {
    /// Searches for a nonce s.t. mixing it to the channel makes the digest have `pow_bits` leading
    /// zero bits.
    fn grind(channel: &C, pow_bits: u32) -> u64;
}
