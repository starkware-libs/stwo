//! Implements a FRI polynomial commitment scheme.
//!
//! This is a protocol where the prover can commit on a set of polynomials and then prove their
//! opening on a set of points.
//! Note: This implementation is not really a polynomial commitment scheme, because we are not in
//! the unique decoding regime. This is enough for a STARK proof though, where we only want to imply
//! the existence of such polynomials, and are ok with having a small decoding list.
//! Note: Opened points cannot come from the commitment domain.

mod prover;
pub mod quotients;
mod utils;
mod verifier;

use serde::{Deserialize, Serialize};

pub use self::prover::{
    CommitmentSchemeProof, CommitmentSchemeProver, CommitmentTreeProver, TreeBuilder,
};
pub use self::utils::TreeVec;
pub use self::verifier::CommitmentSchemeVerifier;
use super::fri::FriConfig;

#[derive(Copy, Debug, Clone, PartialEq, Eq)]
pub struct TreeSubspan {
    pub tree_index: usize,
    pub col_start: usize,
    pub col_end: usize,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct PcsConfig {
    pub pow_bits: u32,
    pub fri_config: FriConfig,
}
impl PcsConfig {
    pub const fn security_bits(&self) -> u32 {
        self.pow_bits + self.fri_config.security_bits()
    }
}

impl Default for PcsConfig {
    fn default() -> Self {
        Self {
            pow_bits: 5,
            fri_config: FriConfig::new(0, 1, 3),
        }
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn test_security_bits() {
        let config = super::PcsConfig {
            pow_bits: 42,
            fri_config: super::FriConfig::new(10, 10, 70),
        };
        assert!(config.security_bits() == 10 * 70 + 42);
    }
}
