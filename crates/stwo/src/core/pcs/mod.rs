//! Implements a FRI polynomial commitment scheme.
//!
//! This is a protocol where the prover can commit on a set of polynomials and then prove their
//! opening on a set of points.
//! Note: This implementation is not really a polynomial commitment scheme, because we are not in
//! the unique decoding regime. This is enough for a STARK proof though, where we only want to imply
//! the existence of such polynomials, and are ok with having a small decoding list.
//! Note: Opened points cannot come from the commitment domain.

pub mod quotients;
pub mod utils;
mod verifier;

use serde::{Deserialize, Serialize};

pub use self::utils::TreeVec;
pub use self::verifier::CommitmentSchemeVerifier;
use super::channel::Channel;
use super::fields::qm31::SecureField;
use super::fri::FriConfig;

#[derive(Copy, Debug, Clone, PartialEq, Eq)]
pub struct TreeSubspan {
    pub tree_index: usize,
    pub col_start: usize,
    pub col_end: usize,
}

/// Controls the size of the lifting domain used by the commitment scheme. This size includes the
/// `log_blowup_factor`.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
pub enum LiftingLogSize {
    /// Lift all polynomials to the domain of exactly this log size.
    Fixed(u32),
    /// Lift each tree's polynomials to the largest domain within that tree - the natural sizing.
    /// An implicit assumption here is that the largest domains are all of equal size across trees,
    /// except possibly for the preprocessed tree.
    Auto,
    /// Lift to at least this log size, bumped up to the call site's natural minimum when the
    /// latter is larger, i.e. `max(x, natural_min)`. The natural minimum is the log size of the
    /// evaluation domain being committed, which already includes the `log_blowup_factor`. At the
    /// preprocessed-tree commitment that is `preprocessed_trace_log_size + log_blowup_factor`, so
    /// this resolves to `max(x, preprocessed_trace_log_size + log_blowup_factor)`.
    AtLeast(u32),
}

impl LiftingLogSize {
    /// Resolves to a concrete lifting log size given the natural minimum of the call site (the size
    /// the tree/domain would take without any override).
    pub fn resolve(self, min_log_size: u32) -> u32 {
        match self {
            LiftingLogSize::Auto => min_log_size,
            LiftingLogSize::Fixed(log_size) => {
                if log_size < min_log_size {
                    panic!("Lifting log size is too small {log_size}. It must be at least {min_log_size}.")
                }
                log_size
            }
            LiftingLogSize::AtLeast(log_size) => log_size.max(min_log_size),
        }
    }

    /// The `(tag, value)` pair mixed into the Fiat-Shamir channel, in that order (tag first), for
    /// domain separation between the variants. `Auto`/`Fixed` keep tag `0`, matching the historical
    /// mixing of `lifting_log_size.unwrap_or(0)` in a single slot; `AtLeast` uses tag `1`.
    const fn channel_tag_and_value(&self) -> (u32, u32) {
        match self {
            LiftingLogSize::Auto => (0, 0),
            LiftingLogSize::Fixed(log_size) => (0, *log_size),
            LiftingLogSize::AtLeast(log_size) => (1, *log_size),
        }
    }
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
/// Configuration parameters for the committment scheme prover.
pub struct PcsConfig {
    /// The number of proof of work bits before the FRI queries.
    pub pow_bits: u32,
    pub fri_config: FriConfig,
    /// Controls the size of the lifting domain. See [`LiftingLogSize`].
    pub lifting_log_size: LiftingLogSize,
}
impl PcsConfig {
    pub const fn security_bits(&self) -> u32 {
        self.pow_bits + self.fri_config.security_bits()
    }

    pub fn mix_into(&self, channel: &mut impl Channel) {
        let PcsConfig {
            pow_bits,
            fri_config,
            lifting_log_size,
        } = self;
        let FriConfig {
            log_blowup_factor,
            n_queries,
            log_last_layer_degree_bound,
            fold_step,
        } = fri_config;

        // The lifting size is domain-separated by a variant tag mixed before the value.
        let (lifting_tag, lifting_value) = lifting_log_size.channel_tag_and_value();
        channel.mix_felts(&[
            SecureField::from_u32_unchecked(
                *pow_bits,
                *log_blowup_factor,
                *n_queries as u32,
                *log_last_layer_degree_bound,
            ),
            SecureField::from_u32_unchecked(*fold_step, lifting_tag, lifting_value, 0),
        ]);
    }
}

impl Default for PcsConfig {
    fn default() -> Self {
        Self {
            pow_bits: 10,
            fri_config: FriConfig::new(0, 1, 3, 1),
            lifting_log_size: LiftingLogSize::Auto,
        }
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn test_security_bits() {
        let config = super::PcsConfig {
            pow_bits: 42,
            fri_config: super::FriConfig::new(10, 10, 70, 1),
            lifting_log_size: super::LiftingLogSize::Auto,
        };
        assert!(config.security_bits() == 10 * 70 + 42);
    }

    #[test]
    fn test_lifting_log_size_resolve() {
        use super::LiftingLogSize;

        // `Auto` uses the call site's natural minimum.
        assert_eq!(LiftingLogSize::Auto.resolve(7), 7);
        // `Fixed` returns its value when it is at least the natural minimum (equal is allowed).
        assert_eq!(LiftingLogSize::Fixed(10).resolve(7), 10);
        assert_eq!(LiftingLogSize::Fixed(7).resolve(7), 7);
        // `AtLeast` bumps up to the natural minimum, never below it.
        assert_eq!(LiftingLogSize::AtLeast(10).resolve(7), 10);
        assert_eq!(LiftingLogSize::AtLeast(5).resolve(7), 7);
        assert_eq!(LiftingLogSize::AtLeast(7).resolve(7), 7);
    }

    #[test]
    #[should_panic(expected = "Lifting log size is too small")]
    fn test_lifting_log_size_resolve_fixed_below_min_panics() {
        // A `Fixed` size strictly below the natural minimum is rejected.
        let _ = super::LiftingLogSize::Fixed(5).resolve(7);
    }

    #[test]
    fn test_lifting_log_size_channel_tag_and_value() {
        use super::LiftingLogSize;

        // Tag comes first; `Auto`/`Fixed` keep tag 0 (transcript-compatible with the historical
        // `Option<u32>` mixing of `unwrap_or(0)`), `AtLeast` uses tag 1 for domain separation.
        assert_eq!(LiftingLogSize::Auto.channel_tag_and_value(), (0, 0));
        assert_eq!(LiftingLogSize::Fixed(8).channel_tag_and_value(), (0, 8));
        assert_eq!(LiftingLogSize::AtLeast(8).channel_tag_and_value(), (1, 8));
    }

    #[test]
    fn test_lifting_log_size_mix_domain_separation() {
        use super::{LiftingLogSize, PcsConfig};
        use crate::core::channel::{Blake2sChannel, Channel};
        use crate::core::fri::FriConfig;

        let mixed = |lifting_log_size| {
            let config = PcsConfig {
                pow_bits: 10,
                fri_config: FriConfig::new(0, 1, 3, 1),
                lifting_log_size,
            };
            let mut channel = Blake2sChannel::default();
            config.mix_into(&mut channel);
            channel.draw_secure_felt()
        };

        // `Fixed(x)` and `AtLeast(x)` carry the same value and differ only by the variant tag, so
        // they must be domain-separated in the Fiat-Shamir channel.
        assert_ne!(
            mixed(LiftingLogSize::Fixed(8)),
            mixed(LiftingLogSize::AtLeast(8))
        );
        // `Auto` and `AtLeast(0)` both carry value 0 and differ only by the tag.
        assert_ne!(
            mixed(LiftingLogSize::Auto),
            mixed(LiftingLogSize::AtLeast(0))
        );
    }
}
