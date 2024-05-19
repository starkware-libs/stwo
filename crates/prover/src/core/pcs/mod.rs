//! Implements a FRI polynomial commitment scheme.
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

pub use self::prover::{CommitmentSchemeProof, CommitmentSchemeProver, CommitmentTreeProver};
pub use self::utils::TreeVec;
pub use self::verifier::CommitmentSchemeVerifier;
