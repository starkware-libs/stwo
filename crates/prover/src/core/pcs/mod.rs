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

use std::ops::{Deref, DerefMut};

pub use self::prover::{CommitmentSchemeProof, CommitmentSchemeProver, CommitmentTreeProver};
pub use self::utils::TreeVec;
pub use self::verifier::CommitmentSchemeVerifier;

#[derive(Copy, Debug, Clone)]
pub struct TreeColumnSpan {
    pub tree_index: usize,
    pub col_start: usize,
    pub col_end: usize,
}

#[derive(Debug, Clone)]
pub struct TreePortion(pub TreeVec<TreeColumnSpan>);
impl Deref for TreePortion {
    type Target = TreeVec<TreeColumnSpan>;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}
impl DerefMut for TreePortion {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}

impl TreePortion {
    pub fn take<'a, T>(&self, source: TreeVec<&'a [T]>) -> TreeVec<&'a [T]> {
        self.0
            .as_ref()
            .map(|span| &source[span.tree_index][span.col_start..span.col_end])
    }
}
