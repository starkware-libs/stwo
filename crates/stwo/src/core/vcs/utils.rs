use core::iter::Peekable;

use starknet_ff::FieldElement as FieldElement252;

/// Fetches the next node that needs to be decommited in the current Merkle layer.
pub fn next_decommitment_node(
    prev_queries: &mut Peekable<impl Iterator<Item = usize>>,
    layer_queries: &mut Peekable<impl Iterator<Item = usize>>,
) -> Option<usize> {
    prev_queries
        .peek()
        .map(|q| *q / 2)
        .into_iter()
        .chain(layer_queries.peek().into_iter().copied())
        .min()
}

pub fn option_flatten_peekable<'a, I: IntoIterator<Item = &'a usize>>(
    a: Option<I>,
) -> Peekable<core::iter::Copied<core::iter::Flatten<<Option<I> as IntoIterator>::IntoIter>>> {
    a.into_iter().flatten().copied().peekable()
}

/// A utility function used to modify the most significant bits of a felt252.
/// Provided that `n_packed_elements` < 8 and `word` < 2^248, the functions injects
/// `n_packed_elements` into the bits at indices [248:251] of `word`.
///
/// Typically, `word` is a packing of u32s or M31s, `n_packed_elements` is the number
/// of packed elements, and the resulting felt252 is fed into a hash.
/// The purpose of this function in this case is to avoid hash collisions between different-length
/// lists of u32s or M31s that would lead to the same packing.
pub fn add_length_padding(word: &mut FieldElement252, n_packed_elements: usize) {
    let two_pow_124: FieldElement252 = (1u128 << 124).into();
    *word += FieldElement252::from(n_packed_elements) * (two_pow_124 * two_pow_124);
}
