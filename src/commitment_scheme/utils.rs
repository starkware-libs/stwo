use std::iter::Peekable;

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
) -> Peekable<std::iter::Copied<std::iter::Flatten<<Option<I> as IntoIterator>::IntoIter>>> {
    a.into_iter().flatten().copied().peekable()
}
