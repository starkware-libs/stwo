use core::ops::Deref;

use itertools::Itertools;
use std_shims::{BTreeSet, Vec};

use super::channel::Channel;

pub const UPPER_BOUND_QUERY_BYTES: usize = 4;

/// Draws `n_queries` values in the range `[0, 2^log_domain_size)` from the channel.
pub fn draw_queries(
    channel: &mut impl Channel,
    log_domain_size: u32,
    n_queries: usize,
) -> Vec<usize> {
    let mut raw_positions = Vec::new();
    let query_mask = (1 << log_domain_size) - 1;
    loop {
        let random_words = channel.draw_u32s();
        for word in random_words {
            let quotient_query = word & query_mask;
            raw_positions.push(quotient_query.try_into().unwrap());
            if raw_positions.len() == n_queries {
                return raw_positions;
            }
        }
    }
}

/// An ordered set of query positions.
#[derive(Debug, Clone)]
pub struct Queries {
    /// Query positions sorted in ascending order.
    pub positions: Vec<usize>,
    /// Size of the domain from which the queries were sampled.
    pub log_domain_size: u32,
}

impl Queries {
    /// Creates a [Queries] instance from the given unsorted `raw_positions`.
    pub fn new(raw_positions: &[usize], log_domain_size: u32) -> Self {
        Self {
            positions: BTreeSet::from_iter(raw_positions.iter())
                .into_iter()
                .cloned()
                .collect(),
            log_domain_size,
        }
    }

    /// Calculates the matching query indices in a folded domain (i.e each domain point is doubled)
    /// given `self` (the queries of the original domain) and the number of folds between domains.
    pub fn fold(&self, n_folds: u32) -> Self {
        assert!(n_folds <= self.log_domain_size);
        Self {
            positions: self.iter().map(|q| q >> n_folds).dedup().collect(),
            log_domain_size: self.log_domain_size - n_folds,
        }
    }

    #[cfg(test)]
    pub fn from_positions(positions: Vec<usize>, log_domain_size: u32) -> Self {
        assert!(positions.is_sorted());
        assert!(positions.iter().all(|p| *p < (1 << log_domain_size)));
        Self {
            positions,
            log_domain_size,
        }
    }
}

impl Deref for Queries {
    type Target = Vec<usize>;

    fn deref(&self) -> &Self::Target {
        &self.positions
    }
}

#[cfg(test)]
mod tests {
    use std_shims::Vec;

    use crate::core::channel::Blake2sChannel;
    use crate::core::poly::circle::CanonicCoset;
    use crate::core::queries::{draw_queries, Queries};
    use crate::core::utils::bit_reverse;

    #[test]
    fn test_generate_queries() {
        let channel = &mut Blake2sChannel::default();
        let log_query_size = 31;
        let n_queries = 100;

        let raw_positions = draw_queries(channel, log_query_size, n_queries);
        let queries = Queries::new(&raw_positions, log_query_size);

        assert!(queries.len() == n_queries);
        assert!(queries.iter().is_sorted());
        assert!(*queries.positions.last().unwrap() < 1 << log_query_size);
    }

    #[test]
    pub fn test_folded_queries() {
        let log_domain_size = 7;
        let domain = CanonicCoset::new(log_domain_size).circle_domain();
        let mut values = domain.iter().collect::<Vec<_>>();
        bit_reverse(&mut values);

        let log_folded_domain_size = 5;
        let folded_domain = CanonicCoset::new(log_folded_domain_size).circle_domain();
        let mut folded_values = folded_domain.iter().collect::<Vec<_>>();
        bit_reverse(&mut folded_values);

        // Generate all possible queries.
        let queries = Queries {
            positions: (0..1 << log_domain_size).collect(),
            log_domain_size,
        };
        let n_folds = log_domain_size - log_folded_domain_size;
        let ratio = 1 << n_folds;

        let folded_queries = queries.fold(n_folds);
        let repeated_folded_queries = folded_queries
            .iter()
            .flat_map(|q| core::iter::repeat_n(q, ratio));
        for (query, folded_query) in queries.iter().zip(repeated_folded_queries) {
            // Check only the x coordinate since folding might give you the conjugate point.
            assert_eq!(
                values[*query].repeated_double(n_folds).x,
                folded_values[*folded_query].x
            );
        }
    }
}
