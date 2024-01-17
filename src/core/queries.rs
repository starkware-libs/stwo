use std::collections::BTreeSet;
use std::ops::Deref;

use super::channel::Channel;

pub const UPPER_BOUND_QUERY_BYTES: usize = 4;

/// A set of query indices over a `CircleDomain`.
pub struct Queries(pub Vec<usize>);

impl Queries {
    /// Randomizes a set of query indices uniformly over the range [0, 2^`log_query_size`).
    pub fn generate(channel: &mut impl Channel, log_query_size: u32, n_queries: usize) -> Self {
        let mut queries = BTreeSet::new();
        let mut query_cnt = 0;
        let max_query = (1 << log_query_size) - 1;
        loop {
            let random_bytes = channel.draw_random_bytes();
            for chunk in random_bytes.chunks_exact(UPPER_BOUND_QUERY_BYTES) {
                let query_bits = u32::from_le_bytes(chunk.try_into().unwrap());
                let quotient_query = query_bits & max_query;
                queries.insert(quotient_query as usize);
                query_cnt += 1;
                if query_cnt == n_queries {
                    return Self(queries.into_iter().collect());
                }
            }
        }
    }

    /// Calculates the matching query indices in a folded domain (i.e each domain point is doubled)
    /// given `self` (the queries of the original domain) and the number of folds between domains.
    pub fn iter_folded(&self, n_folds: u32) -> impl Iterator<Item = usize> + Clone + '_ {
        self.iter().map(move |q| q >> n_folds) // move is needed to move n_folds into the closure.
    }

    /// Calculates the conjugate query indices.
    pub fn iter_conjugate(&self) -> impl Iterator<Item = usize> + Clone + '_ {
        self.iter().map(move |q| q ^ 1)
    }
}

impl Deref for Queries {
    type Target = Vec<usize>;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

#[cfg(test)]
mod tests {
    use crate::commitment_scheme::blake2_hash::Blake2sHash;
    use crate::commitment_scheme::utils::tests::generate_test_queries;
    use crate::core::channel::{Blake2sChannel, Channel};
    use crate::core::poly::circle::CanonicCoset;
    use crate::core::queries::Queries;
    use crate::core::utils::bit_reverse_vec;

    #[test]
    fn test_generate_queries() {
        let channel = &mut Blake2sChannel::new(Blake2sHash::default());
        let log_query_size = 31;
        let n_queries = 100;

        let queries = Queries::generate(channel, log_query_size, n_queries);

        assert!(queries.len() == n_queries);
        for query in queries.iter() {
            assert!(*query < 1 << log_query_size);
        }
    }

    #[test]
    pub fn test_folded_queries() {
        let log_domain_size = 7;
        let domain = CanonicCoset::new(log_domain_size).circle_domain();
        let values = domain.iter().collect();
        let values = bit_reverse_vec(&values, log_domain_size);

        let log_folded_domain_size = 5;
        let folded_domain = CanonicCoset::new(log_folded_domain_size).circle_domain();
        let folded_values = folded_domain.iter().collect();
        let folded_values = bit_reverse_vec(&folded_values, log_folded_domain_size);

        // Generate all possible queries.
        let queries = Queries(generate_test_queries(
            1 << log_domain_size,
            1 << log_domain_size,
        ));
        let n_folds = log_domain_size - log_folded_domain_size;

        for (query, folded_query) in queries.iter().zip(queries.iter_folded(n_folds)) {
            // Check only the x coordinate since folding might give you the conjugate point.
            assert_eq!(
                values[*query].repeated_double(n_folds).x,
                folded_values[folded_query].x
            );
        }
    }

    #[test]
    pub fn test_conjugate_queries() {
        let log_domain_size = 7;
        let domain = CanonicCoset::new(log_domain_size).circle_domain();
        let values = domain.iter().collect();
        let values = bit_reverse_vec(&values, log_domain_size);

        // Generate all possible queries.
        let queries = Queries(generate_test_queries(
            1 << log_domain_size,
            1 << log_domain_size,
        ));

        for (query, conjugate_query) in queries.iter().zip(queries.iter_conjugate()) {
            assert_eq!(values[*query], values[conjugate_query].conjugate());
        }
    }
}
