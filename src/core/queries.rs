use std::collections::BTreeSet;
use std::ops::Deref;

use itertools::Itertools;

use super::channel::Channel;
use super::circle::Coset;
use super::poly::circle::{CanonicCoset, CircleDomain};
use super::poly::commitment::DecommitmentPositions;
use super::utils::bit_reverse_index;

pub const UPPER_BOUND_QUERY_BYTES: usize = 4;

/// An ordered set of query indices over a bit reversed [CircleDomain].
pub struct Queries {
    pub positions: Vec<usize>,
    pub log_domain_size: u32,
}

/// A set of [CircleDomain]s over which to evaluate the polynomial for each respective query in
/// [Queries].
pub struct QueryEvaluationDomains(Vec<CircleDomain>);

impl Queries {
    /// Randomizes a set of query indices uniformly over the range [0, 2^`log_query_size`).
    pub fn generate(channel: &mut impl Channel, log_domain_size: u32, n_queries: usize) -> Self {
        let mut queries = BTreeSet::new();
        let mut query_cnt = 0;
        let max_query = (1 << log_domain_size) - 1;
        loop {
            let random_bytes = channel.draw_random_bytes();
            for chunk in random_bytes.chunks_exact(UPPER_BOUND_QUERY_BYTES) {
                let query_bits = u32::from_le_bytes(chunk.try_into().unwrap());
                let quotient_query = query_bits & max_query;
                queries.insert(quotient_query as usize);
                query_cnt += 1;
                if query_cnt == n_queries {
                    return Self {
                        positions: queries.into_iter().collect(),
                        log_domain_size,
                    };
                }
            }
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

    /// Calculates the decommitment position needed for each query given the (log) folding factor.
    pub fn to_decommitment_positions(&self, log_folding_factor: u32) -> DecommitmentPositions {
        DecommitmentPositions(
            self.iter()
                .map(|q| q >> log_folding_factor)
                .map(|q| (q << log_folding_factor..(q + 1) << log_folding_factor))
                .dedup()
                .flatten()
                .collect(),
        )
    }

    /// Calculates the evaluation domains needed for each query given the (log) folding factor.
    pub fn to_evaluation_domains(&self, log_query_size: u32) -> QueryEvaluationDomains {
        assert!(self.log_domain_size > 0);
        let query_domain = CanonicCoset::new(log_query_size);
        let mut query_evaluation_domains = Vec::with_capacity(self.len());
        for bit_reversed_query in self.iter() {
            let query = bit_reverse_index(*bit_reversed_query as u32, log_query_size);
            let initial_index = query_domain.index_at(query as usize);
            let half_coset = Coset::new(initial_index, self.log_domain_size - 1);
            query_evaluation_domains.push(CircleDomain::new(half_coset));
        }
        QueryEvaluationDomains(query_evaluation_domains)
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
    use crate::commitment_scheme::blake2_hash::Blake2sHash;
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
        let queries = Queries {
            positions: (0..1 << log_domain_size).collect(),
            log_domain_size,
        };
        let n_folds = log_domain_size - log_folded_domain_size;
        let ratio = 1 << n_folds;

        let folded_queries = queries.fold(n_folds);
        let repeated_folded_queries = folded_queries
            .iter()
            .flat_map(|q| std::iter::repeat(q).take(ratio));
        for (query, folded_query) in queries.iter().zip(repeated_folded_queries) {
            // Check only the x coordinate since folding might give you the conjugate point.
            assert_eq!(
                values[*query].repeated_double(n_folds).x,
                folded_values[*folded_query].x
            );
        }
    }

    #[test]
    pub fn test_conjugate_queries() {
        let channel = &mut Blake2sChannel::new(Blake2sHash::default());
        let log_domain_size = 7;
        let domain = CanonicCoset::new(log_domain_size).circle_domain();
        let values = domain.iter().collect();
        let values = bit_reverse_vec(&values, log_domain_size);

        // Test random queries one by one because the conjugate queries are sorted.
        for _ in 0..100 {
            let query = Queries::generate(channel, log_domain_size, 1);
            let conjugate_query = query[0] ^ 1;
            let query_and_conjugate = query.to_decommitment_positions(1);
            let mut expected_query_and_conjugate = vec![query[0], conjugate_query];
            expected_query_and_conjugate.sort();
            assert_eq!(query_and_conjugate.0, expected_query_and_conjugate);
            assert_eq!(values[query[0]], values[conjugate_query].conjugate());
        }
    }
}
