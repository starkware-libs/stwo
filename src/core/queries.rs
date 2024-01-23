use std::collections::BTreeSet;
use std::ops::Deref;

use itertools::Itertools;

use super::channel::Channel;
use super::circle::Coset;
use super::poly::circle::{CanonicCoset, CircleDomain};
use super::poly::commitment::DecommitmentPositions;
use super::utils::bit_reverse_index;

// TODO(AlonH): Move file to fri directory.

pub const UPPER_BOUND_QUERY_BYTES: usize = 4;

// TODO(AlonH): Add log size field to the struct.
/// An ordered set of query indices over a bit reversed [CircleDomain].
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
    pub fn fold(&self, n_folds: u32) -> Self {
        Self(self.iter().map(|q| q >> n_folds).dedup().collect())
    }

    pub fn to_sub_circle_domains(&self, fri_step_size: u32) -> SparseSubCircleDomain {
        assert!(fri_step_size > 0);
        SparseSubCircleDomain(
            self.iter()
                .map(|q| SubCircleDomain {
                    coset_index: q >> fri_step_size,
                    log_size: fri_step_size,
                })
                .dedup()
                .collect(),
        )
    }
}

impl Deref for Queries {
    type Target = Vec<usize>;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

pub struct SparseSubCircleDomain(pub Vec<SubCircleDomain>);

impl SparseSubCircleDomain {
    pub fn to_decommitment_positions(&self) -> DecommitmentPositions {
        DecommitmentPositions(
            self.iter()
                .flat_map(|sub_circle_domain| sub_circle_domain.to_decommitment_positions().0)
                .collect(),
        )
    }
}

impl Deref for SparseSubCircleDomain {
    type Target = Vec<SubCircleDomain>;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

/// Represents a circle domain relative to a larger circle domain. The `initial_index` is the bit
/// reversed query index in the larger domain.
#[derive(PartialEq, Eq, PartialOrd, Ord)]
pub struct SubCircleDomain {
    pub coset_index: usize,
    pub log_size: u32,
}

impl SubCircleDomain {
    /// Calculates the decommitment positions needed for each query given the fri step size.
    pub fn to_decommitment_positions(&self) -> DecommitmentPositions {
        DecommitmentPositions(
            (self.coset_index << self.log_size..(self.coset_index + 1) << self.log_size).collect(),
        )
    }

    /// Returns the represented [CircleDomain].
    pub fn to_circle_domain(&self, query_domain: &CanonicCoset) -> CircleDomain {
        let query = bit_reverse_index(
            (self.coset_index << self.log_size) as u32,
            query_domain.log_size(),
        );
        let initial_index = query_domain.index_at(query as usize);
        let half_coset = Coset::new(initial_index, self.log_size - 1);
        CircleDomain::new(half_coset)
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
        let queries = Queries((0..1 << log_domain_size).collect());
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
            let conjugate_query = query.0[0] ^ 1;
            let query_and_conjugate = query.to_sub_circle_domains(1).to_decommitment_positions();
            let mut expected_query_and_conjugate = vec![query.0[0], conjugate_query];
            expected_query_and_conjugate.sort();
            assert_eq!(query_and_conjugate.0, expected_query_and_conjugate);
            assert_eq!(values[query.0[0]], values[conjugate_query].conjugate());
        }
    }
}
