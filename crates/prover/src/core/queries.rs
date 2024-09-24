use std::collections::BTreeSet;
use std::ops::Deref;

use itertools::Itertools;

use super::channel::Channel;
use super::circle::Coset;
use super::poly::circle::CircleDomain;
use super::utils::bit_reverse_index;

pub const UPPER_BOUND_QUERY_BYTES: usize = 4;

/// An ordered set of query indices over a bit reversed [CircleDomain].
#[derive(Debug, Clone)]
pub struct Queries {
    pub positions: Vec<usize>,
    pub log_domain_size: u32,
}

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

    // TODO docs
    #[allow(clippy::missing_safety_doc)]
    pub fn from_positions(positions: Vec<usize>, log_domain_size: u32) -> Self {
        assert!(positions.is_sorted());
        assert!(positions.iter().all(|p| *p < (1 << log_domain_size)));
        Self {
            positions,
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

    pub fn opening_positions(&self, fri_step_size: u32) -> SparseSubCircleDomain {
        assert!(fri_step_size > 0);
        SparseSubCircleDomain {
            domains: self
                .iter()
                .map(|q| SubCircleDomain {
                    coset_index: q >> fri_step_size,
                    log_size: fri_step_size,
                })
                .dedup()
                .collect(),
            large_domain_log_size: self.log_domain_size,
        }
    }
}

impl Deref for Queries {
    type Target = Vec<usize>;

    fn deref(&self) -> &Self::Target {
        &self.positions
    }
}

#[derive(Debug, Clone, Eq, PartialEq)]
pub struct SparseSubCircleDomain {
    pub domains: Vec<SubCircleDomain>,
    pub large_domain_log_size: u32,
}

impl SparseSubCircleDomain {
    pub fn flatten(&self) -> Vec<usize> {
        self.iter()
            .flat_map(|sub_circle_domain| sub_circle_domain.to_decommitment_positions())
            .collect()
    }
}

impl Deref for SparseSubCircleDomain {
    type Target = Vec<SubCircleDomain>;

    fn deref(&self) -> &Self::Target {
        &self.domains
    }
}

/// Represents a circle domain relative to a larger circle domain. The `initial_index` is the bit
/// reversed query index in the larger domain.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub struct SubCircleDomain {
    pub coset_index: usize,
    pub log_size: u32,
}

impl SubCircleDomain {
    /// Calculates the decommitment positions needed for each query given the fri step size.
    pub fn to_decommitment_positions(&self) -> Vec<usize> {
        (self.coset_index << self.log_size..(self.coset_index + 1) << self.log_size).collect()
    }

    /// Returns the represented [CircleDomain].
    pub fn to_circle_domain(&self, query_domain: &CircleDomain) -> CircleDomain {
        let query = bit_reverse_index(self.coset_index << self.log_size, query_domain.log_size());
        let initial_index = query_domain.index_at(query);
        let half_coset = Coset::new(initial_index, self.log_size - 1);
        CircleDomain::new(half_coset)
    }
}

#[cfg(test)]
mod tests {
    use crate::core::channel::Blake2sChannel;
    use crate::core::poly::circle::CanonicCoset;
    use crate::core::queries::Queries;
    use crate::core::utils::bit_reverse;

    #[test]
    fn test_generate_queries() {
        let channel = &mut Blake2sChannel::default();
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
        let channel = &mut Blake2sChannel::default();
        let log_domain_size = 7;
        let domain = CanonicCoset::new(log_domain_size).circle_domain();
        let mut values = domain.iter().collect::<Vec<_>>();
        bit_reverse(&mut values);

        // Test random queries one by one because the conjugate queries are sorted.
        for _ in 0..100 {
            let query = Queries::generate(channel, log_domain_size, 1);
            let conjugate_query = query[0] ^ 1;
            let query_and_conjugate = query.opening_positions(1).flatten();
            let mut expected_query_and_conjugate = vec![query[0], conjugate_query];
            expected_query_and_conjugate.sort();
            assert_eq!(query_and_conjugate, expected_query_and_conjugate);
            assert_eq!(values[query[0]], values[conjugate_query].conjugate());
        }
    }

    #[test]
    pub fn test_decommitment_positions() {
        let channel = &mut Blake2sChannel::default();
        let log_domain_size = 31;
        let n_queries = 100;
        let fri_step_size = 3;

        let queries = Queries::generate(channel, log_domain_size, n_queries);
        let queries_with_added_positions = queries.opening_positions(fri_step_size).flatten();

        assert!(queries_with_added_positions.is_sorted());
        assert_eq!(
            queries_with_added_positions.len(),
            n_queries * (1 << fri_step_size)
        );
    }

    #[test]
    pub fn test_dedup_decommitment_positions() {
        let log_domain_size = 7;

        // Generate all possible queries.
        let queries = Queries {
            positions: (0..1 << log_domain_size).collect(),
            log_domain_size,
        };
        let queries_with_conjugates = queries.opening_positions(log_domain_size - 2).flatten();

        assert_eq!(*queries, *queries_with_conjugates);
    }
}
