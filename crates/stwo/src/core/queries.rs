use core::ops::Deref;

use itertools::Itertools;
use std_shims::{BTreeSet, Vec};

use super::channel::Channel;

pub const UPPER_BOUND_QUERY_BYTES: usize = 4;

pub(crate) fn query_domain_size(log_domain_size: u32) -> Option<usize> {
    if log_domain_size > (UPPER_BOUND_QUERY_BYTES * u8::BITS as usize) as u32 {
        return None;
    }
    1usize.checked_shl(log_domain_size)
}

pub(crate) fn can_sample_n_unique_queries(log_domain_size: u32, n_queries: usize) -> bool {
    query_domain_size(log_domain_size).is_some_and(|domain_size| n_queries <= domain_size)
}

/// Draws `n_queries` unique values in the range `[0, 2^log_domain_size)` from the channel.
///
/// The channel is sampled deterministically until enough distinct positions have been drawn.
pub fn draw_queries(
    channel: &mut impl Channel,
    log_domain_size: u32,
    n_queries: usize,
) -> Vec<usize> {
    assert!(
        can_sample_n_unique_queries(log_domain_size, n_queries),
        "too many unique FRI query positions requested for query domain"
    );
    assert!(
        log_domain_size <= u32::BITS,
        "query domain size exceeds channel word size"
    );

    let query_mask = if log_domain_size == u32::BITS {
        u32::MAX
    } else {
        (1_u32 << log_domain_size) - 1
    };
    let mut raw_positions = Vec::with_capacity(n_queries);
    let mut seen_positions = BTreeSet::new();

    while raw_positions.len() < n_queries {
        let random_words = channel.draw_u32s();
        for word in random_words {
            let quotient_query = usize::try_from(word & query_mask).unwrap();
            if !seen_positions.insert(quotient_query) {
                continue;
            }
            raw_positions.push(quotient_query);
            if raw_positions.len() == n_queries {
                return raw_positions;
            }
        }
    }

    raw_positions
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

    use crate::core::channel::{Blake2sChannel, Channel};
    use crate::core::fields::qm31::SecureField;
    use crate::core::poly::circle::CanonicCoset;
    use crate::core::queries::{draw_queries, Queries};
    use crate::core::utils::bit_reverse;

    #[derive(Clone, Debug, Default)]
    struct FixedU32Channel {
        draws: Vec<Vec<u32>>,
        next_draw: usize,
    }

    impl FixedU32Channel {
        fn new(draws: Vec<Vec<u32>>) -> Self {
            Self {
                draws,
                next_draw: 0,
            }
        }
    }

    impl Channel for FixedU32Channel {
        const BYTES_PER_HASH: usize = 32;

        fn verify_pow_nonce(&self, _n_bits: u32, _nonce: u64) -> bool {
            true
        }

        fn mix_u32s(&mut self, _data: &[u32]) {}

        fn mix_felts(&mut self, _felts: &[SecureField]) {}

        fn mix_u64(&mut self, _value: u64) {}

        fn draw_secure_felt(&mut self) -> SecureField {
            unimplemented!("test channel only draws u32 words")
        }

        fn draw_secure_felts(&mut self, _n_felts: usize) -> Vec<SecureField> {
            unimplemented!("test channel only draws u32 words")
        }

        fn draw_u32s(&mut self) -> Vec<u32> {
            let words = self.draws[self.next_draw].clone();
            self.next_draw += 1;
            words
        }
    }

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
    fn draw_queries_resamples_until_n_unique_positions() {
        let channel = &mut FixedU32Channel::new(vec![vec![0, 1, 1, 2, 0, 2, 3], vec![3, 4, 5]]);

        let raw_positions = draw_queries(channel, 3, 5);

        assert_eq!(raw_positions, vec![0, 1, 2, 3, 4]);
        assert_eq!(channel.next_draw, 2);
    }

    #[test]
    #[should_panic = "too many unique FRI query positions requested for query domain"]
    fn draw_queries_rejects_impossible_unique_query_count() {
        let channel = &mut FixedU32Channel::new(vec![vec![0, 1, 2, 3]]);

        let _ = draw_queries(channel, 2, 5);
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
