use std::collections::BTreeSet;

use super::channel::{Blake2sChannel, Channel};

pub const UPPER_BOUND_QUERY_BYTES: usize = 4;

pub fn generate_queries(
    channel: &mut Blake2sChannel,
    log_query_size: u32,
    n_queries: usize,
) -> BTreeSet<(usize, usize)> {
    let mut queries = BTreeSet::new();
    let mut query_cnt = 0;
    let max_query = (1 << log_query_size) - 1;
    loop {
        let random_bytes = channel.draw_random_bytes();
        for chunk in random_bytes.chunks_exact(UPPER_BOUND_QUERY_BYTES) {
            let query_bits = u32::from_le_bytes(chunk.try_into().unwrap());
            let quotient_query = query_bits & max_query;
            let conjugate_query = max_query - quotient_query;
            queries.insert((quotient_query as usize, conjugate_query as usize));
            query_cnt += 1;
            if query_cnt == n_queries {
                return queries;
            }
        }
    }
}

/// Calculates the locations of the queries in the trace commitment domain.
pub fn get_projected_queries(
    quotient_queries: &BTreeSet<(usize, usize)>,
    log_domain_ratio: u32,
) -> BTreeSet<(usize, usize)> {
    let domain_ratio = 1 << log_domain_ratio;
    quotient_queries
        .iter()
        .map(|(q, c)| (q / domain_ratio, c / domain_ratio))
        .collect()
}

pub fn flatten_queries(queries: BTreeSet<(usize, usize)>) -> BTreeSet<usize> {
    queries.into_iter().flat_map(|(q, c)| vec![q, c]).collect()
}

#[cfg(test)]
mod tests {
    use super::{generate_queries, get_projected_queries};
    use crate::commitment_scheme::blake2_hash::Blake2sHash;
    use crate::core::channel::{Blake2sChannel, Channel};
    use crate::core::poly::circle::CanonicCoset;

    #[test]
    fn test_generate_queries() {
        let channel = &mut Blake2sChannel::new(Blake2sHash::default());
        let log_query_size = 30;
        let n_queries = 100;
        let query_coset = CanonicCoset::new(log_query_size);

        let queries = generate_queries(channel, log_query_size, n_queries);

        assert_eq!(queries.len(), n_queries);
        for (query, conjugate_query) in queries.into_iter() {
            assert!(query < (1 << log_query_size));
            assert!(conjugate_query < (1 << log_query_size));
            assert_eq!(query_coset.at(query), -query_coset.at(conjugate_query));
        }
    }

    #[test]
    fn test_get_projected_queries() {
        let channel = &mut Blake2sChannel::new(Blake2sHash::default());
        let log_query_size = 30;
        let log_projected_query_size = 27;
        let n_queries = 100;
        let projected_query_coset = CanonicCoset::new(log_projected_query_size);

        let queries = generate_queries(channel, log_query_size, n_queries);
        let projected_queries =
            get_projected_queries(&queries, log_query_size - log_projected_query_size);

        assert_eq!(projected_queries.len(), n_queries);
        for (query, conjugate_query) in projected_queries.into_iter() {
            assert!(query < (1 << log_query_size));
            assert!(conjugate_query < (1 << log_query_size));
            assert_eq!(
                projected_query_coset.at(query),
                -projected_query_coset.at(conjugate_query)
            );
        }
    }
}
