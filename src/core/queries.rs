use std::collections::BTreeSet;

use super::channel::{Blake2sChannel, Channel};

pub const UPPER_BOUND_QUERY_BYTES: usize = 4;

/// Randomizes a set of query indices uniformly over the range [0, 2^`log_query_size`).
pub fn generate_queries(
    channel: &mut Blake2sChannel,
    log_query_size: u32,
    n_queries: usize,
) -> BTreeSet<usize> {
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
                return queries;
            }
        }
    }
}

/// Calculates the matching query indices in a smaller domain given the (log) domain ratio.
pub fn get_projected_queries(queries: &BTreeSet<usize>, log_domain_ratio: u32) -> BTreeSet<usize> {
    queries.iter().map(|q| q >> log_domain_ratio).collect()
}

#[cfg(test)]
mod tests {
    use super::generate_queries;
    use crate::commitment_scheme::blake2_hash::Blake2sHash;
    use crate::core::channel::{Blake2sChannel, Channel};

    #[test]
    fn test_generate_queries() {
        let channel = &mut Blake2sChannel::new(Blake2sHash::default());
        let log_query_size = 31;
        let n_queries = 100;

        let queries = generate_queries(channel, log_query_size, n_queries);

        assert!(queries.len() == n_queries);
        for query in queries {
            assert!(query < 1 << log_query_size);
        }
    }
}
