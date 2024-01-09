use std::collections::BTreeSet;

use super::channel::{Blake2sChannel, Channel};

pub const UPPER_BOUND_QUERY_BYTES: usize = 4;

pub fn generate_queries(
    channel: &mut Blake2sChannel,
    log_query_size: u32,
    n_queries: usize,
) -> BTreeSet<usize> {
    let mut queries = BTreeSet::new();
    let mut query_cnt = 0;
    loop {
        let random_bytes = channel.draw_random_bytes();
        for chunk in random_bytes.chunks_exact(UPPER_BOUND_QUERY_BYTES) {
            let query_bits = u32::from_le_bytes(chunk.try_into().unwrap());
            let quotient_query = query_bits & ((1 << log_query_size) - 1);
            queries.insert(quotient_query as usize);
            query_cnt += 1;
            if query_cnt == n_queries {
                return queries;
            }
        }
    }
}

/// Calculates the locations of the queries in the trace commitment domain.
pub fn get_projected_queries(
    quotient_queries: &BTreeSet<usize>,
    log_trace_domain_size: u32,
    log_quotient_domain_size: u32,
) -> BTreeSet<usize> {
    let domain_ratio = 1 << (log_quotient_domain_size - log_trace_domain_size);
    quotient_queries.iter().map(|q| q / domain_ratio).collect()
}

/// Calculates the matching query indices in a domain that is split into two parts (half a coset and
/// it's conjugate) given the queries of the original domain and its size.
pub fn get_split_queries(queries: &BTreeSet<usize>, domain_size: usize) -> BTreeSet<usize> {
    fn get_split_query(i: usize, domain_size: usize) -> usize {
        let folded_domain_size = domain_size / 2;
        if i < folded_domain_size {
            i
        } else {
            domain_size - i - 1
        }
    }
    queries
        .iter()
        .map(|q| get_split_query(*q, domain_size))
        .collect()
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
