use std::collections::BTreeSet;

use super::channel::{Blake2sChannel, Channel};

pub fn generate_queries(
    channel: &mut Blake2sChannel,
    log_query_size: usize,
    n_queries: usize,
) -> BTreeSet<usize> {
    let mut quotient_queries = BTreeSet::new();
    for _ in 0..n_queries {
        let query_bits = u32::from_le_bytes(channel.draw_random_bytes()[..4].try_into().unwrap());
        let quotient_query = query_bits & ((1 << log_query_size) - 1);
        quotient_queries.insert(quotient_query as usize);
    }
    quotient_queries
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
