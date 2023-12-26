use std::collections::BTreeSet;

use super::channel::{Blake2sChannel, Channel};

pub const MAX_QUERY_BYTES: usize = 8;

pub fn generate_queries(
    channel: &mut Blake2sChannel,
    log_query_size: usize,
    n_queries: usize,
) -> BTreeSet<usize> {
    let mut queries = BTreeSet::new();
    let mut query_cnt = 0;
    loop {
        let random_bytes = channel.draw_random_bytes();
        for i in 0..MAX_QUERY_BYTES {
            let query_bits =
                u32::from_le_bytes(random_bytes[4 * i..4 * (i + 1)].try_into().unwrap());
            let quotient_query = query_bits & ((1 << log_query_size) - 1);
            queries.insert(quotient_query as usize);
            query_cnt += 1;
            if query_cnt == n_queries {
                return queries;
            }
        }
    }
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
