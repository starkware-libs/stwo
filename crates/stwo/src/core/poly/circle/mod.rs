mod canonic;
mod domain;

pub use canonic::CanonicCoset;
pub use domain::{CircleDomain, MAX_CIRCLE_DOMAIN_LOG_SIZE};

#[cfg(test)]
mod tests {
    use super::CanonicCoset;
    use crate::core::utils::bit_reverse_index;

    #[test]
    fn is_canonic_valid_domain() {
        let canonic_domain = CanonicCoset::new(4).circle_domain();

        assert!(canonic_domain.is_canonic());
    }

    #[test]
    pub fn test_bit_reverse_indices() {
        let log_domain_size = 7;
        let log_small_domain_size = 5;
        let domain = CanonicCoset::new(log_domain_size);
        let small_domain = CanonicCoset::new(log_small_domain_size);
        let n_folds = log_domain_size - log_small_domain_size;
        for i in 0..2usize.pow(log_domain_size) {
            let point = domain.at(bit_reverse_index(i, log_domain_size));
            let small_point = small_domain.at(bit_reverse_index(
                i / 2usize.pow(n_folds),
                log_small_domain_size,
            ));
            assert_eq!(point.repeated_double(n_folds), small_point);
        }
    }
}
