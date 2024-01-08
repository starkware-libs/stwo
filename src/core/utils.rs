pub fn bit_reverse_index(x: u32, log_domain_size: u32) -> u32 {
    x.reverse_bits() >> (32 - log_domain_size)
}

pub fn bit_reverse_vec<T: Copy>(vec: &Vec<T>, log_domain_size: u32) -> Vec<T> {
    assert!(vec.len() == (1 << log_domain_size));
    let mut result = vec.clone();
    for i in 0..vec.len() {
        result[bit_reverse_index(i as u32, log_domain_size) as usize] = vec[i];
    }
    result
}

#[cfg(test)]
mod tests {
    use num_traits::Pow;

    use crate::core::utils::bit_reverse_index;

    #[test]
    fn test_bit_reverse_index() {
        fn verify_bit_reversed(x: u32, y: u32, log_domain_size: u32) {
            for i in 0..log_domain_size {
                let x_bit = (x >> i) & 1;
                let y_bit = (y >> (log_domain_size - i - 1)) & 1;
                assert!(x_bit == y_bit);
            }
        }

        let log_domain_size = 7;
        for i in 0..2.pow(log_domain_size) {
            let x = i as u32;
            let y = bit_reverse_index(x, log_domain_size);
            verify_bit_reversed(x, y, log_domain_size);
        }
    }
}
