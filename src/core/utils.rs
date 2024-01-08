pub fn bit_reverse_index(x: u32, log_domain_size: u32) -> u32 {
    x.reverse_bits() >> (32 - log_domain_size)
}

// TODO(AlonH): Consider benchmarking this function.
/// A naive (not cache friendly) implementation of bit-reversal permutation.
pub fn bit_reverse_in_place<T: Copy>(array: &mut [T], log_domain_size: u32) {
    assert!(array.len() == (1 << log_domain_size));
    for i in 0..array.len() {
        let j = bit_reverse_index(i as u32, log_domain_size) as usize;
        if i < j {
            array.swap(i, j);
        }
    }
}

pub fn bit_reverse_vec<T: Copy>(vec: &Vec<T>, log_domain_size: u32) -> Vec<T> {
    let mut result = vec.to_owned();
    bit_reverse_in_place(&mut result, log_domain_size);
    result
}

#[cfg(test)]
mod tests {
    #[test]
    fn test_bit_reverse_vec() {
        let vec = vec![0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15];
        let log_domain_size = 4;
        let expected = vec![0, 8, 4, 12, 2, 10, 6, 14, 1, 9, 5, 13, 3, 11, 7, 15];
        let actual = super::bit_reverse_vec(&vec, log_domain_size);
        assert_eq!(actual, expected);
    }
}
