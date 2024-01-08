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
