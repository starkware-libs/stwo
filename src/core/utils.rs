pub fn bit_reverse_index(x: u32, log_domain_size: u32) -> u32 {
    x.reverse_bits() >> (32 - log_domain_size)
}

// TODO(AlonH): Consider benchmarking this function.
pub fn bit_reverse_vec<T: Copy>(vec: &Vec<T>, log_domain_size: u32) -> Vec<T> {
    assert!(vec.len() == (1 << log_domain_size));
    let mut result = Vec::with_capacity(vec.len());
    for i in 0..vec.len() {
        result.push(vec[bit_reverse_index(i as u32, log_domain_size) as usize]);
    }
    result
}
