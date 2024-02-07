pub(crate) fn bit_reverse_index(i: usize, log_size: u32) -> usize {
    i.reverse_bits() >> (usize::BITS - log_size)
}

/// Performs a naive bit-reversal permutation.
///
/// # Panics
///
/// Panics if the length of the slice is not a power of two.
// TODO(AlonH): Consider benchmarking this function.
// TODO: Implement cache friendly implementation.
pub(crate) fn bit_reverse<T, U: AsMut<[T]>>(mut v: U) -> U {
    let n = v.as_mut().len();
    assert!(n.is_power_of_two());
    let log_n = n.ilog2();
    for i in 0..n {
        let j = bit_reverse_index(i, log_n);
        if j > i {
            v.as_mut().swap(i, j);
        }
    }
    v
}

#[cfg(test)]
mod tests {
    use crate::core::utils::bit_reverse;

    #[test]
    fn bit_reverse_works() {
        assert_eq!(
            bit_reverse([0, 1, 2, 3, 4, 5, 6, 7]),
            [0, 4, 2, 6, 1, 5, 3, 7]
        );
    }

    #[test]
    #[should_panic]
    fn bit_reverse_non_power_of_two_size_fails() {
        bit_reverse([0, 1, 2, 3, 4, 5]);
    }
}
