#[inline]
pub fn log2_ceil(n: usize) -> usize {
    assert_ne!(n, 0, "Attempt log(0)!");
    const NUM_OF_BITS_IN_BYTE: usize = 8;
    let num_of_bits = std::mem::size_of_val(&n) * NUM_OF_BITS_IN_BYTE;
    num_of_bits - (n - 1).leading_zeros() as usize
}

/// Get the ceiling value of an unsigned integer division.
// TODO(Ohad): Consider removing assertion.
#[inline]
pub fn usize_div_ceil(a: usize, b: usize) -> usize {
    assert_ne!(b, 0, "Attempt division by 0!");
    (a + b - 1) / b
}

#[cfg(test)]
mod tests {
    use crate::math::log2_ceil;
    use crate::math::usize_div_ceil;

    #[test]
    fn log2_test() {
        assert_eq!(log2_ceil(1), 0);
        assert_eq!(log2_ceil(64), 6);
    }

    #[test]
    fn div_ceil_test() {
        assert_eq!(usize_div_ceil(6, 4), 2);
        assert_eq!(usize_div_ceil(6, 3), 2);
    }

    #[test]
    #[should_panic(expected = "Attempt division by 0!")]
    fn div_ceil_by_zero_test() {
        usize_div_ceil(6, 0);
    }
}
