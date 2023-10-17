pub fn log2_ceil(n: usize) -> usize {
    let num_of_bits = std::mem::size_of_val(&n) * 8;
    num_of_bits - (n - 1).leading_zeros() as usize
}

pub fn usize_div_ceil(a: usize, b: usize) -> usize {
    (a + b - 1) / b
}

#[cfg(test)]
mod tests {
    use crate::math::log2_ceil;
    use crate::math::usize_div_ceil;

    #[test]
    fn log2_test() {
        assert_eq!(log2_ceil(32), 5);
        assert_eq!(log2_ceil(64), 6);
    }

    #[test]
    fn div_ceil_test() {
        assert_eq!(usize_div_ceil(6, 4), 2);
        assert_eq!(usize_div_ceil(6, 3), 2);
    }
}
