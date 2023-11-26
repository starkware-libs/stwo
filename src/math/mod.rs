#[inline]
pub fn log2_ceil(n: usize) -> usize {
    assert_ne!(n, 0, "Attempt log(0)!");
    const NUM_OF_BITS_IN_BYTE: usize = 8;
    let num_of_bits = std::mem::size_of_val(&n) * NUM_OF_BITS_IN_BYTE;
    num_of_bits - (n - 1).leading_zeros() as usize
}

#[inline]
pub fn log2_floor(n: usize) -> usize {
    assert_ne!(n, 0, "Attempt log(0)!");
    const NUM_OF_BITS_IN_BYTE: usize = 8;
    let num_of_bits = std::mem::size_of_val(&n) * NUM_OF_BITS_IN_BYTE;
    num_of_bits - n.leading_zeros() as usize - 1
}

#[inline]
pub fn next_pow_two(n: usize) -> usize {
    2_usize.pow(log2_ceil(n) as u32)
}

#[inline]
pub fn prev_pow_two(n: usize) -> usize {
    2_usize.pow(log2_floor(n) as u32)
}

/// Get the ceiling value of an unsigned integer division.
// TODO(Ohad): Consider removing assertion.
#[inline]
pub fn usize_div_ceil(a: usize, b: usize) -> usize {
    assert_ne!(b, 0, "Attempt division by 0!");
    (a + b - 1) / b
}

#[inline]
pub fn usize_safe_div(a: usize, b: usize) -> usize {
    assert_ne!(b, 0, "Attempt division by 0!");
    let quotient = a / b;
    assert_eq!(a, quotient * b, "Attempt division with remainder!");
    quotient
}

/// Returns s, t, g such that g = gcd(x,y),  sx + ty = g.
pub fn egcd(x: i128, y: i128) -> (i128, i128, i128) {
    if x == 0 {
        return (0, 1, y);
    }
    let k = y / x;
    let (s, t, g) = egcd(y % x, x);
    (t - s * k, s, g)
}

#[cfg(test)]
mod tests {
    use crate::math::{egcd, log2_ceil, log2_floor, next_pow_two, prev_pow_two, usize_div_ceil};

    #[test]
    fn log2_ceil_test() {
        assert_eq!(log2_ceil(1), 0);
        assert_eq!(log2_ceil(63), 6);
        assert_eq!(log2_ceil(64), 6);
        assert_eq!(log2_ceil(65), 7);
    }

    #[test]
    fn log2_floor_test() {
        assert_eq!(log2_floor(1), 0);
        assert_eq!(log2_floor(63), 5);
        assert_eq!(log2_floor(64), 6);
        assert_eq!(log2_floor(65), 6);
    }

    #[test]
    fn next_power_of_two_test() {
        assert_eq!(next_pow_two(1), 1);
        assert_eq!(next_pow_two(2), 2);
        assert_eq!(next_pow_two(3), 4);
    }

    #[test]
    fn prev_power_of_two_test() {
        assert_eq!(prev_pow_two(1), 1);
        assert_eq!(prev_pow_two(2), 2);
        assert_eq!(prev_pow_two(3), 2);
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

    #[test]
    fn test_egcd() {
        let pairs = [(17, 5, 1), (6, 4, 2), (7, 7, 7)];
        for (x, y, res) in pairs.into_iter() {
            let (a, b, gcd) = egcd(x, y);
            assert_eq!(gcd, res);
            assert_eq!(a * x + b * y, gcd);
        }
    }
}
