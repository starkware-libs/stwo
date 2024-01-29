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

/// Returns the next chunk of the iterator and advances it by `chunk_size`.
///
/// # Panics
///
/// Panics if there are less then `chunk_size` elements in the iterator.
pub(crate) fn next_chunk<T>(iter: &mut impl Iterator<Item = T>, chunk_size: usize) -> Vec<T> {
    let mut vec = Vec::with_capacity(chunk_size);
    for _ in 0..chunk_size {
        vec.push(
            iter.next()
                .unwrap_or_else(|| panic!("Not enough elements in iterator.")),
        );
    }
    vec
}

#[cfg(test)]
mod tests {
    use crate::core::utils::{bit_reverse, next_chunk};

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

    #[test]
    fn test_next_chunk() {
        let mut iter = 0..10;
        assert_eq!(next_chunk(&mut iter, 3), vec![0, 1, 2]);
        assert_eq!(next_chunk(&mut iter, 3), vec![3, 4, 5]);
        assert_eq!(next_chunk(&mut iter, 3), vec![6, 7, 8]);
        assert_eq!(next_chunk(&mut iter, 1), vec![9]);
    }

    #[test]
    #[should_panic]
    fn test_insufficient_elements_in_next_chunk() {
        let mut iter = 0..3;
        next_chunk(&mut iter, 4);
    }
}
