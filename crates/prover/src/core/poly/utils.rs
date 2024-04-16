use super::line::LineDomain;
use crate::core::fields::{ExtensionOf, Field};

/// Folds values recursively in `O(n)` by a hierarchical application of folding factors.
///
/// i.e. folding `n = 8` values with `folding_factors = [x, y, z]`:
///
/// ```text
///               n2=n1+x*n2
///           /               \
///     n1=n3+y*n4          n2=n5+y*n6
///      /      \            /      \
/// n3=a+z*b  n4=c+z*d  n5=e+z*f  n6=g+z*h
///   /  \      /  \      /  \      /  \
///  a    b    c    d    e    f    g    h
/// ```
///
/// # Panics
///
/// Panics if the number of values is not a power of two or if an incorrect number of of folding
/// factors is provided.
// TODO(Andrew): Can be made to run >10x faster by unrolling lower layers of recursion
pub fn fold<F: Field, E: ExtensionOf<F>>(values: &[F], folding_factors: &[E]) -> E {
    let n = values.len();
    assert_eq!(n, 1 << folding_factors.len());
    if n == 1 {
        return values[0].into();
    }
    let (lhs_values, rhs_values) = values.split_at(n / 2);
    let (&folding_factor, folding_factors) = folding_factors.split_first().unwrap();
    let lhs_val = fold(lhs_values, folding_factors);
    let rhs_val = fold(rhs_values, folding_factors);
    lhs_val + rhs_val * folding_factor
}

/// Repeats each value sequentially `duplicity` many times.
///
/// # Examples
///
/// ```rust,ignore
/// assert_eq!(repeat_value(&[1, 2, 3], 2), vec![1, 1, 2, 2, 3, 3]);
/// ```
pub fn repeat_value<T: Copy>(values: &[T], duplicity: usize) -> Vec<T> {
    let n = values.len();
    let mut res: Vec<T> = Vec::with_capacity(n * duplicity);

    // Fill each chunk with its corresponding value.
    for &v in values {
        for _ in 0..duplicity {
            res.push(v)
        }
    }

    res
}

/// Computes the line twiddles for a [`CircleDomain`] or a [`LineDomain`] from the precomputed
/// twiddles tree.
///
/// [`CircleDomain`]: super::circle::CircleDomain
pub fn domain_line_twiddles_from_tree<T>(
    domain: impl Into<LineDomain>,
    twiddle_buffer: &[T],
) -> Vec<&[T]> {
    (0..domain.into().coset().log_size())
        .map(|i| {
            let len = 1 << i;
            &twiddle_buffer[twiddle_buffer.len() - len * 2..twiddle_buffer.len() - len]
        })
        .rev()
        .collect()
}

#[cfg(test)]
mod tests {
    use super::repeat_value;

    #[test]
    fn repeat_value_0_times_works() {
        assert!(repeat_value(&[1, 2, 3], 0).is_empty());
    }

    #[test]
    fn repeat_value_2_times_works() {
        assert_eq!(repeat_value(&[1, 2, 3], 2), vec![1, 1, 2, 2, 3, 3]);
    }

    #[test]
    fn repeat_value_3_times_works() {
        assert_eq!(repeat_value(&[1, 2], 3), vec![1, 1, 1, 2, 2, 2]);
    }
}
