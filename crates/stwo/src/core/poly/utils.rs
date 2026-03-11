use std_shims::Vec;

use super::line::LineDomain;
use crate::core::circle::CirclePoint;
use crate::core::fields::m31::BaseField;
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
    let (folding_factor, folding_factors) = folding_factors.split_first().unwrap();
    let lhs_val = fold(lhs_values, folding_factors);
    let rhs_val = fold(rhs_values, folding_factors);
    lhs_val + rhs_val * *folding_factor
}

/// Computes the folding alphas for evaluation by folding.
/// The folding alphas are the basis for the `len` dimensional FFT-basis:
/// y, x, pi(x), pi^2(x), ..., pi^{len-2}(x).
/// Returns the folding alphas in reverse order.
#[allow(clippy::uninit_vec)]
pub fn get_folding_alphas<F: Field + ExtensionOf<BaseField>>(
    point: CirclePoint<F>,
    len: usize,
) -> Vec<F> {
    if len == 0 {
        return Vec::new();
    }
    // Manually set the length of the vector so we can directly assign the elements instead of push
    // and then reverse.
    let mut alphas = Vec::with_capacity(len);
    unsafe {
        alphas.set_len(len);
    }

    // Fill the vector with the correct values.
    alphas[len - 1] = point.y;
    if len > 1 {
        let mut x = point.x;
        for i in (0..len - 1).rev() {
            alphas[i] = x;
            x = CirclePoint::double_x(x);
        }
    }

    alphas
}

/// Repeats each value sequentially `duplicity` many times.
///
/// # Examples
///
/// ```rust
/// # use stwo::core::poly::utils::repeat_value;
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
    let domain = domain.into();
    assert!(
        domain.coset().size() <= twiddle_buffer.len(),
        "Not enough twiddles!"
    );
    (0..domain.coset().log_size())
        .map(|i| {
            let len = 1 << i;
            &twiddle_buffer[twiddle_buffer.len() - len * 2..twiddle_buffer.len() - len]
        })
        .rev()
        .collect()
}

/// Extracts twiddles for a subdomain from a larger twiddle buffer.
///
/// The subdomain is obtained by splitting the domain corresponding to
/// `committed_half_log_size` some number of times. In bit-reversed order, the subdomain's
/// twiddles at each FFT layer are the first portion of the corresponding committed domain's
/// twiddle layer.
///
/// The returned buffer has the same layout as a canonical twiddle buffer of the subdomain's size,
/// so it can be used with [`domain_line_twiddles_from_tree`].
pub fn repack_subdomain_twiddles<T: Copy>(
    subdomain_half_log_size: u32,
    committed_half_log_size: u32,
    twiddle_buffer: &[T],
) -> Vec<T> {
    let root_half_log_size = twiddle_buffer.len().ilog2();
    assert!(
        subdomain_half_log_size <= committed_half_log_size
            && committed_half_log_size <= root_half_log_size,
        "Invalid sizes: subdomain={subdomain_half_log_size}, committed={committed_half_log_size}, \
         root={root_half_log_size}"
    );

    // First, locate the committed domain's layers within the root buffer.
    // The committed domain's outermost layer (layer 0) starts at the offset where the root
    // buffer's layer for coset log_size `committed_half_log_size` begins.
    // Root buffer layout: [layer for root (size 2^{K-1}) | layer after 1 double (size 2^{K-2}) |
    // ...] The committed layer j corresponds to root layer (K - C + j), where K =
    // root_half_log_size, C = committed_half_log_size.
    let skip_layers = root_half_log_size - committed_half_log_size;

    // Compute offset to the first committed layer within the root buffer.
    // Skip `skip_layers` root layers: sizes 2^{K-1}, 2^{K-2}, ..., 2^{K-skip_layers}.
    // Total skip = 2^K - 2^{K-skip_layers}.
    let committed_start = if skip_layers == 0 {
        0
    } else {
        (1usize << root_half_log_size) - (1usize << (root_half_log_size - skip_layers))
    };

    // Output buffer: 2^{L-1} + 2^{L-2} + ... + 1 + 1 (padding) = 2^L.
    let out_size = 1usize << subdomain_half_log_size;
    let mut result = Vec::with_capacity(out_size);
    let mut committed_offset = committed_start;
    let mut committed_layer_size = 1usize << (committed_half_log_size - 1);
    for _ in 0..subdomain_half_log_size {
        let subdomain_layer_size =
            committed_layer_size >> (committed_half_log_size - subdomain_half_log_size);
        result.extend_from_slice(
            &twiddle_buffer[committed_offset..committed_offset + subdomain_layer_size],
        );
        committed_offset += committed_layer_size;
        committed_layer_size /= 2;
    }
    // Padding to make length a power of 2 (copy from committed buffer's padding).
    result.push(twiddle_buffer[twiddle_buffer.len() - 1]);
    debug_assert_eq!(result.len(), out_size);
    result
}

#[cfg(test)]
mod tests {
    use std_shims::vec;

    use super::repeat_value;
    use crate::core::poly::circle::CanonicCoset;
    use crate::core::poly::line::LineDomain;
    use crate::core::poly::utils::domain_line_twiddles_from_tree;

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

    #[test]
    fn test_domain_line_twiddles_works() {
        let domain: LineDomain = CanonicCoset::new(4).circle_domain().into();
        let twiddles = domain_line_twiddles_from_tree(domain, &[0, 1, 2, 3, 4, 5, 6, 7]);
        assert_eq!(twiddles.len(), 3);
        assert_eq!(twiddles[0], &[0, 1, 2, 3]);
        assert_eq!(twiddles[1], &[4, 5]);
        assert_eq!(twiddles[2], &[6]);
    }

    #[test]
    #[should_panic]
    fn test_domain_line_twiddles_fails() {
        let domain: LineDomain = CanonicCoset::new(5).circle_domain().into();
        domain_line_twiddles_from_tree(domain, &[0, 1, 2, 3, 4, 5, 6, 7]);
    }
}
