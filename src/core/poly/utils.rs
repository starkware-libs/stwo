use std::ops::{Add, Mul};

/// Folds values recursively in `O(n)` by a hierarchical application of folding factors.
///
/// i.e. folding `n = 8` values with `folding_factors = [α, β, γ]`:
///
/// ```text
///               n2=n1+α*n2
///           /               \
///     n1=n3+β*n4          n2=n5+β*n6
///      /      \            /      \
/// n3=a+γ*b  n4=c+γ*d  n5=e+γ*f  n6=g+γ*h
///   /  \      /  \      /  \      /  \
///  a    b    c    d    e    f    g    h
/// ```
///
/// # Panics
///
/// Panics if the number of values is not a power of two.
// TODO(Andrew): Can be made to run >10x faster by unrolling lower layers of recursion.
pub(crate) fn fold<T, U>(values: &[T], folding_factors: &[U]) -> T
where
    T: Copy + Add<Output = T> + Mul<U, Output = T>,
    U: Copy,
{
    let n = values.len();
    assert!(n.is_power_of_two());
    if n == 1 {
        return values[0];
    }
    let (lhs_values, rhs_values) = values.split_at(n / 2);
    let (&folding_factor, folding_factors) = folding_factors.split_first().unwrap();
    let lhs_val = fold(lhs_values, folding_factors);
    let rhs_val = fold(rhs_values, folding_factors);
    lhs_val + rhs_val * folding_factor
}
