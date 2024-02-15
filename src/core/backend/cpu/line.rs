use super::CPUBackend;
use crate::core::circle::CirclePoint;
use crate::core::fft::{butterfly, ibutterfly};
use crate::core::fields::m31::BaseField;
use crate::core::fields::{ExtensionOf, Field};
use crate::core::poly::line::{LineDomain, LineEvaluation, LinePoly, LinePolyOps};
use crate::core::poly::utils::{fold, repeat_value};

impl<F: ExtensionOf<BaseField>> LinePolyOps<F> for CPUBackend {
    fn eval_at_point<E: ExtensionOf<F>>(poly: &LinePoly<Self, F>, mut x: E) -> E {
        // TODO(Andrew): Allocation here expensive for small polynomials.
        let mut doublings = vec![x];
        for _ in 1..poly.log_size {
            x = CirclePoint::double_x(x);
            doublings.push(x);
        }
        fold(&poly.coeffs, &doublings)
    }

    fn evaluate(poly: LinePoly<Self, F>, domain: LineDomain) -> LineEvaluation<Self, F> {
        assert!(domain.size() >= poly.coeffs.len());

        // The first few FFT layers may just copy coefficients so we do it directly.
        // See the docs for `n_skipped_layers` in [line_fft].
        let log_degree_bound = poly.log_size;
        let n_skipped_layers = (domain.log_size() - log_degree_bound) as usize;
        let duplicity = 1 << n_skipped_layers;
        let mut coeffs = repeat_value(&poly.coeffs, duplicity);

        line_fft(&mut coeffs, domain, n_skipped_layers);
        LineEvaluation::new(domain, coeffs)
    }

    fn interpolate(eval: LineEvaluation<Self, F>) -> LinePoly<Self, F> {
        let domain = eval.domain();
        let mut values = eval.values;
        line_ifft(&mut values, domain);
        // Normalize the coefficients.
        let len_inv = BaseField::from(values.len()).inverse();
        values.iter_mut().for_each(|v| *v *= len_inv);
        LinePoly::new(values)
    }
}

/// Performs a univariate FFT of a polynomial over a [LineDomain].
///
/// The transform happens in-place. `values` consist of coefficients in [line_ifft] algorithm's
/// basis need to be stored in bit-reversed order. After the transformation `values` becomes
/// evaluations of the polynomial over `domain` stored in natural order.
///
/// The `n_skipped_layers` argument allows specifying how many of the initial butterfly layers of
/// the FFT to skip. This is useful when doing more efficient degree aware FFTs as the butterflies
/// in the first layers of the FFT only involve copying coefficients to different locations (because
/// one or more of the coefficients is zero). This new algorithm is `O(n log d)` vs `O(n log n)`
/// where `n` is the domain size and `d` is the number of coefficients.
///
/// # Panics
///
/// Panics if the number of values doesn't match the size of the domain.
fn line_fft<F: ExtensionOf<BaseField>>(
    values: &mut [F],
    mut domain: LineDomain,
    n_skipped_layers: usize,
) {
    assert_eq!(values.len(), domain.size());

    // Construct the domains we need.
    let mut domains = vec![];
    while domain.size() > 1 << n_skipped_layers {
        domains.push(domain);
        domain = domain.double();
    }

    // Execute the butterfly layers.
    for domain in domains.iter().rev() {
        for chunk in values.chunks_exact_mut(domain.size()) {
            let (l, r) = chunk.split_at_mut(domain.size() / 2);
            for (i, x) in domain.iter().take(domain.size() / 2).enumerate() {
                butterfly(&mut l[i], &mut r[i], x);
            }
        }
    }
}

/// Performs a univariate IFFT on a polynomial's evaluation over a [LineDomain].
///
/// This is not the standard univariate IFFT, because [LineDomain] is not a cyclic group.
///
/// The transform happens in-place. `values` should be the evaluations of a polynomial over `domain`
/// in their natural order. After the transformation `values` becomes the coefficients of the
/// polynomial stored in bit-reversed order.
///
/// For performance reasons and flexibility the normalization of the coefficients is omitted. The
/// normalized coefficients can be obtained by scaling all coefficients by `1 / len(values)`.
///
/// This algorithm does not return coefficients in the standard monomial basis but rather returns
/// coefficients in a basis relating to the circle's x-coordinate doubling map `pi(x) = 2x^2 - 1`
/// i.e.
///
/// ```text
/// B = { 1 } * { x } * { pi(x) } * { pi(pi(x)) } * ...
///   = { 1, x, pi(x), pi(x) * x, pi(pi(x)), pi(pi(x)) * x, pi(pi(x)) * pi(x), ... }
/// ```
///
/// # Panics
///
/// Panics if the number of values doesn't match the size of the domain.
fn line_ifft<F: ExtensionOf<BaseField>>(values: &mut [F], mut domain: LineDomain) {
    assert_eq!(values.len(), domain.size());
    while domain.size() > 1 {
        for chunk in values.chunks_exact_mut(domain.size()) {
            let (l, r) = chunk.split_at_mut(domain.size() / 2);
            for (i, x) in domain.iter().take(domain.size() / 2).enumerate() {
                ibutterfly(&mut l[i], &mut r[i], x.inverse());
            }
        }
        domain = domain.double();
    }
}
