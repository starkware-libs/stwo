use std::cmp::Ordering;
use std::fmt::Debug;
use std::ops::Deref;

use num_traits::Zero;

use super::utils::repeat_value;
use crate::core::circle::Coset;
use crate::core::fft::{butterfly, ibutterfly};
use crate::core::fields::m31::BaseField;
use crate::core::fields::{ExtensionOf, Field};

/// Domain comprising of the x-coordinates of points in a [Coset].
///
/// For use with univariate polynomials.
#[derive(Copy, Clone, Debug)]
pub struct LineDomain {
    coset: Coset,
}

impl LineDomain {
    /// Returns a domain comprising of the x-coordinates of points in a coset.
    ///
    /// # Panics
    ///
    /// Panics if the coset items don't have unique x-coordinates.
    pub fn new(coset: Coset) -> Self {
        match coset.size().cmp(&2) {
            Ordering::Less => {}
            Ordering::Equal => {
                // If the coset with two points contains (0, y) then the coset is {(0, y), (0, -y)}.
                assert!(!coset.initial.x.is_zero(), "coset x-coordinates not unique");
            }
            Ordering::Greater => {
                // Let our coset be `E = c + <G>` with `|E| > 2` then:
                // 1. if `ord(c) <= ord(G)` the coset contains two points at x=0
                // 2. if `ord(c) = 2 * ord(G)` then `c` and `-c` are in our coset
                assert!(
                    coset.initial.order_bits() >= coset.step.order_bits() + 2,
                    "coset x-coordinates not unique"
                );
            }
        }
        Self { coset }
    }

    /// Returns the `i`th domain element.
    pub fn at(&self, i: usize) -> BaseField {
        self.coset.at(i).x
    }

    /// Returns the size of the domain.
    pub fn size(&self) -> usize {
        self.coset.size()
    }

    /// Returns an iterator over elements in the domain.
    pub fn iter(&self) -> impl Iterator<Item = BaseField> {
        self.coset.iter().map(|p| p.x)
    }

    /// Returns a new domain comprising of all points in current domain doubled.
    pub fn double(&self) -> Self {
        Self {
            coset: self.coset.double(),
        }
    }

    /// Returns the domain's underlying coset.
    pub fn coset(&self) -> Coset {
        self.coset
    }
}

/// A univariate polynomial defined on a [LineDomain].
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub struct LinePoly<F> {
    /// Coefficients of the polynomial in [line_ifft] algorithm's basis.
    ///
    /// The coefficients are stored in bit-reversed order.
    coeffs: Vec<F>,
    /// The number of coefficients stored as `log2(len(coeffs))`.
    n_bits: u32,
}

impl<F: ExtensionOf<BaseField>> LinePoly<F> {
    /// Creates a new line polynomial from bit reversed coefficients.
    ///
    /// # Panics
    ///
    /// Panics if the number of coefficients is not a power of two.
    pub fn new(coeffs: Vec<F>) -> Self {
        assert!(coeffs.len().is_power_of_two());
        let n_bits = coeffs.len().ilog2();
        Self { coeffs, n_bits }
    }

    /// Evaluates the polynomial at a single point.
    pub fn eval_at_point(&self, _x: F) -> F {
        todo!()
    }

    /// Evaluates the polynomial at all points in the domain.
    pub fn evaluate(mut self, domain: LineDomain) -> LineEvaluation<F> {
        assert!(domain.size() >= self.coeffs.len());

        // The first few FFT layers may just copy coefficients so we do it directly.
        // See the docs for `n_skipped_layers` in [line_fft].
        let domain_bits = domain.size().ilog2();
        let degree_bound_bits = self.coeffs.len().ilog2();
        let n_skipped_layers = (domain_bits - degree_bound_bits) as usize;
        let duplicity = 1 << n_skipped_layers;
        self.coeffs = repeat_value(&self.coeffs, duplicity);

        line_fft(&mut self.coeffs, domain, n_skipped_layers);
        LineEvaluation::new(self.coeffs)
    }

    /// Returns the number of coefficients.
    #[allow(clippy::len_without_is_empty)]
    pub fn len(&self) -> usize {
        debug_assert_eq!(self.coeffs.len(), 1 << self.n_bits);
        1 << self.n_bits
    }
}

impl<F: ExtensionOf<BaseField>> Deref for LinePoly<F> {
    type Target = Vec<F>;

    fn deref(&self) -> &Vec<F> {
        &self.coeffs
    }
}

/// Evaluations of a univariate polynomial on a [LineDomain].
pub struct LineEvaluation<F> {
    /// Evaluations of a univariate polynomial on a [LineDomain].
    evals: Vec<F>,
    /// The number of evaluations stored as `log2(len(evals))`.
    n_bits: u32,
}

impl<F: ExtensionOf<BaseField>> LineEvaluation<F> {
    /// Creates new [LineEvaluation] from a set of polynomial evaluations over a [LineDomain].
    ///
    /// # Panics
    ///
    /// Panics if the number of evaluations is not a power of two.
    pub fn new(evals: Vec<F>) -> Self {
        assert!(evals.len().is_power_of_two());
        let n_bits = evals.len().ilog2();
        Self { evals, n_bits }
    }

    /// Interpolates the polynomial as evaluations on `domain`.
    pub fn interpolate(mut self, domain: LineDomain) -> LinePoly<F> {
        line_ifft(&mut self.evals, domain);
        // Normalize the coefficients.
        let len_inv = BaseField::from(self.evals.len()).inverse();
        self.evals.iter_mut().for_each(|v| *v *= len_inv);
        LinePoly::new(self.evals)
    }

    /// Returns the number of evaluations.
    #[allow(clippy::len_without_is_empty)]
    pub fn len(&self) -> usize {
        debug_assert_eq!(self.evals.len(), 1 << self.n_bits);
        1 << self.n_bits
    }
}

impl<F: ExtensionOf<BaseField>> Deref for LineEvaluation<F> {
    type Target = Vec<F>;

    fn deref(&self) -> &Vec<F> {
        &self.evals
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

#[cfg(test)]
mod tests {
    use super::LineDomain;
    use crate::core::circle::{CirclePoint, Coset};
    use crate::core::fields::m31::BaseField;
    use crate::core::poly::line::{LineEvaluation, LinePoly};

    #[test]
    #[should_panic]
    fn bad_line_domain() {
        // This coset doesn't have points with unique x-coordinates.
        let coset = Coset::odds(2);

        LineDomain::new(coset);
    }

    #[test]
    fn line_domain_of_size_two_works() {
        const LOG_N: usize = 1;
        let coset = Coset::subgroup(LOG_N);

        LineDomain::new(coset);
    }

    #[test]
    fn line_domain_of_size_one_works() {
        const LOG_N: usize = 0;
        let coset = Coset::subgroup(LOG_N);

        LineDomain::new(coset);
    }

    #[test]
    fn line_domain_size_is_correct() {
        const LOG_N: usize = 8;
        let coset = Coset::half_odds(LOG_N);
        let domain = LineDomain::new(coset);

        let size = domain.size();

        assert_eq!(size, 1 << LOG_N);
    }

    #[test]
    fn line_domain_coset_returns_the_coset() {
        let coset = Coset::half_odds(5);
        let domain = LineDomain::new(coset);

        assert_eq!(domain.coset(), coset);
    }

    #[test]
    fn line_domain_double_works() {
        const LOG_N: usize = 8;
        let coset = Coset::half_odds(LOG_N);
        let domain = LineDomain::new(coset);

        let doubled_domain = domain.double();

        assert_eq!(doubled_domain.size(), 1 << (LOG_N - 1));
        assert_eq!(doubled_domain.at(0), CirclePoint::double_x(domain.at(0)));
        assert_eq!(doubled_domain.at(1), CirclePoint::double_x(domain.at(1)));
    }

    #[test]
    fn line_domain_iter_works() {
        const LOG_N: usize = 8;
        let coset = Coset::half_odds(LOG_N);
        let domain = LineDomain::new(coset);

        let elements = domain.iter().collect::<Vec<BaseField>>();

        assert_eq!(elements.len(), domain.size());
        for (i, element) in elements.into_iter().enumerate() {
            assert_eq!(element, domain.at(i), "mismatch at {i}");
        }
    }

    #[test]
    fn line_polynomial_evaluation() {
        let poly = LinePoly::new(vec![
            BaseField::from(7), // 7 * 1
            BaseField::from(9), // 9 * pi(x)
            BaseField::from(5), // 5 * x
            BaseField::from(3), // 3 * pi(x)*x
        ]);
        let coset = Coset::half_odds(poly.len().ilog2() as usize);
        let domain = LineDomain::new(coset);
        let expected_evals = domain
            .iter()
            .map(|x| {
                let pi_x = CirclePoint::double_x(x);
                poly[0] + poly[1] * pi_x + poly[2] * x + poly[3] * pi_x * x
            })
            .collect::<Vec<BaseField>>();

        let actual_evals = poly.evaluate(domain);

        assert_eq!(*actual_evals, expected_evals);
    }

    #[test]
    fn line_polynomial_evaluation_on_larger_domain() {
        let poly = LinePoly::new(vec![
            BaseField::from(7), // 7 * 1
            BaseField::from(9), // 9 * pi(x)
            BaseField::from(5), // 5 * x
            BaseField::from(3), // 3 * pi(x)*x
        ]);
        let coset = Coset::half_odds(4 + poly.len().ilog2() as usize);
        let domain = LineDomain::new(coset);
        let expected_evals = domain
            .iter()
            .map(|x| {
                let pi_x = CirclePoint::double_x(x);
                poly[0] + poly[1] * pi_x + poly[2] * x + poly[3] * pi_x * x
            })
            .collect::<Vec<BaseField>>();

        let actual_evals = poly.evaluate(domain);

        assert_eq!(*actual_evals, expected_evals);
    }

    #[test]
    fn line_evaluation_interpolation() {
        let poly = LinePoly::new(vec![
            BaseField::from(7), // 7 * 1
            BaseField::from(9), // 9 * pi(x)
            BaseField::from(5), // 5 * x
            BaseField::from(3), // 3 * pi(x)*x
        ]);
        let coset = Coset::half_odds(poly.len().ilog2() as usize);
        let domain = LineDomain::new(coset);
        let evals = LineEvaluation::new(
            domain
                .iter()
                .map(|x| {
                    let pi_x = CirclePoint::double_x(x);
                    poly[0] + poly[1] * pi_x + poly[2] * x + poly[3] * pi_x * x
                })
                .collect::<Vec<BaseField>>(),
        );

        let interpolated_poly = evals.interpolate(domain);

        assert_eq!(interpolated_poly, poly);
    }
}
