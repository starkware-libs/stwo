use std::cmp::Ordering;
use std::fmt::Debug;
use std::iter::Map;
use std::marker::PhantomData;

use num_traits::Zero;

use super::{BitReversedOrder, NaturalOrder};
use crate::core::backend::{Column, ColumnTrait, FieldOps};
use crate::core::circle::{CirclePoint, Coset, CosetIterator};
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
                    coset.initial.log_order() >= coset.step.log_order() + 2,
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

    /// Returns the log size of the domain.
    pub fn log_size(&self) -> u32 {
        self.coset.log_size()
    }

    /// Returns an iterator over elements in the domain.
    pub fn iter(&self) -> LineDomainIterator {
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

impl IntoIterator for LineDomain {
    type Item = BaseField;
    type IntoIter = LineDomainIterator;

    /// Returns an iterator over elements in the domain.
    fn into_iter(self) -> LineDomainIterator {
        self.iter()
    }
}

/// An iterator over the x-coordinates of points in a coset.
type LineDomainIterator =
    Map<CosetIterator<CirclePoint<BaseField>>, fn(CirclePoint<BaseField>) -> BaseField>;

pub trait LinePolyOps<F: Field>: FieldOps<F> + Sized {
    fn eval_at_point<E: ExtensionOf<F>>(poly: &LinePoly<Self, F>, x: E) -> E;
    fn evaluate(poly: LinePoly<Self, F>, domain: LineDomain) -> LineEvaluation<Self, F>;
    fn interpolate(eval: LineEvaluation<Self, F>) -> LinePoly<Self, F>;
}

/// A univariate polynomial defined on a [LineDomain].
#[derive(Clone, Debug)]
pub struct LinePoly<B: LinePolyOps<F>, F: Field> {
    /// Coefficients of the polynomial in [line_ifft] algorithm's basis.
    ///
    /// The coefficients are stored in bit-reversed order.
    pub coeffs: Column<B, F>,
    /// The number of coefficients stored as `log2(len(coeffs))`.
    pub log_size: u32,
}

impl<B: LinePolyOps<F>, F: Field> LinePoly<B, F> {
    /// Creates a new line polynomial from bit reversed coefficients.
    ///
    /// # Panics
    ///
    /// Panics if the number of coefficients is not a power of two.
    pub fn new(coeffs: Column<B, F>) -> Self {
        assert!(coeffs.len().is_power_of_two());
        let log_size = coeffs.len().ilog2();
        Self { coeffs, log_size }
    }

    /// Evaluates the polynomial at a single point.
    pub fn eval_at_point<E: ExtensionOf<F>>(&self, x: E) -> E {
        B::eval_at_point(self, x)
    }

    /// Evaluates the polynomial at all points in the domain.
    pub fn evaluate(self, domain: LineDomain) -> LineEvaluation<B, F> {
        B::evaluate(self, domain)
    }

    /// Returns the number of coefficients.
    #[allow(clippy::len_without_is_empty)]
    pub fn len(&self) -> usize {
        // `.len().ilog2()` is a common operation. By returning the length like so the compiler
        // optimizes `.len().ilog2()` to a load of `log_size` instead of a branch and a bit count.
        debug_assert_eq!(self.coeffs.len(), 1 << self.log_size);
        1 << self.log_size
    }

    /// Returns the polynomial's coefficients in their natural order.
    pub fn into_ordered_coefficients(self) -> Column<B, F> {
        B::bit_reverse_column(self.coeffs)
    }

    /// Creates a new line polynomial from coefficients in their natural order.
    ///
    /// # Panics
    ///
    /// Panics if the number of coefficients is not a power of two.
    pub fn from_ordered_coefficients(coeffs: Column<B, F>) -> Self {
        Self::new(B::bit_reverse_column(coeffs))
    }
}

/// Evaluations of a univariate polynomial on a [LineDomain].
// TODO(andrew): Remove EvalOrder. Bit-reversed evals are only necessary since LineEvaluation is
// only used by FRI where evaluations are in bit-reversed order.
#[derive(Clone, Debug)]
pub struct LineEvaluation<B: LinePolyOps<F>, F: Field, EvalOrder = NaturalOrder> {
    /// Evaluations of a univariate polynomial on `domain`.
    pub values: Column<B, F>,
    domain: LineDomain,
    _eval_order: PhantomData<EvalOrder>,
}

impl<B: LinePolyOps<F>, F: Field, EvalOrder> LineEvaluation<B, F, EvalOrder> {
    /// Creates new [LineEvaluation] from a set of polynomial evaluations over a [LineDomain].
    ///
    /// # Panics
    ///
    /// Panics if the number of evaluations does not match the size of the domain.
    pub fn new(domain: LineDomain, evals: Column<B, F>) -> Self {
        assert_eq!(evals.len(), domain.size());
        Self {
            values: evals,
            domain,
            _eval_order: PhantomData,
        }
    }

    pub fn new_zero(domain: LineDomain) -> Self {
        Self::new(domain, Column::<B, F>::zeros(domain.size()))
    }

    /// Returns the number of evaluations.
    #[allow(clippy::len_without_is_empty)]
    pub fn len(&self) -> usize {
        1 << self.domain.log_size()
    }

    pub fn domain(&self) -> LineDomain {
        self.domain
    }
}

impl<B: LinePolyOps<F>, F: Field> LineEvaluation<B, F> {
    /// Interpolates the polynomial as evaluations on `domain`.
    pub fn interpolate(self) -> LinePoly<B, F> {
        B::interpolate(self)
    }

    pub fn bit_reverse(self) -> LineEvaluation<B, F, BitReversedOrder> {
        LineEvaluation {
            values: B::bit_reverse_column(self.values),
            domain: self.domain,
            _eval_order: PhantomData,
        }
    }
}

impl<B: LinePolyOps<F>, F: Field> LineEvaluation<B, F, BitReversedOrder> {
    pub fn bit_reverse(self) -> LineEvaluation<B, F, NaturalOrder> {
        LineEvaluation {
            values: B::bit_reverse_column(self.values),
            domain: self.domain,
            _eval_order: PhantomData,
        }
    }
}

#[cfg(test)]
mod tests {
    type B = CPUBackend;

    use super::LineDomain;
    use crate::core::backend::CPUBackend;
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
        const LOG_SIZE: u32 = 1;
        let coset = Coset::subgroup(LOG_SIZE);

        LineDomain::new(coset);
    }

    #[test]
    fn line_domain_of_size_one_works() {
        const LOG_SIZE: u32 = 0;
        let coset = Coset::subgroup(LOG_SIZE);

        LineDomain::new(coset);
    }

    #[test]
    fn line_domain_size_is_correct() {
        const LOG_SIZE: u32 = 8;
        let coset = Coset::half_odds(LOG_SIZE);
        let domain = LineDomain::new(coset);

        let size = domain.size();

        assert_eq!(size, 1 << LOG_SIZE);
    }

    #[test]
    fn line_domain_coset_returns_the_coset() {
        let coset = Coset::half_odds(5);
        let domain = LineDomain::new(coset);

        assert_eq!(domain.coset(), coset);
    }

    #[test]
    fn line_domain_double_works() {
        const LOG_SIZE: u32 = 8;
        let coset = Coset::half_odds(LOG_SIZE);
        let domain = LineDomain::new(coset);

        let doubled_domain = domain.double();

        assert_eq!(doubled_domain.size(), 1 << (LOG_SIZE - 1));
        assert_eq!(doubled_domain.at(0), CirclePoint::double_x(domain.at(0)));
        assert_eq!(doubled_domain.at(1), CirclePoint::double_x(domain.at(1)));
    }

    #[test]
    fn line_domain_iter_works() {
        const LOG_SIZE: u32 = 8;
        let coset = Coset::half_odds(LOG_SIZE);
        let domain = LineDomain::new(coset);

        let elements = domain.iter().collect::<Vec<BaseField>>();

        assert_eq!(elements.len(), domain.size());
        for (i, element) in elements.into_iter().enumerate() {
            assert_eq!(element, domain.at(i), "mismatch at {i}");
        }
    }

    #[test]
    fn line_polynomial_evaluation() {
        let poly = LinePoly::<B, _>::new(vec![
            BaseField::from(7), // 7 * 1
            BaseField::from(9), // 9 * pi(x)
            BaseField::from(5), // 5 * x
            BaseField::from(3), // 3 * pi(x)*x
        ]);
        let coset = Coset::half_odds(poly.len().ilog2());
        let domain = LineDomain::new(coset);
        let expected_evals = domain
            .iter()
            .map(|x| {
                let pi_x = CirclePoint::double_x(x);
                poly.coeffs[0]
                    + poly.coeffs[1] * pi_x
                    + poly.coeffs[2] * x
                    + poly.coeffs[3] * pi_x * x
            })
            .collect::<Vec<BaseField>>();

        let actual_evals = poly.evaluate(domain);

        assert_eq!(actual_evals.values, expected_evals);
    }

    #[test]
    fn line_polynomial_evaluation_on_larger_domain() {
        let poly = LinePoly::<B, _>::new(vec![
            BaseField::from(7), // 7 * 1
            BaseField::from(9), // 9 * pi(x)
            BaseField::from(5), // 5 * x
            BaseField::from(3), // 3 * pi(x)*x
        ]);
        let coset = Coset::half_odds(4 + poly.len().ilog2());
        let domain = LineDomain::new(coset);
        let expected_evals = domain
            .iter()
            .map(|x| {
                let pi_x = CirclePoint::double_x(x);
                poly.coeffs[0]
                    + poly.coeffs[1] * pi_x
                    + poly.coeffs[2] * x
                    + poly.coeffs[3] * pi_x * x
            })
            .collect::<Vec<BaseField>>();

        let actual_evals = poly.evaluate(domain);

        assert_eq!(actual_evals.values, expected_evals);
    }

    #[test]
    fn line_evaluation_interpolation() {
        let poly = LinePoly::<B, _>::new(vec![
            BaseField::from(7), // 7 * 1
            BaseField::from(9), // 9 * pi(x)
            BaseField::from(5), // 5 * x
            BaseField::from(3), // 3 * pi(x)*x
        ]);
        let coset = Coset::half_odds(poly.len().ilog2());
        let domain = LineDomain::new(coset);
        let evals = LineEvaluation::<B, _>::new(
            domain,
            domain
                .iter()
                .map(|x| {
                    let pi_x = CirclePoint::double_x(x);
                    poly.coeffs[0]
                        + poly.coeffs[1] * pi_x
                        + poly.coeffs[2] * x
                        + poly.coeffs[3] * pi_x * x
                })
                .collect::<Vec<BaseField>>(),
        );

        let interpolated_poly = evals.interpolate();

        assert_eq!(interpolated_poly.coeffs, poly.coeffs);
    }

    #[test]
    fn line_polynomial_eval_at_point() {
        const LOG_SIZE: u32 = 2;
        let coset = Coset::half_odds(LOG_SIZE);
        let domain = LineDomain::new(coset);
        let evals =
            LineEvaluation::<B, _>::new(domain, (0..1 << LOG_SIZE).map(BaseField::from).collect());
        let poly = evals.clone().interpolate();

        for (i, x) in domain.iter().enumerate() {
            assert_eq!(poly.eval_at_point(x), evals.values[i], "mismatch at {i}");
        }
    }
}
