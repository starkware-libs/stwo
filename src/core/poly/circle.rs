use std::fmt::Debug;
use std::iter::Chain;
use std::marker::PhantomData;
use std::ops::{Deref, Index};

use super::twiddles::{TwiddleBank, TwiddleTree};
use super::{BitReversedOrder, NaturalOrder};
use crate::core::air::evaluation::SECURE_EXTENSION_DEGREE;
use crate::core::backend::cpu::CPUCircleEvaluation;
use crate::core::backend::CPUBackend;
use crate::core::circle::{CirclePoint, CirclePointIndex, Coset, CosetIterator};
use crate::core::fields::m31::BaseField;
use crate::core::fields::qm31::SecureField;
use crate::core::fields::{Col, Column, ExtensionOf, FieldOps};
use crate::core::utils::bit_reverse_index;

/// A valid domain for circle polynomial interpolation and evaluation.
/// Valid domains are a disjoint union of two conjugate cosets: +-C + <G_n>.
/// The ordering defined on this domain is C + iG_n, and then -C - iG_n.
#[derive(Copy, Clone, Debug)]
pub struct CircleDomain {
    pub half_coset: Coset,
}

impl CircleDomain {
    /// Given a coset C + <G_n>, constructs the circle domain +-C + <G_n> (i.e.,
    /// this coset and its conjugate).
    pub fn new(half_coset: Coset) -> Self {
        Self { half_coset }
    }

    /// Constructs a domain for constraint evaluation.
    pub fn constraint_evaluation_domain(log_size: u32) -> Self {
        assert!(log_size > 0);
        CircleDomain::new(Coset::new(CirclePointIndex::generator(), log_size - 1))
    }

    pub fn iter(&self) -> CircleDomainIterator {
        self.half_coset
            .iter()
            .chain(self.half_coset.conjugate().iter())
    }

    /// Iterates over point indices.
    pub fn iter_indices(&self) -> CircleDomainIndexIterator {
        self.half_coset
            .iter_indices()
            .chain(self.half_coset.conjugate().iter_indices())
    }

    /// Returns the size of the domain.
    pub fn size(&self) -> usize {
        1 << self.log_size()
    }

    /// Returns the log size of the domain.
    pub fn log_size(&self) -> u32 {
        self.half_coset.log_size + 1
    }

    /// Returns the `i` th domain element.
    pub fn at(&self, i: usize) -> CirclePoint<BaseField> {
        self.index_at(i).to_point()
    }

    /// Returns the [CirclePointIndex] of the `i`th domain element.
    pub fn index_at(&self, i: usize) -> CirclePointIndex {
        if i < self.half_coset.size() {
            self.half_coset.index_at(i)
        } else {
            -self.half_coset.index_at(i - self.half_coset.size())
        }
    }

    pub fn find(&self, i: CirclePointIndex) -> Option<usize> {
        if let Some(d) = self.half_coset.find(i) {
            return Some(d);
        }
        if let Some(d) = self.half_coset.conjugate().find(i) {
            return Some(self.half_coset.size() + d);
        }
        None
    }

    /// Returns true if the domain is canonic.
    ///
    /// Canonic domains are domains with elements that are the entire set of points defined by
    /// `G_2n + <G_n>` where `G_n` and `G_2n` are obtained by repeatedly doubling [M31_CIRCLE_GEN].
    pub fn is_canonic(&self) -> bool {
        self.half_coset.initial_index * 4 == self.half_coset.step_size
    }
}

impl IntoIterator for CircleDomain {
    type Item = CirclePoint<BaseField>;
    type IntoIter = CircleDomainIterator;

    /// Iterates over the points in the domain.
    fn into_iter(self) -> CircleDomainIterator {
        self.iter()
    }
}

/// An iterator over points in a circle domain.
///
/// Let the domain be `+-c + <G>`. The first iterated points are `c + <G>`, then `-c + <-G>`.
pub type CircleDomainIterator =
    Chain<CosetIterator<CirclePoint<BaseField>>, CosetIterator<CirclePoint<BaseField>>>;

/// Like [CircleDomainIterator] but returns corresponding [CirclePointIndex]s.
type CircleDomainIndexIterator =
    Chain<CosetIterator<CirclePointIndex>, CosetIterator<CirclePointIndex>>;

/// A coset of the form G_{2n} + <G_n>, where G_n is the generator of the
/// subgroup of order n. The ordering on this coset is G_2n + i * G_n.
/// These cosets can be used as a [CircleDomain], and be interpolated on.
/// Note that this changes the ordering on the coset to be like [CircleDomain],
/// which is G_2n + i * G_n/2 and then -G_2n -i * G_n/2.
/// For example, the Xs below are a canonic coset with n=8.
/// ```text
///    X O X
///  O       O
/// X         X
/// O         O
/// X         X
///  O       O
///    X O X
/// ```
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct CanonicCoset {
    pub coset: Coset,
}

impl CanonicCoset {
    pub fn new(log_size: u32) -> Self {
        assert!(log_size > 0);
        Self {
            coset: Coset::odds(log_size),
        }
    }

    /// Gets the full coset represented G_{2n} + <G_n>.
    pub fn coset(&self) -> Coset {
        self.coset
    }

    /// Gets half of the coset (its conjugate complements to the whole coset), G_{2n} + <G_{n/2}>
    pub fn half_coset(&self) -> Coset {
        Coset::half_odds(self.log_size() - 1)
    }

    /// Gets the [CircleDomain] representing the same point set (in another order).
    pub fn circle_domain(&self) -> CircleDomain {
        CircleDomain::new(Coset::half_odds(self.coset.log_size - 1))
    }

    /// Gets a good [CircleDomain] for extension of a poly defined on this coset.
    /// The reason the domain looks like this is a bit more intricate, and not covered here.
    pub fn evaluation_domain(&self, log_size: u32) -> CircleDomain {
        assert!(log_size > self.coset.log_size);
        // TODO(spapini): Document why this is like this.
        CircleDomain::new(Coset::new(
            CirclePointIndex::generator() + CirclePointIndex::subgroup_gen(self.coset.log_size + 1),
            log_size - 1,
        ))
    }

    /// Returns the log size of the coset.
    pub fn log_size(&self) -> u32 {
        self.coset.log_size
    }

    /// Returns the size of the coset.
    pub fn size(&self) -> usize {
        self.coset.size()
    }

    pub fn initial_index(&self) -> CirclePointIndex {
        self.coset.initial_index
    }

    pub fn step_size(&self) -> CirclePointIndex {
        self.coset.step_size
    }

    pub fn index_at(&self, index: usize) -> CirclePointIndex {
        self.coset.index_at(index)
    }

    pub fn at(&self, i: usize) -> CirclePoint<BaseField> {
        self.coset.at(i)
    }
}

/// An evaluation defined on a [CircleDomain].
/// The values are ordered according to the [CircleDomain] ordering.
#[derive(Clone, Debug)]
pub struct CircleEvaluation<B: FieldOps<F>, F: ExtensionOf<BaseField>, EvalOrder = NaturalOrder> {
    pub domain: CircleDomain,
    pub values: Col<B, F>,
    _eval_order: PhantomData<EvalOrder>,
}

impl<B: FieldOps<F>, F: ExtensionOf<BaseField>, EvalOrder> CircleEvaluation<B, F, EvalOrder> {
    pub fn new(domain: CircleDomain, values: Col<B, F>) -> Self {
        assert_eq!(domain.size(), values.len());
        Self {
            domain,
            values,
            _eval_order: PhantomData,
        }
    }
}

pub trait PolyOps<F: ExtensionOf<BaseField>>: FieldOps<F> + Sized {
    /// Creates a [CircleEvaluation] from values ordered according to [CanonicCoset].
    /// Used by the [`CircleEvaluation::new_canonical_ordered()`] function.
    fn new_canonical_ordered(
        coset: CanonicCoset,
        values: Col<Self, F>,
    ) -> CircleEvaluation<Self, F, BitReversedOrder>;

    /// Computes a minimal [CirclePoly] that evaluates to the same values as this evaluation.
    /// Used by the [`CircleEvaluation::interpolate()`] function.
    fn interpolate(
        eval: CircleEvaluation<Self, F, BitReversedOrder>,
        itwiddles: &TwiddleTree<Self, F>,
    ) -> CirclePoly<Self, F>;

    /// Evaluates the polynomial at a single point.
    /// Used by the [`CirclePoly::eval_at_point()`] function.
    fn eval_at_point<E: ExtensionOf<F>>(poly: &CirclePoly<Self, F>, point: CirclePoint<E>) -> E;

    /// Extends the polynomial to a larger degree bound.
    /// Used by the [`CirclePoly::extend()`] function.
    fn extend(poly: &CirclePoly<Self, F>, log_size: u32) -> CirclePoly<Self, F>;

    /// Evaluates the polynomial at all points in the domain.
    /// Used by the [`CirclePoly::evaluate()`] function.
    fn evaluate(
        poly: &CirclePoly<Self, F>,
        domain: CircleDomain,
        twiddles: &TwiddleTree<Self, F>,
    ) -> CircleEvaluation<Self, F, BitReversedOrder>;

    type Twiddles;
    fn precompute_twiddles(coset: Coset) -> TwiddleTree<Self, F>;
}

// Note: The concrete implementation of the poly operations is in the specific backend used.
// For example, the CPU backend implementation is in `src/core/backend/cpu/poly.rs`.
impl<F: ExtensionOf<BaseField>, B: PolyOps<F>> CircleEvaluation<B, F, NaturalOrder> {
    pub fn get_at(&self, point_index: CirclePointIndex) -> F {
        self.values
            .at(self.domain.find(point_index).expect("Not in domain"))
    }

    pub fn bit_reverse(mut self) -> CircleEvaluation<B, F, BitReversedOrder> {
        B::bit_reverse_column(&mut self.values);
        CircleEvaluation::new(self.domain, self.values)
    }
}

impl<F: ExtensionOf<BaseField>> CPUCircleEvaluation<F, NaturalOrder> {
    pub fn fetch_eval_on_coset(&self, coset: Coset) -> CosetSubEvaluation<'_, F> {
        assert!(coset.log_size() <= self.domain.half_coset.log_size());
        if let Some(offset) = self.domain.half_coset.find(coset.initial_index) {
            return CosetSubEvaluation::new(
                &self.values[..self.domain.half_coset.size()],
                offset,
                coset.step_size / self.domain.half_coset.step_size,
            );
        }
        if let Some(offset) = self.domain.half_coset.conjugate().find(coset.initial_index) {
            return CosetSubEvaluation::new(
                &self.values[self.domain.half_coset.size()..],
                offset,
                (-coset.step_size) / self.domain.half_coset.step_size,
            );
        }
        panic!("Coset not found in domain");
    }
}

impl<B: PolyOps<F>, F: ExtensionOf<BaseField>> CircleEvaluation<B, F, BitReversedOrder> {
    /// Creates a [CircleEvaluation] from values ordered according to
    /// [CanonicCoset]. For example, the canonic coset might look like this:
    ///   G_8, G_8 + G_4, G_8 + 2G_4, G_8 + 3G_4.
    /// The circle domain will be ordered like this:
    ///   G_8, G_8 + 2G_4, -G_8, -G_8 - 2G_4.
    pub fn new_canonical_ordered(coset: CanonicCoset, values: Col<B, F>) -> Self {
        B::new_canonical_ordered(coset, values)
    }

    /// Computes a minimal [CirclePoly] that evaluates to the same values as this evaluation.
    pub fn interpolate(self) -> CirclePoly<B, F> {
        let coset = self.domain.half_coset;
        B::interpolate(self, &B::precompute_twiddles(coset))
    }

    pub fn interpolate_with_twiddles(self, twiddles: &TwiddleBank<B, F>) -> CirclePoly<B, F> {
        let coset = self.domain.half_coset;
        B::interpolate(self, twiddles.get_tree(coset))
    }

    pub fn bit_reverse(mut self) -> CircleEvaluation<B, F, NaturalOrder> {
        B::bit_reverse_column(&mut self.values);
        CircleEvaluation::new(self.domain, self.values)
    }

    pub fn get_at(&self, point_index: CirclePointIndex) -> F {
        self.values.at(bit_reverse_index(
            self.domain.find(point_index).expect("Not in domain"),
            self.domain.log_size(),
        ))
    }
}

impl<B: FieldOps<F>, F: ExtensionOf<BaseField>, EvalOrder> Deref
    for CircleEvaluation<B, F, EvalOrder>
{
    type Target = Col<B, F>;

    fn deref(&self) -> &Self::Target {
        &self.values
    }
}

/// A part of a [CircleEvaluation], for a specific coset that is a subset of the circle domain.
pub struct CosetSubEvaluation<'a, F: ExtensionOf<BaseField>> {
    evaluation: &'a [F],
    offset: usize,
    step: isize,
}

impl<'a, F: ExtensionOf<BaseField>> CosetSubEvaluation<'a, F> {
    fn new(evaluation: &'a [F], offset: usize, step: isize) -> Self {
        assert!(evaluation.len().is_power_of_two());
        Self {
            evaluation,
            offset,
            step,
        }
    }
}

impl<'a, F: ExtensionOf<BaseField>> Index<isize> for CosetSubEvaluation<'a, F> {
    type Output = F;

    fn index(&self, index: isize) -> &Self::Output {
        let index =
            ((self.offset as isize) + index * self.step) & ((self.evaluation.len() - 1) as isize);
        &self.evaluation[index as usize]
    }
}

impl<'a, F: ExtensionOf<BaseField>> Index<usize> for CosetSubEvaluation<'a, F> {
    type Output = F;

    fn index(&self, index: usize) -> &Self::Output {
        &self[index as isize]
    }
}

/// A polynomial defined on a [CircleDomain].
#[derive(Clone, Debug)]
pub struct CirclePoly<B: FieldOps<F>, F: ExtensionOf<BaseField>> {
    /// Coefficients of the polynomial in the FFT basis.
    /// Note: These are not the coefficients of the polynomial in the standard
    /// monomial basis. The FFT basis is a tensor product of the twiddles:
    /// y, x, pi(x), pi^2(x), ..., pi^{log_size-2}(x).
    /// pi(x) := 2x^2 - 1.
    pub coeffs: Col<B, F>,
    /// The number of coefficients stored as `log2(len(coeffs))`.
    log_size: u32,
}

impl<F: ExtensionOf<BaseField>, B: PolyOps<F>> CirclePoly<B, F> {
    /// Creates a new circle polynomial.
    ///
    /// Coefficients must be in the circle IFFT algorithm's basis stored in bit-reversed order.
    ///
    /// # Panics
    ///
    /// Panics if the number of coefficients isn't a power of two.
    pub fn new(coeffs: Col<B, F>) -> Self {
        assert!(coeffs.len().is_power_of_two());
        let log_size = coeffs.len().ilog2();
        Self { log_size, coeffs }
    }

    pub fn log_size(&self) -> u32 {
        self.log_size
    }

    /// Evaluates the polynomial at a single point.
    pub fn eval_at_point<E: ExtensionOf<F>>(&self, point: CirclePoint<E>) -> E {
        B::eval_at_point(self, point)
    }

    /// Extends the polynomial to a larger degree bound.
    pub fn extend(&self, log_size: u32) -> Self {
        B::extend(self, log_size)
    }

    /// Evaluates the polynomial at all points in the domain.
    pub fn evaluate(&self, domain: CircleDomain) -> CircleEvaluation<B, F, BitReversedOrder> {
        B::evaluate(self, domain, &B::precompute_twiddles(domain.half_coset))
    }

    pub fn evaluate_with_twiddles(
        &self,
        domain: CircleDomain,
        twiddles: &TwiddleBank<B, F>,
    ) -> CircleEvaluation<B, F, BitReversedOrder> {
        B::evaluate(self, domain, twiddles.get_tree(domain.half_coset))
    }
}

#[cfg(test)]
impl<F: ExtensionOf<BaseField>> crate::core::backend::cpu::CPUCirclePoly<F> {
    pub fn is_in_fft_space(&self, log_fft_size: u32) -> bool {
        let mut coeffs = self.coeffs.clone();
        while coeffs.last() == Some(&F::zero()) {
            coeffs.pop();
        }
        coeffs.len() <= 1 << log_fft_size
    }
}

pub struct SecureCirclePoly(pub [CirclePoly<CPUBackend, BaseField>; SECURE_EXTENSION_DEGREE]);

impl SecureCirclePoly {
    pub fn eval_at_point(&self, point: CirclePoint<SecureField>) -> SecureField {
        let mut res = self.0[0].eval_at_point(point);
        res += self.0[1].eval_at_point(point) * SecureField::from_u32_unchecked(0, 1, 0, 0);
        res += self.0[2].eval_at_point(point) * SecureField::from_u32_unchecked(0, 0, 1, 0);
        res += self.0[3].eval_at_point(point) * SecureField::from_u32_unchecked(0, 0, 0, 1);
        res
    }

    // TODO(AlonH): Remove this temporary function.
    pub fn to_circle_poly(&self) -> CirclePoly<CPUBackend, SecureField> {
        let coeffs_len = self[0].coeffs.len();
        let mut coeffs = Col::<CPUBackend, SecureField>::zeros(coeffs_len);
        #[allow(clippy::needless_range_loop)]
        for index in 0..coeffs_len {
            coeffs[index] =
                SecureField::from_m31_array(std::array::from_fn(|i| self[i].coeffs[index]));
        }
        CirclePoly::new(coeffs)
    }
}

impl Deref for SecureCirclePoly {
    type Target = [CirclePoly<CPUBackend, BaseField>; SECURE_EXTENSION_DEGREE];

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

#[derive(Clone, Debug)]
pub struct PointMapping<F: ExtensionOf<BaseField>> {
    pub points: Vec<CirclePoint<F>>,
    pub values: Vec<F>,
}

impl<F: ExtensionOf<BaseField>> PointMapping<F> {
    pub fn new(points: Vec<CirclePoint<F>>, values: Vec<F>) -> Self {
        assert_eq!(points.len(), values.len());
        Self { points, values }
    }

    pub fn len(&self) -> usize {
        self.points.len()
    }

    pub fn is_empty(&self) -> bool {
        self.points.is_empty()
    }

    // TODO(AlonH): Consider optimizing data structures for a more efficient get_at.
    pub fn get_at(&self, point: CirclePoint<F>) -> F {
        self.values[self
            .points
            .iter()
            .position(|p| *p == point)
            .expect("Not in points")]
    }
}

#[cfg(test)]
mod tests {
    use super::{CanonicCoset, CircleDomain, Coset};
    use crate::core::backend::cpu::{CPUCircleEvaluation, CPUCirclePoly};
    use crate::core::circle::{CirclePoint, CirclePointIndex};
    use crate::core::fields::m31::{BaseField, M31};
    use crate::core::fields::Field;
    use crate::core::poly::NaturalOrder;
    use crate::core::utils::bit_reverse_index;
    use crate::m31;

    #[test]
    fn test_circle_domain_iterator() {
        let domain = CircleDomain::constraint_evaluation_domain(3);
        for (i, point) in domain.iter().enumerate() {
            if i < 4 {
                assert_eq!(
                    point,
                    (CirclePointIndex::generator() + CirclePointIndex::subgroup_gen(2) * i)
                        .to_point()
                );
            } else {
                assert_eq!(
                    point,
                    (-(CirclePointIndex::generator() + CirclePointIndex::subgroup_gen(2) * i))
                        .to_point()
                );
            }
        }
    }

    #[test]
    fn test_interpolate_and_eval() {
        let domain = CircleDomain::constraint_evaluation_domain(3);
        assert_eq!(domain.log_size(), 3);
        let evaluation =
            CPUCircleEvaluation::new(domain, (0..8).map(BaseField::from_u32_unchecked).collect());
        let poly = evaluation.clone().interpolate();
        let evaluation2 = poly.evaluate(domain);
        assert_eq!(evaluation.values, evaluation2.values);
    }

    #[test]
    fn test_interpolate_canonic_eval() {
        let domain = CircleDomain::constraint_evaluation_domain(3);
        assert_eq!(domain.log_size(), 3);
        let evaluation = CPUCircleEvaluation::<_, NaturalOrder>::new(
            domain,
            (0..8).map(BaseField::from_u32_unchecked).collect(),
        )
        .bit_reverse();
        let poly = evaluation.interpolate();
        for (i, point) in domain.iter().enumerate() {
            assert_eq!(poly.eval_at_point(point), m31!(i as u32));
        }
    }

    #[test]
    fn test_interpolate_canonic() {
        let coset = CanonicCoset::new(3);
        let evaluation = CPUCircleEvaluation::new_canonical_ordered(
            coset,
            (0..8).map(BaseField::from_u32_unchecked).collect(),
        );
        let poly = evaluation.interpolate();
        for (i, point) in Coset::odds(3).iter().enumerate() {
            assert_eq!(poly.eval_at_point(point), m31!(i as u32));
        }
    }

    #[test]
    fn test_mixed_degree_example() {
        let log_size = 4;

        // Compute domains.
        let domain0 = CanonicCoset::new(log_size);
        let eval_domain0 = domain0.evaluation_domain(log_size + 4);
        let domain1 = CanonicCoset::new(log_size + 2);
        let eval_domain1 = domain1.evaluation_domain(log_size + 3);
        let constraint_domain = CircleDomain::constraint_evaluation_domain(log_size + 1);

        // Compute values.
        let values1: Vec<_> = (0..(domain1.size() as u32))
            .map(BaseField::from_u32_unchecked)
            .collect();
        let values0: Vec<_> = values1[1..].iter().step_by(4).map(|x| *x * *x).collect();

        // Extend.
        let trace_eval0 = CPUCircleEvaluation::new_canonical_ordered(domain0, values0);
        let eval0 = trace_eval0.interpolate().evaluate(eval_domain0);
        let trace_eval1 = CPUCircleEvaluation::new_canonical_ordered(domain1, values1);
        let eval1 = trace_eval1.interpolate().evaluate(eval_domain1);

        // Compute constraint.
        let constraint_eval = CPUCircleEvaluation::<BaseField, NaturalOrder>::new(
            constraint_domain,
            constraint_domain
                .iter_indices()
                .map(|ind| {
                    // The constraint is poly0(x+off0)^2 = poly1(x+off1).
                    eval0.get_at(ind).square() - eval1.get_at(domain1.index_at(1) + ind).square()
                })
                .collect(),
        );
        // TODO(spapini): Check low degree.
        println!("{:?}", constraint_eval);
    }

    #[test]
    fn is_canonic_valid_domain() {
        let canonic_domain = CanonicCoset::new(4).circle_domain();

        assert!(canonic_domain.is_canonic());
    }

    #[test]
    fn is_canonic_invalid_domain() {
        let half_coset = Coset::new(CirclePointIndex::generator(), 4);
        let not_canonic_domain = CircleDomain::new(half_coset);

        assert!(!not_canonic_domain.is_canonic());
    }

    #[test]
    pub fn test_bit_reverse_indices() {
        let log_domain_size = 7;
        let log_small_domain_size = 5;
        let domain = CanonicCoset::new(log_domain_size);
        let small_domain = CanonicCoset::new(log_small_domain_size);
        let n_folds = log_domain_size - log_small_domain_size;
        for i in 0..2usize.pow(log_domain_size) {
            let point = domain.at(bit_reverse_index(i, log_domain_size));
            let small_point = small_domain.at(bit_reverse_index(
                i / 2usize.pow(n_folds),
                log_small_domain_size,
            ));
            assert_eq!(point.repeated_double(n_folds), small_point);
        }
    }

    #[test]
    pub fn test_at_circle_domain() {
        let domain = CanonicCoset::new(7).circle_domain();
        let half_domain_size = domain.size() / 2;

        for i in 0..half_domain_size {
            assert_eq!(domain.index_at(i), -domain.index_at(i + half_domain_size));
            assert_eq!(domain.at(i), domain.at(i + half_domain_size).conjugate());
        }
    }

    #[test]
    pub fn test_get_at_circle_evaluation() {
        let domain = CanonicCoset::new(7).circle_domain();
        let values = (0..domain.size()).map(|i| m31!(i as u32)).collect();
        let circle_evaluation = CPUCircleEvaluation::<_, NaturalOrder>::new(domain, values);
        let bit_reversed_circle_evaluation = circle_evaluation.clone().bit_reverse();
        for index in domain.iter_indices() {
            assert_eq!(
                circle_evaluation.get_at(index),
                bit_reversed_circle_evaluation.get_at(index)
            );
        }
    }

    #[test]
    fn test_circle_poly_extend() {
        let poly = CPUCirclePoly::new((0..16).map(BaseField::from_u32_unchecked).collect());
        let extended = poly.clone().extend(8);
        let random_point = CirclePoint::get_point(21903);

        assert_eq!(
            poly.eval_at_point(random_point),
            extended.eval_at_point(random_point)
        );
    }

    #[test]
    fn test_sub_evaluation() {
        let domain = CanonicCoset::new(7).circle_domain();
        let values = (0..domain.size()).map(|i| m31!(i as u32)).collect();
        let circle_evaluation = CPUCircleEvaluation::new(domain, values);
        let coset = Coset::new(domain.index_at(17), 3);
        let sub_eval = circle_evaluation.fetch_eval_on_coset(coset);
        for i in 0..coset.size() {
            assert_eq!(sub_eval[i], circle_evaluation.get_at(coset.index_at(i)));
        }
    }
}
