use std::fmt::Debug;
use std::iter::Chain;
use std::marker::PhantomData;
use std::ops::Deref;

use super::utils::fold;
use super::{BitReversedOrder, NaturalOrder};
use crate::core::circle::{CirclePoint, CirclePointIndex, Coset, CosetIterator};
use crate::core::fft::{butterfly, ibutterfly};
use crate::core::fields::m31::BaseField;
use crate::core::fields::{ExtensionOf, Field};
use crate::core::utils::bit_reverse;

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
        Self {
            half_coset: Coset::new(CirclePointIndex::generator(), log_size - 1),
        }
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

    pub fn at(&self, index: usize) -> CirclePoint<BaseField> {
        if index < self.half_coset.size() {
            self.half_coset.at(index)
        } else {
            self.half_coset
                .at(index - self.half_coset.size())
                .conjugate()
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

    pub fn index_at(&self, i: usize) -> CirclePointIndex {
        self.coset.index_at(i)
    }

    pub fn at(&self, i: usize) -> CirclePoint<BaseField> {
        self.coset.at(i)
    }
}

/// An evaluation defined on a [CircleDomain].
/// The values are ordered according to the [CircleDomain] ordering.
pub struct CircleEvaluation<F: ExtensionOf<BaseField>, EvalOrder = NaturalOrder> {
    pub domain: CircleDomain,
    pub values: Vec<F>,
    _eval_order: PhantomData<EvalOrder>,
}

impl<F: ExtensionOf<BaseField>, EvalOrder> CircleEvaluation<F, EvalOrder> {
    pub fn new(domain: CircleDomain, values: Vec<F>) -> Self {
        assert_eq!(domain.size(), values.len());
        Self {
            domain,
            values,
            _eval_order: PhantomData,
        }
    }
}

impl<F: ExtensionOf<BaseField>> CircleEvaluation<F> {
    /// Creates a [CircleEvaluation] from values ordered according to
    /// [CanonicCoset]. For example, the canonic coset might look like this:
    ///   G_8, G_8 + G_4, G_8 + 2G_4, G_8 + 3G_4.
    /// The circle domain will be ordered like this:
    ///   G_8, G_8 + 2G_4, -G_8, -G_8 - 2G_4.
    pub fn new_canonical_ordered(coset: CanonicCoset, values: Vec<F>) -> Self {
        let domain = coset.circle_domain();
        assert_eq!(values.len(), domain.size());
        let mut new_values = Vec::with_capacity(values.len());
        let half_len = 1 << (coset.log_size() - 1);
        for i in 0..half_len {
            new_values.push(values[i << 1]);
        }
        for i in 0..half_len {
            new_values.push(values[domain.size() - 1 - (i << 1)]);
        }
        Self {
            domain,
            values: new_values,
            _eval_order: PhantomData,
        }
    }

    /// Computes a minimal [CirclePoly] that evaluates to the same values as this evaluation.
    pub fn interpolate(self) -> CirclePoly<F> {
        // Use CFFT to interpolate.
        let mut coset = self.domain.half_coset;
        let mut values = self.values;
        let (l, r) = values.split_at_mut(coset.size());
        for (i, p) in coset.iter().enumerate() {
            ibutterfly(&mut l[i], &mut r[i], p.y.inverse());
        }
        while coset.size() > 1 {
            for chunk in values.chunks_exact_mut(coset.size()) {
                let (l, r) = chunk.split_at_mut(coset.size() / 2);
                for (i, p) in coset.iter().take(coset.size() / 2).enumerate() {
                    ibutterfly(&mut l[i], &mut r[i], p.x.inverse());
                }
            }
            coset = coset.double();
        }

        // Divide all values by 2^log_size.
        let inv = BaseField::from_u32_unchecked(self.domain.size() as u32).inverse();
        for val in &mut values {
            *val *= inv;
        }

        CirclePoly::new(values)
    }

    pub fn get_at(&self, point_index: CirclePointIndex) -> F {
        self.values[self.domain.find(point_index).expect("Not in domain")]
    }

    pub fn bit_reverse(self) -> CircleEvaluation<F, BitReversedOrder> {
        CircleEvaluation {
            values: bit_reverse(self.values),
            domain: self.domain,
            _eval_order: PhantomData,
        }
    }
}

impl<F: ExtensionOf<BaseField>> CircleEvaluation<F, BitReversedOrder> {
    pub fn bit_reverse(self) -> CircleEvaluation<F, NaturalOrder> {
        CircleEvaluation {
            values: bit_reverse(self.values),
            domain: self.domain,
            _eval_order: PhantomData,
        }
    }
}

impl<F: ExtensionOf<BaseField>, EvalOrder> Debug for CircleEvaluation<F, EvalOrder> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CircleEvaluation")
            .field("domain", &self.domain)
            .field("values", &self.values)
            .field("_eval_order", &self._eval_order)
            .finish()
    }
}

impl<F: ExtensionOf<BaseField>, EvalOrder> Clone for CircleEvaluation<F, EvalOrder> {
    fn clone(&self) -> Self {
        Self {
            domain: self.domain,
            values: self.values.clone(),
            _eval_order: PhantomData,
        }
    }
}

impl<F: ExtensionOf<BaseField>, EvalOrder> Deref for CircleEvaluation<F, EvalOrder> {
    type Target = [F];

    fn deref(&self) -> &[F] {
        &self.values
    }
}

impl<F: ExtensionOf<BaseField>, EvalOrder> IntoIterator for CircleEvaluation<F, EvalOrder> {
    type Item = F;
    type IntoIter = std::vec::IntoIter<F>;

    /// Creates a consuming iterator over the evaluations.
    ///
    /// Evaluations are returned in the same order as elements of the domain.
    fn into_iter(self) -> Self::IntoIter {
        self.values.into_iter()
    }
}

/// A polynomial defined on a [CircleDomain].
#[derive(Clone, Debug)]
pub struct CirclePoly<F: ExtensionOf<BaseField>> {
    /// Coefficients of the polynomial in the FFT basis.
    /// Note: These are not the coefficients of the polynomial in the standard
    /// monomial basis. The FFT basis is a tensor product of the twiddles:
    /// y, x, pi(x), pi^2(x), ..., pi^{log_size-2}(x).
    /// pi(x) := 2x^2 - 1.
    coeffs: Vec<F>,
    /// The number of coefficients stored as `log2(len(coeffs))`.
    log_size: u32,
}

impl<F: ExtensionOf<BaseField>> CirclePoly<F> {
    /// Creates a new circle polynomial.
    ///
    /// Coefficients must be in the circle IFFT algorithm's basis stored in bit-reversed order.
    ///
    /// # Panics
    ///
    /// Panics if the number of coefficients isn't a power of two.
    pub fn new(coeffs: Vec<F>) -> Self {
        assert!(coeffs.len().is_power_of_two());
        let log_size = coeffs.len().ilog2();
        Self { log_size, coeffs }
    }

    /// Evaluates the polynomial at a single point.
    pub fn eval_at_point<E: ExtensionOf<F>>(&self, point: CirclePoint<E>) -> E {
        // TODO(Andrew): Allocation here expensive for small polynomials.
        let mut mappings = vec![point.y, point.x];
        let mut x = point.x;
        for _ in 2..self.log_size {
            x = CirclePoint::double_x(x);
            mappings.push(x);
        }
        fold(&self.coeffs, &mappings)
    }

    /// Evaluates the polynomial at all points in the domain.
    pub fn evaluate(&self, domain: CircleDomain) -> CircleEvaluation<F> {
        // Use CFFT to evaluate.
        let mut coset = domain.half_coset;
        let mut cosets = vec![];

        // TODO(spapini): extend better.
        assert!(domain.log_size() >= self.log_size);
        let mut values = vec![F::zero(); domain.size()];
        let log_jump = domain.log_size() - self.log_size;
        for (i, val) in self.coeffs.iter().enumerate() {
            values[i << log_jump] = *val;
        }

        while coset.size() > 1 {
            cosets.push(coset);
            coset = coset.double();
        }
        for coset in cosets.iter().rev() {
            for chunk in values.chunks_exact_mut(coset.size()) {
                let (l, r) = chunk.split_at_mut(coset.size() / 2);
                for (i, p) in coset.iter().take(coset.size() / 2).enumerate() {
                    butterfly(&mut l[i], &mut r[i], p.x);
                }
            }
        }
        let coset = domain.half_coset;
        let (l, r) = values.split_at_mut(coset.size());
        for (i, p) in coset.iter().enumerate() {
            butterfly(&mut l[i], &mut r[i], p.y);
        }
        CircleEvaluation {
            domain,
            values,
            _eval_order: PhantomData,
        }
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
    use super::{CanonicCoset, CircleDomain, CircleEvaluation, Coset};
    use crate::core::circle::CirclePointIndex;
    use crate::core::constraints::{EvalByEvaluation, PolyOracle};
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
            CircleEvaluation::new(domain, (0..8).map(BaseField::from_u32_unchecked).collect());
        let poly = evaluation.clone().interpolate();
        let evaluation2 = poly.evaluate(domain);
        assert_eq!(evaluation.values, evaluation2.values);
    }

    #[test]
    fn test_interpolate_canonic_eval() {
        let domain = CircleDomain::constraint_evaluation_domain(3);
        assert_eq!(domain.log_size(), 3);
        let evaluation =
            CircleEvaluation::new(domain, (0..8).map(BaseField::from_u32_unchecked).collect());
        let poly = evaluation.interpolate();
        for (i, point) in domain.iter().enumerate() {
            assert_eq!(poly.eval_at_point(point), m31!(i as u32));
        }
    }

    #[test]
    fn test_interpolate_canonic() {
        let coset = CanonicCoset::new(3);
        let evaluation = CircleEvaluation::new_canonical_ordered(
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
        let trace_eval0 = CircleEvaluation::new_canonical_ordered(domain0, values0);
        let eval0 = trace_eval0.interpolate().evaluate(eval_domain0);
        let trace_eval1 = CircleEvaluation::new_canonical_ordered(domain1, values1);
        let eval1 = trace_eval1.interpolate().evaluate(eval_domain1);

        // Compute constraint.
        let constraint_eval = CircleEvaluation::<BaseField, NaturalOrder>::new(
            constraint_domain,
            constraint_domain
                .iter_indices()
                .map(|ind| {
                    // The constraint is poly0(x+off0)^2 = poly1(x+off1).
                    EvalByEvaluation::new(domain0.initial_index(), &eval0)
                        .get_at(ind)
                        .square()
                        - EvalByEvaluation::new(domain1.index_at(1), &eval1)
                            .get_at(ind)
                            .square()
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
}
