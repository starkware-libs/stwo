use std::collections::BTreeMap;
use std::iter::Chain;
use std::ops::Deref;

use num_traits::One;

use crate::core::circle::{CirclePoint, CirclePointIndex, Coset, CosetIterator};
use crate::core::fft::{butterfly, ibutterfly};
use crate::core::fields::m31::BaseField;
use crate::core::fields::{ExtensionOf, Field};

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
    pub fn constraint_evaluation_domain(n_bits: usize) -> Self {
        assert!(n_bits > 0);
        Self {
            half_coset: Coset::new(CirclePointIndex::generator(), n_bits - 1),
        }
    }

    pub fn iter(
        &self,
    ) -> Chain<CosetIterator<CirclePoint<BaseField>>, CosetIterator<CirclePoint<BaseField>>> {
        self.half_coset
            .iter()
            .chain(self.half_coset.conjugate().iter())
    }

    pub fn iter_indices(
        &self,
    ) -> Chain<CosetIterator<CirclePointIndex>, CosetIterator<CirclePointIndex>> {
        self.half_coset
            .iter_indices()
            .chain(self.half_coset.conjugate().iter_indices())
    }

    /// Returns the size of the domain.
    pub fn size(&self) -> usize {
        self.half_coset.size() * 2
    }

    pub fn n_bits(&self) -> usize {
        self.half_coset.n_bits + 1
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
}

/// A coset of the form G_{2n} + <G_n>, where G_n is the generator of the
/// subgroup of order n. The ordering on this coset is G_2n + i * G_n.
/// These cosets can be used as a [CircleDomain], and be interpolated on.
/// Not that this changes the ordering on the coset to be like [CircleDomain],
/// which is G_2n + i * G_2n and then -G_2n -i * G_2n.
/// For example, the Xs below are a canonic coset with n_bits=3.
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
    pub fn new(n_bits: usize) -> Self {
        assert!(n_bits > 0);
        Self {
            coset: Coset::odds(n_bits),
        }
    }

    /// Gets the full coset represented G_{2n} + <G_n>.
    pub fn coset(&self) -> Coset {
        self.coset
    }

    /// Gets half of the coset (its conjugate complements to the whole coset), G_{2n} + <G_{n/2}>
    pub fn half_coset(&self) -> Coset {
        Coset::half_odds(self.n_bits - 1)
    }

    /// Gets the [CircleDomain] representing the same point set (in another order).
    pub fn circle_domain(&self) -> CircleDomain {
        CircleDomain::new(Coset::half_odds(self.coset.n_bits - 1))
    }

    /// Gets a good [CircleDomain] for extension of a poly defined on this coset.
    /// The reason the domain looks like this is a bit more intricate, and not covered here.
    pub fn evaluation_domain(&self, eval_n_bits: usize) -> CircleDomain {
        assert!(eval_n_bits > self.coset.n_bits);
        // TODO(spapini): Document why this is like this.
        if eval_n_bits == self.coset.n_bits + 1 {
            return CircleDomain::new(Coset::new(
                CirclePointIndex::generator() + CirclePointIndex::subgroup_gen(eval_n_bits),
                eval_n_bits - 1,
            ));
        }
        CircleDomain::new(Coset::new(CirclePointIndex::generator(), eval_n_bits - 1))
    }

    pub fn n_bits(&self) -> usize {
        self.coset.n_bits
    }
}

impl Deref for CanonicCoset {
    type Target = Coset;

    fn deref(&self) -> &Self::Target {
        &self.coset
    }
}

pub trait Evaluation: Clone {
    fn get_at(&self, point_index: CirclePointIndex) -> BaseField;
}

/// An evaluation defined on a [CircleDomain].
/// The values are ordered according to the [CircleDomain] ordering.
#[derive(Clone, Debug)]
pub struct CircleEvaluation<F: ExtensionOf<BaseField>> {
    pub domain: CircleDomain,
    pub values: Vec<F>,
}

impl<F: ExtensionOf<BaseField>> CircleEvaluation<F> {
    pub fn new(domain: CircleDomain, values: Vec<F>) -> Self {
        assert_eq!(domain.size(), values.len());
        Self { domain, values }
    }

    /// Creates a [CircleEvaluation] from values ordered according to
    /// [CanonicCoset]. For example, the canonic coset might look like this:
    ///   G_8, G_8 + G_4, G_8 + 2G_4, G_8 + 3G_4.
    /// The circle domain will be ordered like this:
    ///   G_8, G_8 + 2G_4, -G_8, -G_8 - 2G_4.
    pub fn new_canonical_ordered(coset: CanonicCoset, values: Vec<F>) -> Self {
        let domain = coset.circle_domain();
        assert_eq!(values.len(), domain.size());
        let mut new_values = Vec::with_capacity(values.len());
        let half_len = 1 << (coset.n_bits - 1);
        for i in 0..half_len {
            new_values.push(values[i << 1]);
        }
        for i in 0..half_len {
            new_values.push(values[domain.size() - 1 - (i << 1)]);
        }
        Self {
            domain,
            values: new_values,
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

        // Divide all values by 2^n_bits.
        let inv = BaseField::from_u32_unchecked(self.domain.size() as u32).inverse();
        for val in &mut values {
            *val *= inv;
        }

        CirclePoly {
            bound_bits: self.domain.n_bits(),
            coeffs: values,
        }
    }
}

impl Evaluation for CircleEvaluation<BaseField> {
    fn get_at(&self, point_index: CirclePointIndex) -> BaseField {
        self.values[self.domain.find(point_index).expect("Not in domain")]
    }
}

/// A polynomial defined on a [CircleDomain].
#[derive(Clone, Debug)]
pub struct CirclePoly<F: ExtensionOf<BaseField>> {
    /// log size of the number of coefficients.
    bound_bits: usize,
    /// Coefficients of the polynomial in the FFT basis.
    /// Note: These are not the coefficients of the polynomial in the standard
    /// monomial basis. The FFT basis is a tensor product of the twiddles:
    /// y, x, psi_x(x), psi_x^2(x), ..., psi_x^{bound_bits-2}(x).
    /// psi_x(x) := 2x^2 - 1.
    coeffs: Vec<F>,
}

impl<F: ExtensionOf<BaseField>> CirclePoly<F> {
    pub fn new(bound_bits: usize, coeffs: Vec<F>) -> Self {
        assert!(coeffs.len() == (1 << bound_bits));
        Self { bound_bits, coeffs }
    }

    pub fn eval_at_point(&self, point: CirclePoint<BaseField>) -> F {
        let mut mults = vec![BaseField::one(), point.y];
        let mut x = point.x;
        for _ in 0..(self.bound_bits - 1) {
            mults.push(x);
            x = CirclePoint::double_x(x)
        }
        mults.reverse();

        let mut sum = F::zero();
        for (i, val) in self.coeffs.iter().enumerate() {
            let mut cur_mult = F::one();
            for (j, mult) in mults.iter().enumerate() {
                if i & (1 << j) != 0 {
                    cur_mult *= *mult;
                }
            }
            sum += *val * cur_mult;
        }
        sum
    }

    /// Evaluates the polynomial at all points in the domain.
    pub fn evaluate(self, domain: CircleDomain) -> CircleEvaluation<F> {
        // Use CFFT to evaluate.
        let mut coset = domain.half_coset;
        let mut cosets = vec![];

        // TODO(spapini): extend better.
        assert!(domain.n_bits() >= self.bound_bits);
        let mut values = vec![F::zero(); domain.size()];
        let jump_bits = domain.n_bits() - self.bound_bits;
        for (i, val) in self.coeffs.iter().enumerate() {
            values[i << jump_bits] = *val;
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
        CircleEvaluation { domain, values }
    }
}

#[derive(Clone, Debug)]
pub struct PointSetEvaluation(BTreeMap<CirclePointIndex, BaseField>);

impl PointSetEvaluation {
    pub fn new(evaluations: BTreeMap<CirclePointIndex, BaseField>) -> Self {
        Self(evaluations)
    }

    pub fn len(&self) -> usize {
        self.0.len()
    }

    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }
}

impl Evaluation for PointSetEvaluation {
    fn get_at(&self, point_index: CirclePointIndex) -> BaseField {
        *self
            .0
            .get(&point_index)
            .unwrap_or_else(|| panic!("Point not found in evaluation for {:?}", point_index))
    }
}

#[cfg(test)]
mod tests {
    use super::{CanonicCoset, CircleDomain, CircleEvaluation, Coset};
    use crate::core::circle::CirclePointIndex;
    use crate::core::constraints::{EvalByEvaluation, PolyOracle};
    use crate::core::fields::m31::{BaseField, M31};
    use crate::core::fields::Field;
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
        assert_eq!(domain.n_bits(), 3);
        let evaluation =
            CircleEvaluation::new(domain, (0..8).map(BaseField::from_u32_unchecked).collect());
        let poly = evaluation.clone().interpolate();
        let evaluation2 = poly.evaluate(domain);
        assert_eq!(evaluation.values, evaluation2.values);
    }

    #[test]
    fn test_interpolate_canonic_eval() {
        let domain = CircleDomain::constraint_evaluation_domain(3);
        assert_eq!(domain.n_bits(), 3);
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
        let n_bits = 4;

        // Compute domains.
        let domain0 = CanonicCoset::new(n_bits);
        let eval_domain0 = domain0.evaluation_domain(n_bits + 4);
        let domain1 = CanonicCoset::new(n_bits + 2);
        let eval_domain1 = domain1.evaluation_domain(n_bits + 3);
        let constraint_domain = CircleDomain::constraint_evaluation_domain(n_bits + 1);

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
        let constraint_eval = CircleEvaluation::new(
            constraint_domain,
            constraint_domain
                .iter_indices()
                .map(|ind| {
                    // The constraint is poly0(x+off0)^2 = poly1(x+off1).
                    EvalByEvaluation {
                        offset: domain0.initial_index,
                        eval: &eval0,
                    }
                    .get_at(ind)
                    .square()
                        - EvalByEvaluation {
                            offset: domain1.index_at(1),
                            eval: &eval1,
                        }
                        .get_at(ind)
                        .square()
                })
                .collect(),
        );
        // TODO(spapini): Check low degree.
        println!("{:?}", constraint_eval);
    }
}
