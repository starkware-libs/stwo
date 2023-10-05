use std::{iter::Chain, ops::Deref};

use crate::core::{
    circle::{CirclePoint, CirclePointIndex, Coset, CosetIterator},
    fft::FFTree,
    fields::m31::Field,
};

use super::line::{LineDomain, LineEvaluation, LinePoly};

/// A valid domain for circle polynomial interpolation and evaluation.
/// Valid domains are a disjoint union of two conjugate cosets: +-C + <G_n>.
/// The ordering defined on this domain is C + iG_n, and then -C - iG_n.
#[derive(Copy, Clone, Debug)]
pub struct CircleDomain {
    pub half_coset: Coset,
}
impl CircleDomain {
    /// Given a coset C + <G_n>, constructs the circle domain +-C + <G_n> (i.e., this coset and
    /// its conjugate).
    pub fn new(half_coset: Coset) -> Self {
        Self { half_coset }
    }
    /// Constructs a domain for constraint evaluation.
    pub fn constraint_domain(n_bits: usize) -> Self {
        assert!(n_bits > 0);
        Self {
            half_coset: Coset::new(CirclePointIndex::generator(), n_bits - 1),
        }
    }
    /// Returns the [LineDomain] of all x coordinates of this domain.
    pub fn line_domain(&self) -> LineDomain {
        LineDomain::new(self.half_coset)
    }
    pub fn iter(&self) -> Chain<CosetIterator<CirclePoint>, CosetIterator<CirclePoint>> {
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
    pub fn len(&self) -> usize {
        self.half_coset.len() * 2
    }
    pub fn is_empty(&self) -> bool {
        false
    }
    pub fn n_bits(&self) -> usize {
        self.half_coset.n_bits + 1
    }
    pub fn at(&self, index: usize) -> CirclePoint {
        if index < self.half_coset.len() {
            self.half_coset.at(index)
        } else {
            self.half_coset
                .at(index - self.half_coset.len())
                .conjugate()
        }
    }
    pub fn find(&self, i: CirclePointIndex) -> Option<usize> {
        if let Some(d) = self.half_coset.find(i) {
            return Some(d);
        }
        if let Some(d) = self.half_coset.conjugate().find(i) {
            return Some(self.half_coset.len() + d);
        }
        None
    }
}

/// A coset of the form G_{2n} + <G_n>, where G_n is the generator of the subgroup of order n.
/// The ordering on this coset is G_2n + i * G_n.
/// These cosets can be used as a [CircleDomain], and be interpolated on.
/// Not that this changes the ordering on the coset to be like [CircleDomain], which is
/// G_2n + i * G_2n and then -G_2n -i * G_2n.
/// For example, the Xs below are a canonic coset with n_bits=3.
///    X O X
///  O       O
/// X         X
/// O         O
/// X         X
///  O       O
///    X O X
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
    /// Gets the [LineDomain] of the x coordinates of the coset.
    pub fn line_domain(&self) -> LineDomain {
        LineDomain::new(self.half_coset())
    }
    /// Gets a good [CircleDomain] for extension of a poly defined on this coset.
    /// The reason the domain looks like this is a bit more intricate, and not covered here.
    pub fn eval_domain(&self, eval_n_bits: usize) -> CircleDomain {
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

/// An evaluation defined on a [CircleDomain].
/// The values are ordered according to the [CircleDomain] ordering.
#[derive(Clone, Debug)]
pub struct CircleEvaluation {
    pub domain: CircleDomain,
    pub values: Vec<Field>,
}
impl CircleEvaluation {
    pub fn new(domain: CircleDomain, values: Vec<Field>) -> Self {
        assert_eq!(domain.len(), values.len());
        Self { domain, values }
    }
    /// Creates a [CircleEvaluation] from values ordered according to [CanonicCoset].
    /// For example, the canonic coset might look like this:
    ///   G_8, G_8 + G_4, G_8 + 2G_4, G_8 + 3G_4.
    /// The circle domain will be ordered like this:
    ///   G_8, G_8 + 2G_4, -G_8, -G_8 - 2G_4.
    pub fn new_canonical_ordered(coset: CanonicCoset, values: Vec<Field>) -> Self {
        let domain = coset.circle_domain();
        assert_eq!(values.len(), domain.len());
        let mut new_values = Vec::with_capacity(values.len());
        let half_len = 1 << (coset.n_bits - 1);
        for i in 0..half_len {
            new_values.push(values[i << 1]);
        }
        for i in 0..half_len {
            new_values.push(values[domain.len() - 1 - (i << 1)]);
        }
        Self {
            domain,
            values: new_values,
        }
    }
    pub fn semi_interpolate(self) -> CircleSemiEval {
        assert!(self.domain.n_bits() > 0);
        let line_domain = self.domain.line_domain();

        let mut poly0_values = Vec::with_capacity(line_domain.len());
        let mut poly1_values = Vec::with_capacity(line_domain.len());

        for (i, point) in line_domain.coset().iter().enumerate() {
            let v0 = self.values[i];
            let v1 = self.values[i + line_domain.len()];
            let r0 = (v0 + v1) / Field::from_u32_unchecked(2);
            let r1 = (v0 - v1) * point.y.inverse() / Field::from_u32_unchecked(2);
            poly0_values.push(r0);
            poly1_values.push(r1);
        }

        CircleSemiEval {
            domain: self.domain,
            poly0_eval: LineEvaluation::new(line_domain, poly0_values),
            poly1_eval: LineEvaluation::new(line_domain, poly1_values),
        }
    }
    pub fn interpolate(self, tree: &FFTree) -> CirclePoly {
        self.semi_interpolate().interpolate(tree)
    }
}

// TODO(spapini): Try to get rid of this.
/// An evaluation of 2 univariate polyomials defined on the x coordinates of a circle polynomial.
#[derive(Clone, Debug)]
pub struct CircleSemiEval {
    pub domain: CircleDomain,
    pub poly0_eval: LineEvaluation,
    pub poly1_eval: LineEvaluation,
}
impl CircleSemiEval {
    pub fn interpolate(self, tree: &FFTree) -> CirclePoly {
        let poly0 = self.poly0_eval.interpolate(tree);
        let poly1 = self.poly1_eval.interpolate(tree);
        CirclePoly {
            domain: self.domain,
            poly0,
            poly1,
        }
    }
    pub fn evaluate(self) -> CircleEvaluation {
        let mut values = vec![Field::zero(); self.domain.len()];
        let line_domain = self.domain.line_domain();
        for (i, point) in line_domain.coset().iter().enumerate() {
            let v0 = self.poly0_eval.values[i];
            let v1 = self.poly1_eval.values[i];
            values[i] = v0 + v1 * point.y;
            values[i + line_domain.len()] = v0 - v1 * point.y;
        }
        CircleEvaluation::new(self.domain, values)
    }
}

/// A polynomial defined on a [CircleDomain].
#[derive(Clone, Debug)]
pub struct CirclePoly {
    pub domain: CircleDomain,
    pub poly0: LinePoly,
    pub poly1: LinePoly,
}
impl CirclePoly {
    pub fn eval_at_point(&self, point: CirclePoint) -> Field {
        let v0 = self.poly0.eval_at_point(point.x);
        let v1 = self.poly1.eval_at_point(point.x);
        v0 + v1 * point.y
    }
    pub fn extend(&self, extended_domain: CircleDomain) -> CirclePoly {
        assert!(extended_domain.n_bits() >= self.domain.n_bits());

        let extended_line_domain = extended_domain.line_domain();
        let poly0 = self.poly0.extend(extended_line_domain);
        let poly1 = self.poly1.extend(extended_line_domain);
        CirclePoly {
            domain: extended_domain,
            poly0,
            poly1,
        }
    }
    pub fn semi_evaluate(self, tree: &FFTree) -> CircleSemiEval {
        let pol0_eval = self.poly0.evaluate(tree);
        let pol1_eval = self.poly1.evaluate(tree);
        CircleSemiEval {
            domain: self.domain,
            poly0_eval: pol0_eval,
            poly1_eval: pol1_eval,
        }
    }
    pub fn evaluate(self, tree: &FFTree) -> CircleEvaluation {
        self.semi_evaluate(tree).evaluate()
    }
}

#[test]
fn test_circle_domain_iterator() {
    let domain = CircleDomain::constraint_domain(3);
    for (i, point) in domain.iter().enumerate() {
        if i < 4 {
            assert_eq!(
                point,
                (CirclePointIndex::generator() + CirclePointIndex::subgroup_gen(2) * i).to_point()
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
    let domain = CircleDomain::constraint_domain(3);
    assert_eq!(domain.n_bits(), 3);
    let evaluation = CircleEvaluation::new(domain, (0..8).map(Field::from_u32_unchecked).collect());
    let poly = evaluation
        .clone()
        .interpolate(&FFTree::preprocess(domain.line_domain()));
    let evaluation2 = poly.evaluate(&FFTree::preprocess(domain.line_domain()));
    assert_eq!(evaluation.values, evaluation2.values);
}

#[test]
fn test_interpolate_canonic_eval() {
    let domain = CircleDomain::constraint_domain(3);
    assert_eq!(domain.n_bits(), 3);
    let evaluation = CircleEvaluation::new(domain, (0..8).map(Field::from_u32_unchecked).collect());
    let poly = evaluation.interpolate(&FFTree::preprocess(domain.line_domain()));
    for (i, point) in domain.iter().enumerate() {
        assert_eq!(
            poly.eval_at_point(point),
            Field::from_u32_unchecked(i as u32)
        );
    }
}

#[test]
fn test_interpolate_canonic() {
    let coset = CanonicCoset::new(3);
    let evaluation = CircleEvaluation::new_canonical_ordered(
        coset,
        (0..8).map(Field::from_u32_unchecked).collect(),
    );
    let domain = evaluation.domain.line_domain();
    let poly = evaluation.interpolate(&FFTree::preprocess(domain));
    for (i, point) in Coset::odds(3).iter().enumerate() {
        assert_eq!(
            poly.eval_at_point(point),
            Field::from_u32_unchecked(i as u32)
        );
    }
}

#[test]
fn test_mixed_degree_example() {
    use crate::core::constraints::EvalByEvaluation;
    use crate::core::constraints::PolyOracle;
    use crate::core::fft::FFTree;
    use crate::core::poly::circle::CanonicCoset;

    let n_bits = 4;

    // Compute domains.
    let domain0 = CanonicCoset::new(n_bits);
    let eval_domain0 = domain0.eval_domain(n_bits + 4);
    let domain1 = CanonicCoset::new(n_bits + 2);
    let eval_domain1 = domain1.eval_domain(n_bits + 3);
    let constraint_domain = CircleDomain::constraint_domain(n_bits + 1);

    // Compute values.
    let values1: Vec<_> = (0..(domain1.len() as u32))
        .map(Field::from_u32_unchecked)
        .collect();
    let values0: Vec<_> = values1[1..].iter().step_by(4).map(|x| *x * *x).collect();

    // Extend.
    let trace_eval0 = CircleEvaluation::new_canonical_ordered(domain0, values0);
    let trace_line_domain0 = trace_eval0.domain.line_domain();
    let eval0 = trace_eval0
        .interpolate(&FFTree::preprocess(trace_line_domain0))
        .extend(eval_domain0)
        .evaluate(&FFTree::preprocess(eval_domain0.line_domain()));
    let trace_eval1 = CircleEvaluation::new_canonical_ordered(domain1, values1);
    let trace_line_domain1 = trace_eval1.domain.line_domain();
    let eval1 = trace_eval1
        .interpolate(&FFTree::preprocess(trace_line_domain1))
        .extend(eval_domain1)
        .evaluate(&FFTree::preprocess(eval_domain1.line_domain()));

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
