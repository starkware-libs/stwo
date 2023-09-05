use crate::core::{
    circle::{CircleIndex, CirclePoint, Coset, CosetIterator},
    fft::FFTree,
    field::field::Field,
};

use super::line::{LineDomain, LineEvaluation, LinePoly};

#[derive(Copy, Clone, Debug)]
pub struct CircleDomain {
    pub coset: Coset,
    pub projection_shift: CircleIndex,
    pub projected_line_domain: LineDomain,
}
impl CircleDomain {
    fn new(coset: Coset) -> Self {
        assert!(coset.n_bits > 0);
        let projected_line_domain = LineDomain::canonic(coset.n_bits - 1);
        let projection_shift = coset.step_size.half() - coset.initial_index;
        Self {
            coset,
            projection_shift,
            projected_line_domain,
        }
    }
    pub fn canonic_evaluation(n_bits: usize) -> Self {
        assert!(n_bits > 0);
        Self::new(Coset::new(CircleIndex::generator(), n_bits))
    }
    pub fn deduce_from_extension_domain(extension_domain: CircleDomain, n_bits: usize) -> Self {
        assert!(extension_domain.n_bits() >= n_bits);
        let extension_line_domain = LineDomain::canonic(extension_domain.n_bits() - 1);
        let line_domain = LineDomain::canonic(n_bits - 1);
        let shift_size =
            extension_line_domain.initial_index() - extension_domain.coset.initial_index;
        Self::new(line_domain.associated_coset().shift(-shift_size))
    }

    pub fn iter(&self) -> CosetIterator {
        self.coset.iter()
    }
    pub fn len(&self) -> usize {
        self.coset.len()
    }
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
    pub fn n_bits(&self) -> usize {
        self.coset.n_bits
    }
    pub fn double(&self) -> Self {
        Self::new(self.coset.double())
    }
    pub fn at(&self, index: usize) -> CirclePoint {
        self.coset.at(index)
    }
}

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
    pub fn semi_interpolate(self) -> CircleSemiEval {
        assert!(self.domain.n_bits() > 0);
        let line_domain = LineDomain::canonic(self.domain.n_bits() - 1);
        let shift_size = self.domain.projection_shift;
        let shifted_coset = self.domain.coset.shift(shift_size);
        assert!(shifted_coset == line_domain.associated_coset());

        let mut poly0_values = Vec::with_capacity(line_domain.len());
        let mut poly1_values = Vec::with_capacity(line_domain.len());

        for (i, point) in line_domain
            .associated_coset()
            .iter()
            .take(line_domain.len())
            .enumerate()
        {
            let v0 = self.values[i];
            let v1 = self.values[self.domain.len() - 1 - i];
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
        let line_domain = self.domain.projected_line_domain;
        for (i, point) in line_domain
            .associated_coset()
            .iter()
            .take(line_domain.len())
            .enumerate()
        {
            let v0 = self.poly0_eval.values[i];
            let v1 = self.poly1_eval.values[i];
            values[i] = v0 + v1 * point.y;
            values[self.domain.len() - 1 - i] = v0 - v1 * point.y;
        }
        CircleEvaluation::new(self.domain, values)
    }
}

#[derive(Clone, Debug)]
pub struct CirclePoly {
    pub domain: CircleDomain,
    pub poly0: LinePoly,
    pub poly1: LinePoly,
}
impl CirclePoly {
    pub fn eval_at_point(&self, point: CirclePoint) -> Field {
        let shifted_point = point + self.domain.projection_shift.to_point();
        let v0 = self.poly0.eval_at_point(shifted_point.x);
        let v1 = self.poly1.eval_at_point(shifted_point.x);
        v0 + v1 * shifted_point.y
    }
    pub fn extend(&self, extended_domain: CircleDomain) -> CirclePoly {
        assert!(extended_domain.n_bits() >= self.domain.n_bits());
        assert_eq!(
            extended_domain.projection_shift,
            self.domain.projection_shift
        );

        let extended_line_domain = extended_domain.projected_line_domain;
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
fn test_interpolate_and_eval() {
    let domain = CircleDomain::canonic_evaluation(3);
    assert_eq!(domain.n_bits(), 3);
    let evaluation = CircleEvaluation::new(domain, (0..8).map(Field::from_u32_unchecked).collect());
    let poly = evaluation
        .clone()
        .interpolate(&FFTree::preprocess(domain.projected_line_domain));
    let evaluation2 = poly.evaluate(&FFTree::preprocess(domain.projected_line_domain));
    assert_eq!(evaluation.values, evaluation2.values);
}

#[test]
fn test_interpolate() {
    let domain = CircleDomain::canonic_evaluation(3);
    assert_eq!(domain.n_bits(), 3);
    let evaluation = CircleEvaluation::new(domain, (0..8).map(Field::from_u32_unchecked).collect());
    let poly = evaluation.interpolate(&FFTree::preprocess(domain.projected_line_domain));
    for (i, point) in domain.iter().enumerate() {
        assert_eq!(
            poly.eval_at_point(point),
            Field::from_u32_unchecked(i as u32)
        );
    }
}

#[test]
fn test_domain_deduction() {
    let extended_domain = CircleDomain::canonic_evaluation(5);
    let domain = CircleDomain::deduce_from_extension_domain(extended_domain, 3);
    assert_eq!(domain.n_bits(), 3);
    let evaluation = CircleEvaluation::new(domain, (0..8).map(Field::from_u32_unchecked).collect());
    let poly = evaluation.interpolate(&FFTree::preprocess(domain.projected_line_domain));
    let extended_poly = poly.extend(extended_domain);
    for (i, point) in domain.iter().enumerate() {
        assert_eq!(
            poly.eval_at_point(point),
            extended_poly.eval_at_point(point)
        );
        assert_eq!(
            poly.eval_at_point(point),
            Field::from_u32_unchecked(i as u32)
        );
    }
}
