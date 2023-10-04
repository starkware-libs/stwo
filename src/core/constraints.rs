use super::{
    circle::{CirclePoint, CirclePointIndex, Coset},
    fields::m31::Field,
    poly::circle::{CircleDomain, CircleEvaluation, CirclePoly},
};

pub fn coset_vanishing(coset: Coset, mut p: CirclePoint) -> Field {
    // Rotate the point by coset.initial and then by 90 degree.
    p = p - coset.initial + coset.step_size.half().to_point();
    let mut x = p.x;

    // Double it n_bits - 1 times.
    for _ in 0..(coset.n_bits - 1) {
        x = x.square().double() - Field::one();
    }
    x
}

pub fn circle_domain_vanishing(domain: CircleDomain, p: CirclePoint) -> Field {
    coset_vanishing(domain.half_coset, p) * coset_vanishing(domain.half_coset.conjugate(), p)
}

pub fn point_excluder(point: CirclePoint, excluded: CirclePoint) -> Field {
    (point - excluded).x - Field::one()
}

// Utils for computing constraints.
// Oracle to a polynomial constrained to a coset.
pub trait PolyOracle: Copy {
    fn get_at(&self, i: CirclePointIndex) -> Field;
    fn point(&self) -> CirclePoint;
}
#[derive(Copy, Clone)]
pub struct EvalByPoly<'a> {
    pub point: CirclePoint,
    pub poly: &'a CirclePoly,
}
impl<'a> PolyOracle for EvalByPoly<'a> {
    fn point(&self) -> CirclePoint {
        self.point
    }
    fn get_at(&self, i: CirclePointIndex) -> Field {
        self.poly.eval_at_point(self.point + i.to_point())
    }
}

// TODO(spapini): make an iterator instead, so we do all computations beforehand.
#[derive(Copy, Clone)]
pub struct EvalByEvaluation<'a> {
    pub domain: CircleDomain,
    pub offset: CirclePointIndex,
    pub eval: &'a CircleEvaluation,
}
impl<'a> PolyOracle for EvalByEvaluation<'a> {
    fn point(&self) -> CirclePoint {
        self.offset.to_point()
    }
    fn get_at(&self, mut i: CirclePointIndex) -> Field {
        i = i + self.offset;

        // Check if it is in the first half.
        let d = self.domain.find(i).expect("Not in domain");
        self.eval.values[d]
    }
}

#[test]
fn test_vanishing() {
    let coset0 = Coset::twisted(5);
    let coset1 = Coset::odds(5);
    let coset2 = Coset::new(CirclePointIndex::zero(), 5);
    for p in coset0.iter() {
        assert_eq!(coset_vanishing(coset0, p), Field::zero());
        assert_ne!(coset_vanishing(coset1, p), Field::zero());
        assert_ne!(coset_vanishing(coset2, p), Field::zero());
    }
    for p in coset1.iter() {
        assert_ne!(coset_vanishing(coset0, p), Field::zero());
        assert_eq!(coset_vanishing(coset1, p), Field::zero());
        assert_ne!(coset_vanishing(coset2, p), Field::zero());
    }
    for p in coset2.iter() {
        assert_ne!(coset_vanishing(coset0, p), Field::zero());
        assert_ne!(coset_vanishing(coset1, p), Field::zero());
        assert_eq!(coset_vanishing(coset2, p), Field::zero());
    }
}
