use super::{
    circle::{CirclePoint, CirclePointIndex, Coset},
    fields::m31::Field,
    poly::circle::{CircleDomain, CircleEvaluation, CirclePoly},
};

// Evaluates a vanishing polynomial of the coset at a point.
pub fn coset_vanishing(coset: Coset, mut p: CirclePoint) -> Field {
    // Doubling a point `n_bits / 2` times and taking the x coordinate is essentially evaluating
    // a polynomial in x of degree `2**(n_bits-1)`. If the entire `2**n_bits` points of the coset
    // are roots (i.e. yield 0), then this is a vanishing polynomial of these points.

    // Rotating the coset -coset.initial + step / 2 yields a canonic coset:
    // `step/2 + <step>.`
    // Doubling this coset n_bits - 1 times yields the coset +-G_4.
    // th polynomial x vanishes on these points.
    //   X
    // .   .
    //   X
    p = p - coset.initial + coset.step_size.half().to_point();
    let mut x = p.x;

    // The formula for the x coordinate of the double of a point is 2x^2-1.
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
        let d = self.eval.domain.find(i).expect("Not in domain");
        self.eval.values[d]
    }
}

#[test]
fn test_vanishing() {
    let cosets = [
        Coset::half_odds(5),
        Coset::odds(5),
        Coset::new(CirclePointIndex::zero(), 5),
        Coset::half_odds(5).conjugate(),
    ];
    for c0 in cosets.iter() {
        for el in c0.iter() {
            assert_eq!(coset_vanishing(*c0, el), Field::zero());
            for c1 in cosets.iter() {
                if c0 == c1 {
                    continue;
                }
                assert_ne!(coset_vanishing(*c1, el), Field::zero());
            }
        }
    }
}
