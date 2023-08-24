use super::{
    circle::{CirclePoint, Coset},
    field::Field,
    poly::circle::{CircleDomain, CircleEvaluation, CirclePoly},
};

pub fn domain_poly_eval(domain: CircleDomain, mut p: CirclePoint) -> Field {
    p = p + domain.projection_shift.to_point();
    let mut x = p.x;
    for _ in 0..domain.n_bits() - 1 {
        x = x.square().double() - Field::one();
    }
    x
}

pub fn point_excluder(point: CirclePoint, excluded: CirclePoint) -> Field {
    (point - excluded).x - Field::one()
}

// pub fn subcoset_excluder(mut point: CirclePoint, coset: Coset) -> Field {
//     if coset.n_bits == 0 {
//         return point_excluder(point, coset.initial);
//     }
//     point = point - coset.initial;
//     for _ in 0..(coset.n_bits - 1) {
//         point = point.double();
//     }
//     point.y
// }

// Utils for computing constraints.
// Oracle to a polynomial constrained to a coset.
pub trait PolyOracle: Copy {
    fn domain(&self) -> Coset;
    fn get_at(&self, i: usize) -> Field;
    fn point(&self) -> CirclePoint {
        self.domain().initial
    }
}
#[derive(Copy, Clone)]
pub struct EvalByPoly<'a> {
    pub domain: Coset,
    pub poly: &'a CirclePoly,
}
impl<'a> PolyOracle for EvalByPoly<'a> {
    fn domain(&self) -> Coset {
        self.domain
    }
    fn get_at(&self, i: usize) -> Field {
        self.poly.eval_at_point(self.domain.at(i))
    }
}

#[derive(Copy, Clone)]
pub struct EvalByEvaluation<'a> {
    pub domain: Coset,
    pub offset: usize,
    pub eval: &'a CircleEvaluation,
}
impl<'a> PolyOracle for EvalByEvaluation<'a> {
    fn domain(&self) -> Coset {
        self.domain
    }
    fn get_at(&self, i: usize) -> Field {
        self.eval.values[(self.offset + i) & (self.eval.values.len() - 1)]
    }
}
