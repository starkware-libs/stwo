use super::{
    circle::{CircleIndex, CirclePoint, Coset},
    fields::m31::Field,
    poly::circle::{CircleDomain, CircleEvaluation, CirclePoly},
};

pub fn coset_vanishing(coset: Coset, mut p: CirclePoint) -> Field {
    p = p - coset.initial;
    let mut x = p.y;
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
    fn get_at(&self, i: CircleIndex) -> Field;
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
    fn get_at(&self, i: CircleIndex) -> Field {
        self.poly.eval_at_point(self.point + i.to_point())
    }
}

// TODO: make an iterator instead, so we do all computations beforehand.
#[derive(Copy, Clone)]
pub struct EvalByEvaluation<'a> {
    pub domain: CircleDomain,
    pub offset: CircleIndex,
    pub eval: &'a CircleEvaluation,
}
impl<'a> PolyOracle for EvalByEvaluation<'a> {
    fn point(&self) -> CirclePoint {
        self.offset.to_point()
    }
    fn get_at(&self, mut i: CircleIndex) -> Field {
        i = i + self.offset;

        // Check if it is in the first half.
        if let Some(d) =
            (i - self.domain.half_coset.initial_index).try_div(self.domain.half_coset.step_size)
        {
            let res = self.eval.values[d];
            return res;
        }
        if let Some(d) =
            (i + self.domain.half_coset.initial_index).try_div(-self.domain.half_coset.step_size)
        {
            let res = self.eval.values[self.domain.half_coset.len() + d];
            return res;
        }
        panic!("Not on domain!")
    }
}
