use super::{
    curve::{CanonicCoset, CirclePoint, Coset},
    fft::{Evaluation, Polynomial},
    field::Field,
};

/// Computes prod_i (x-Re(g^i)).
pub fn domain_poly_eval(coset: CanonicCoset, mut x: Field) -> Field {
    // 2x^2-1 is always a 2-to-1 transformation for our points.
    // Applying it n_bits times, gives a poly with 2^n roots.
    // The last x we get is -1 (for the point 0-1i) for canonical cosets.
    assert!(coset.n_bits > 0, "Not a good domain");
    for _ in 0..coset.n_bits - 1 {
        x = x.square().double() - Field::one();
    }
    x
}

pub fn point_excluder(point: CirclePoint, excluded: CirclePoint) -> Field {
    (point - excluded).x - Field::one()
}

pub fn subcoset_excluder(mut point: CirclePoint, coset: Coset) -> Field {
    if coset.n_bits == 0 {
        return point_excluder(point, coset.initial);
    }
    point = point - coset.initial;
    for _ in 0..(coset.n_bits - 1) {
        point = point.double();
    }
    point.y
}

// Utils for computing constraints.
pub trait TraceOracle: Copy {
    fn point(&self) -> CirclePoint;
    fn get_at(&self, i: usize) -> Field;
}
#[derive(Copy, Clone)]
pub struct EvalByPoly<'a> {
    pub point: CirclePoint,
    pub poly: &'a Polynomial,
}
impl<'a> TraceOracle for EvalByPoly<'a> {
    fn point(&self) -> CirclePoint {
        self.point
    }
    fn get_at(&self, i: usize) -> Field {
        self.poly
            .eval(self.point + self.poly.coset.step().mul(i as u64))
    }
}

#[derive(Copy, Clone)]
pub struct EvalByEvaluation<'a> {
    pub index: usize,
    pub eval: &'a Evaluation,
}
impl<'a> TraceOracle for EvalByEvaluation<'a> {
    fn point(&self) -> CirclePoint {
        self.eval.coset.at(self.index)
    }
    fn get_at(&self, i: usize) -> Field {
        self.eval.values[(self.index + 2 * i) & (self.eval.values.len() - 1)]
    }
}
