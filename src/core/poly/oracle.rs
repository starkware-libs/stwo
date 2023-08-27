use crate::core::{
    circle::{CircleIndex, CirclePoint, Coset},
    field::Field,
};

use super::{
    circle::{CircleEvaluation, CirclePoly},
    line::LinePoly,
};

// Oracle to a line polynomial.
pub trait LineFunctionOracle: Copy {
    fn get_at(&self, i: CircleIndex) -> Field;
    fn point(&self) -> Field;
}
#[derive(Copy, Clone)]
pub struct LinePolyOracle<'a> {
    pub point: Field,
    pub poly: &'a LinePoly,
}
impl<'a> LineFunctionOracle for LinePolyOracle<'a> {
    fn point(&self) -> Field {
        self.point
    }
    fn get_at(&self, _i: CircleIndex) -> Field {
        todo!()
        // self.poly.eval_at_point((self.point + i.to_point()).x)
    }
}

// Oracle to a circle polynomial.
pub trait CircleFunctionOracle: Copy {
    fn get_at(&self, i: CircleIndex) -> Field;
    fn point(&self) -> CirclePoint;
}
#[derive(Copy, Clone)]
pub struct CirclePolyOracle<'a> {
    pub point: CirclePoint,
    pub poly: &'a CirclePoly,
}
impl<'a> CircleFunctionOracle for CirclePolyOracle<'a> {
    fn point(&self) -> CirclePoint {
        self.point
    }
    fn get_at(&self, i: CircleIndex) -> Field {
        self.poly.eval_at_point(self.point + i.to_point())
    }
}

#[derive(Copy, Clone)]
pub struct CircleEvalOracle<'a> {
    pub domain: Coset,
    pub offset: usize,
    pub eval: &'a CircleEvaluation,
}
impl<'a> CircleFunctionOracle for CircleEvalOracle<'a> {
    fn point(&self) -> CirclePoint {
        self.domain.at(self.offset)
    }
    fn get_at(&self, i: CircleIndex) -> Field {
        assert_eq!(i.0 % self.domain.step_size.0, 0);
        let rel = self.offset + i.0 / self.domain.step_size.0;
        self.eval.values[rel % self.eval.values.len()]
    }
}
