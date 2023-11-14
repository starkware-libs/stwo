use crate::core::{
    circle::{Coset, CirclePointIndex},
    constraints::{coset_vanishing, point_excluder, point_vanishing, PolyOracle},
    fields::m31::Field,
    poly::circle::{CanonicCoset, CircleDomain, CircleEvaluation},
};
use num_traits::One;

pub struct Fibonacci {
    pub trace_coset: CanonicCoset,
    pub eval_domain: CircleDomain,
    pub constraint_coset: Coset,
    pub constraint_eval_domain: CircleDomain,
    pub claim: Field,
}
impl Fibonacci {
    pub fn new(n_bits: usize, claim: Field) -> Self {
        let trace_coset = CanonicCoset::new(n_bits);
        let eval_domain = trace_coset.eval_domain(n_bits + 1);
        let constraint_coset = Coset::subgroup(n_bits);
        let constraint_eval_domain = CircleDomain::constraint_domain(n_bits + 1);
        Self {
            trace_coset,
            eval_domain,
            constraint_coset,
            constraint_eval_domain,
            claim,
        }
    }
    pub fn get_trace(&self) -> CircleEvaluation {
        // Trace.
        let mut trace = Vec::with_capacity(self.trace_coset.len());

        // Fill trace with fibonacci squared.
        let mut a = Field::one();
        let mut b = Field::one();
        for _ in 0..self.trace_coset.len() {
            trace.push(a);
            let tmp = a.square() + b.square();
            a = b;
            b = tmp;
        }

        // Returns as a CircleEvaluation.
        CircleEvaluation::new_canonical_ordered(self.trace_coset, trace)
    }
    pub fn eval_step_constraint(&self, trace: &impl PolyOracle, eval_point_index: CirclePointIndex) -> Field {
        trace
            .get_at(self.trace_coset.index_at(0), eval_point_index)
            .square()
            + trace
                .get_at(self.trace_coset.index_at(1), eval_point_index)
                .square()
            - trace.get_at(self.trace_coset.index_at(2), eval_point_index)
    }

    pub fn eval_step_quotient(&self, trace: &impl PolyOracle, eval_point_index: CirclePointIndex) -> Field {
        let excluded0 = self.constraint_coset.at(self.constraint_coset.len() - 2);
        let excluded1 = self.constraint_coset.at(self.constraint_coset.len() - 1);
        let num = self.eval_step_constraint(trace, eval_point_index)
            * point_excluder(excluded0, eval_point_index.to_point())
            * point_excluder(excluded1, eval_point_index.to_point());
        let denom = coset_vanishing(self.constraint_coset, eval_point_index.to_point());
        num / denom
    }

    pub fn eval_boundary_constraint(
        &self,
        trace: &impl PolyOracle,
        eval_point_index: CirclePointIndex,
        value: Field,
    ) -> Field {
        trace.get_at(self.trace_coset.index_at(0), eval_point_index) - value
    }

    pub fn eval_boundary_quotient(
        &self,
        trace: &impl PolyOracle,
        eval_point_index: CirclePointIndex,
        point_index: usize,
        value: Field,
    ) -> Field {
        let num = self.eval_boundary_constraint(trace, eval_point_index, value);
        let denom = point_vanishing(self.constraint_coset.at(point_index), eval_point_index.to_point());
        num / denom
    }

    pub fn eval_quotient(
        &self,
        random_coeff: Field,
        trace: &impl PolyOracle,
        eval_point_index: CirclePointIndex,
    ) -> Field {
        let mut quotient = random_coeff.pow(0) * self.eval_step_quotient(trace, eval_point_index);
        quotient +=
            random_coeff.pow(1) * self.eval_boundary_quotient(trace, eval_point_index, 0, Field::one());
        quotient += random_coeff.pow(2)
            * self.eval_boundary_quotient(
                trace,
                eval_point_index,
                self.constraint_coset.len() - 1,
                self.claim,
            );
        quotient
    }
}

#[test]
fn test_constraint_on_trace() {
    use num_traits::Zero;

    let fib = Fibonacci::new(3, Field::from_u32_unchecked(1056169651));
    let trace = fib.get_trace();

    // Assert that the step constraint is satisfied on the trace.
    for p_ind in fib
        .constraint_coset
        .iter_indices()
        .take(fib.constraint_coset.len() - 2)
    {
        let res = fib.eval_step_constraint(&trace, p_ind);
        assert_eq!(res, Field::zero());
    }

    // Assert that the first trace value is 1.
    assert_eq!(
        fib.eval_boundary_constraint(&trace, fib.constraint_coset.index_at(0), Field::one()),
        Field::zero()
    );

    // Assert that the last trace value is the fibonacci claim.
    assert_eq!(
        fib.eval_boundary_constraint(
            &trace,
            fib.constraint_coset.index_at(fib.constraint_coset.len() - 1),
            fib.claim
        ),
        Field::zero()
    );
}

#[test]
fn test_quotient_is_low_degree() {
    use crate::core::circle::CirclePointIndex;

    let fib = Fibonacci::new(5, Field::from_u32_unchecked(443693538));
    let trace = fib.get_trace();
    let trace_poly = trace.interpolate();

    let extended_evaluation = trace_poly.clone().evaluate(fib.eval_domain);

    // TODO(ShaharS), Change to a channel implementation to retrieve the random coefficients from extension field.
    let random_coeff = Field::from_u32_unchecked(2213980);

    // Compute quotient on the evaluation domain.
    let mut quotient_values = Vec::with_capacity(fib.constraint_eval_domain.len());
    for p_ind in fib.constraint_eval_domain.iter_indices() {
        quotient_values.push(fib.eval_quotient(
            random_coeff,
            &extended_evaluation,
            p_ind,
        ));
    }
    let quotient_eval = CircleEvaluation::new(fib.constraint_eval_domain, quotient_values);
    // Interpolate the poly. The the poly is indeed of degree lower than the size of eval_domain,
    // then it should interpolate correctly.
    let quotient_poly = quotient_eval.interpolate();

    // Evaluate this polynomial at another point, out of eval_domain and compare to what we expect.
    let point_index = CirclePointIndex::generator() * 2;
    assert!(fib.constraint_eval_domain.find(point_index).is_none());
    let point = point_index.to_point();

    // Quotient is low degree if it evaluates the same as a low degree interpolation of the trace.
    assert_eq!(
        quotient_poly.eval_at_point(point),
        fib.eval_quotient(
            random_coeff,
            &trace_poly,
            point_index
        )
    );
}
