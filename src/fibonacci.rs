use crate::core::{
    circle::Coset,
    constraints::{coset_vanishing, point_excluder, PolyOracle},
    fields::m31::Field,
    poly::circle::{CanonicCoset, CircleDomain, CircleEvaluation},
};

pub struct Fibonacci {
    pub trace_coset: CanonicCoset,
    pub eval_domain: CircleDomain,
    pub constraint_coset: Coset,
    pub constraint_eval_domain: CircleDomain,
}
impl Fibonacci {
    pub fn new(n_bits: usize) -> Self {
        let trace_coset = CanonicCoset::new(n_bits);
        let eval_domain = trace_coset.eval_domain(n_bits + 1);
        let constraint_coset = Coset::subgroup(n_bits);
        let constraint_eval_domain = CircleDomain::constraint_eval_domain(n_bits + 1);
        Self {
            trace_coset,
            eval_domain,
            constraint_coset,
            constraint_eval_domain,
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
    pub fn eval_constraint(&self, trace: impl PolyOracle) -> Field {
        trace.get_at(self.trace_coset.index_at(0)).square()
            + trace.get_at(self.trace_coset.index_at(1)).square()
            - trace.get_at(self.trace_coset.index_at(2))
    }
    pub fn eval_quotient(&self, trace: impl PolyOracle) -> Field {
        let excluded0 = self.constraint_coset.at(self.constraint_coset.len() - 2);
        let excluded1 = self.constraint_coset.at(self.constraint_coset.len() - 1);
        let num = self.eval_constraint(trace)
            * point_excluder(trace.point(), excluded0)
            * point_excluder(trace.point(), excluded1);
        let denom = coset_vanishing(self.constraint_coset, trace.point());
        num / denom
    }
}

#[test]
fn test_constraint_on_trace() {
    use crate::core::constraints::EvalByEvaluation;

    let fib = Fibonacci::new(3);
    let trace = fib.get_trace();

    for p_ind in fib
        .constraint_coset
        .iter_indices()
        .take(fib.constraint_coset.len() - 2)
    {
        let res = fib.eval_constraint(EvalByEvaluation {
            offset: p_ind,
            eval: &trace,
        });
        assert_eq!(res, Field::zero());
    }
}

#[test]
fn test_quotient_is_low_degree() {
    use crate::core::circle::CirclePointIndex;
    use crate::core::constraints::EvalByEvaluation;
    use crate::core::constraints::EvalByPoly;

    let fib = Fibonacci::new(5);

    let trace = fib.get_trace();
    let trace_poly = trace.interpolate();

    let extended_evaluation = trace_poly.clone().evaluate(fib.eval_domain);

    // Compute quotient on the evaluation domain.
    let mut quotient_values = Vec::with_capacity(fib.constraint_eval_domain.len());
    for p_ind in fib.constraint_eval_domain.iter_indices() {
        quotient_values.push(fib.eval_quotient(EvalByEvaluation {
            offset: p_ind,
            eval: &extended_evaluation,
        }));
    }
    let quotient_eval = CircleEvaluation::new(fib.constraint_eval_domain, quotient_values);
    // Interpolate the poly. The the poly is indeed of degree lower than the size of eval_domain,
    // then it should interpolate correctly.
    let quotient_poly = quotient_eval.interpolate();

    // Evaluate this polynomial at another point, out of eval_domain and compare to what we expect.
    let point_index = CirclePointIndex::generator() * 2;
    assert!(fib.constraint_eval_domain.find(point_index).is_none());
    let point = point_index.to_point();
    assert_eq!(
        quotient_poly.eval_at_point(point),
        fib.eval_quotient(EvalByPoly {
            point,
            poly: &trace_poly
        })
    );
}
