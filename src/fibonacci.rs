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
        let constraint_eval_domain = CircleDomain::constraint_domain(n_bits + 1);
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
        let denom = coset_vanishing(self.trace_coset.coset(), trace.point());
        num / denom
    }
}

#[test]
fn test_constraint_on_trace() {
    use crate::core::constraints::EvalByEvaluation;

    let fib = Fibonacci::new(3);
    let trace = fib.get_trace();

    for p_ind in fib.constraint_coset.iter_indices().take(6) {
        let res = fib.eval_constraint(EvalByEvaluation {
            offset: p_ind,
            eval: &trace,
        });
        assert_eq!(res, Field::zero());
    }
}

#[test]
fn test_quotient_is_low_degree() {
    use crate::core::circle::CIRCLE_GEN;
    use crate::core::constraints::EvalByEvaluation;
    use crate::core::constraints::EvalByPoly;
    use crate::core::fft::FFTree;

    let fib = Fibonacci::new(5);

    let trace = fib.get_trace();
    let trace_poly = trace.interpolate(&FFTree::preprocess(fib.trace_coset.line_domain()));

    let extended_evaluation = trace_poly
        .extend(fib.eval_domain)
        .evaluate(&FFTree::preprocess(fib.eval_domain.line_domain()));

    // Compute quotient on other cosets.
    let mut quotient_values = Vec::with_capacity(fib.constraint_eval_domain.len());
    for p_ind in fib.constraint_eval_domain.iter_indices() {
        quotient_values.push(fib.eval_quotient(EvalByEvaluation {
            offset: p_ind,
            eval: &extended_evaluation,
        }));
    }
    let quotient_eval = CircleEvaluation::new(fib.constraint_eval_domain, quotient_values);
    let quotient_poly = quotient_eval.interpolate(&FFTree::preprocess(
        fib.constraint_eval_domain.line_domain(),
    ));
    assert_eq!(
        quotient_poly.eval_at_point(-CIRCLE_GEN),
        fib.eval_quotient(EvalByPoly {
            point: -CIRCLE_GEN,
            poly: &trace_poly
        })
    );
}
