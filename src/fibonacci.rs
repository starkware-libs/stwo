use crate::core::{
    circle::CirclePointIndex,
    constraints::{coset_vanishing, point_excluder, PolyOracle},
    fields::m31::Field,
    poly::circle::{CanonicCoset, CircleDomain, CircleEvaluation},
};

pub struct TraceInfo {
    pub trace_domain: CanonicCoset,
    pub evaluation_domain: CircleDomain,
}
impl TraceInfo {
    pub fn new(n_bits: usize) -> Self {
        let evaluation_domain = CircleDomain::canonic_evaluation(n_bits + 1);
        let trace_domain = CanonicCoset::new(n_bits);
        Self {
            trace_domain,
            evaluation_domain,
        }
    }
    pub fn get_trace(&self) -> CircleEvaluation {
        // Trace.
        let mut trace = Vec::with_capacity(self.trace_domain.len());

        // Fill trace with fibonacci squared.
        let mut a = Field::one();
        let mut b = Field::one();
        for _ in 0..self.trace_domain.len() {
            trace.push(a);
            let tmp = a.square() + b.square();
            a = b;
            b = tmp;
        }

        // Returns as a CircleEvaluation.
        CircleEvaluation::new_canonical_ordered(self.trace_domain, trace)
    }
    pub fn eval_constraint(&self, trace: impl PolyOracle) -> Field {
        let step = CirclePointIndex::subgroup_gen(self.trace_domain.n_bits);
        trace.get_at(CirclePointIndex::zero()).square() + trace.get_at(step * 1).square()
            - trace.get_at(step * 2)
    }
    pub fn eval_quotient(&self, trace: impl PolyOracle) -> Field {
        let excluded0 = self.trace_domain.at(self.trace_domain.len() - 2);
        let excluded1 = self.trace_domain.at(self.trace_domain.len() - 1);
        let num = self.eval_constraint(trace)
            * point_excluder(trace.point(), excluded0)
            * point_excluder(trace.point(), excluded1);
        let denom = coset_vanishing(self.trace_domain.coset(), trace.point());
        num / denom
    }
}

#[test]
fn test_constraint_on_trace() {
    use crate::core::constraints::EvalByEvaluation;

    let trace_info = TraceInfo::new(3);
    let trace_domain = trace_info.trace_domain;
    let trace = trace_info.get_trace();

    for p_ind in trace_domain.iter_indices().take(6) {
        let res = trace_info.eval_constraint(EvalByEvaluation {
            domain: trace.domain,
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

    let trace_info = TraceInfo::new(5);
    let trace_domain = trace_info.trace_domain;
    let evaluation_domain = trace_info.evaluation_domain;

    let trace = trace_info.get_trace();
    let trace_poly = trace.interpolate(&FFTree::preprocess(
        trace_domain.circle_domain().line_domain(),
    ));

    let extended_evaluation = trace_poly
        .extend(evaluation_domain)
        .evaluate(&FFTree::preprocess(evaluation_domain.line_domain()));

    // Compute quotient on other cosets.
    let mut quotient_values = Vec::with_capacity(evaluation_domain.len());
    for p_ind in evaluation_domain.iter_indices() {
        quotient_values.push(trace_info.eval_quotient(EvalByEvaluation {
            domain: evaluation_domain,
            offset: p_ind,
            eval: &extended_evaluation,
        }));
    }
    let quotient_eval = CircleEvaluation::new(evaluation_domain, quotient_values);
    let quotient_poly =
        quotient_eval.interpolate(&FFTree::preprocess(evaluation_domain.line_domain()));
    assert_eq!(
        quotient_poly.eval_at_point(-CIRCLE_GEN),
        trace_info.eval_quotient(EvalByPoly {
            point: -CIRCLE_GEN,
            poly: &trace_poly
        })
    );
}
