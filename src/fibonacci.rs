use crate::core::{
    constraints::{domain_poly_eval, point_excluder, PolyOracle},
    field::Field,
    poly::circle::{CircleDomain, CircleEvaluation},
};

pub struct TraceInfo {
    pub trace_domain: CircleDomain,
    pub evaluation_domain: CircleDomain,
}
impl TraceInfo {
    pub fn new(n_bits: usize) -> Self {
        let evaluation_domain = CircleDomain::canonic_evaluation(n_bits + 1);
        let trace_domain = CircleDomain::deduce_from_extension_domain(evaluation_domain, n_bits);
        Self {
            trace_domain,
            evaluation_domain,
        }
    }
    pub fn get_trace(&self) -> CircleEvaluation {
        // Trace.
        let trace_n_bits = self.trace_domain.n_bits();
        let n = 1 << trace_n_bits;
        let mut trace = vec![];
        trace.reserve(n);

        // Fill trace with fibonacci squared.
        let mut a = Field::one();
        let mut b = Field::one();
        for _ in 0..n {
            trace.push(a);
            let tmp = a.square() + b.square();
            a = b;
            b = tmp;
        }

        // Returns as a CircleEvaluation.
        CircleEvaluation::new(self.trace_domain, trace)
    }
    pub fn eval_constraint(&self, trace: impl PolyOracle) -> Field {
        debug_assert!(trace.domain().n_bits == self.trace_domain.n_bits());
        trace.get_at(0).square() + trace.get_at(1).square() - trace.get_at(2)
    }
    pub fn eval_quotient(&self, trace: impl PolyOracle) -> Field {
        let excluded0 = self.trace_domain.at(self.trace_domain.len() - 2);
        let excluded1 = self.trace_domain.at(self.trace_domain.len() - 1);
        let num = self.eval_constraint(trace)
            * point_excluder(trace.point(), excluded0)
            * point_excluder(trace.point(), excluded1);
        let denom = domain_poly_eval(self.trace_domain, trace.point());
        num / denom
    }
}
