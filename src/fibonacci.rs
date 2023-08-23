use crate::core::{
    constraints::{domain_poly_eval, point_excluder, TraceOracle},
    curve::CanonicCoset,
    fft::Evaluation,
    field::Field,
};

pub struct TraceInfo {
    pub trace_coset: CanonicCoset,
}
impl TraceInfo {
    pub fn new(n_bits: usize) -> Self {
        Self {
            trace_coset: CanonicCoset::new(n_bits),
        }
    }
    pub fn get_trace(&self) -> Evaluation {
        // Trace.
        let trace_n_bits = self.trace_coset.n_bits;
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

        // Returns as an Evaluation.
        let trace_coset = CanonicCoset::new(trace_n_bits);
        Evaluation::new(trace_coset, trace)
    }
    pub fn eval_constraint(&self, trace: impl TraceOracle) -> Field {
        trace.get_at(0).square() + trace.get_at(1).square() - trace.get_at(2)
    }
    pub fn eval_quotient(&self, trace: impl TraceOracle) -> Field {
        let excluded0 = self.trace_coset.at(self.trace_coset.size() - 2);
        let excluded1 = self.trace_coset.at(self.trace_coset.size() - 1);
        let num = self.eval_constraint(trace)
            * point_excluder(trace.point(), excluded0)
            * point_excluder(trace.point(), excluded1);
        let denom = domain_poly_eval(self.trace_coset, trace.point().x);
        num / denom
    }
}
