use crate::core::{
    constraints::{EvalByEvaluation, EvalByPoly},
    curve::{CanonicCoset, CIRCLE_GEN},
    fft::{Evaluation, FFTree},
};
use crate::fibonacci::TraceInfo;

pub mod core;
pub mod fibonacci;

fn main() {
    // Parameters.
    let trace_info = TraceInfo::new(16);
    let trace_coset = trace_info.trace_coset;
    let trace_tree = FFTree::preprocess(trace_coset);
    let constraint_coset = CanonicCoset::new(trace_coset.n_bits + 1);
    let constraint_tree = FFTree::preprocess(constraint_coset);

    // Generate trace.
    let trace = trace_info.get_trace();

    // Extend trace.
    let trace_poly = trace_tree.ifft(trace);
    let trace_extension = constraint_tree.fft(trace_poly.extend(constraint_coset));

    // Compute quotient on other cosets.
    let mut quotient_eval = Vec::with_capacity(constraint_coset.size());
    for (i, _point) in constraint_coset.iter().enumerate() {
        quotient_eval.push(trace_info.eval_quotient(EvalByEvaluation {
            index: i,
            eval: &trace_extension,
        }));
    }

    // Get quotient as a polynomial.
    let quotient_poly = constraint_tree.ifft(Evaluation::new(constraint_coset, quotient_eval));

    // Check out of domain.
    let point = CIRCLE_GEN;
    let expected = trace_info.eval_quotient(EvalByPoly {
        point,
        poly: &trace_poly,
    });
    let sampled = quotient_poly.eval(point);
    assert!(
        expected == sampled,
        "expected: {expected:?}, got: {sampled:?}",
    );
    println!("Verified successfully!");
}
