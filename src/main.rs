// use crate::core::{
//     circle::CIRCLE_GEN,
//     constraints::{EvalByEvaluation, EvalByPoly},
//     fft::FFTree,
//     poly::line::LineDomain,
// };
// use crate::fibonacci::TraceInfo;

pub mod core;
pub mod fibonacci;

fn main() {
    // // Parameters.
    // let trace_info = TraceInfo::new(16);
    // let trace_domain = trace_info.trace_domain;
    // let evaluation_domain = trace_info.evaluation_domain;
    // let trace_tree = FFTree::preprocess(LineDomain::canonic(trace_domain.n_bits - 1));
    // let evaluation_tree = FFTree::preprocess(LineDomain::canonic(evaluation_domain.n_bits - 1));

    // // Generate trace.
    // let trace = trace_info.get_trace();

    // // Extend trace.
    // let trace_poly = trace.interpolate(trace_tree);
    // let trace_extension = constraint_tree.fft(trace_poly.extend(evaluation_domain));

    // // Compute quotient on other cosets.
    // let mut quotient_eval = Vec::with_capacity(constraint_coset.size());
    // for (i, _point) in constraint_coset.iter().enumerate() {
    //     quotient_eval.push(trace_info.eval_quotient(EvalByEvaluation {
    //         offset: i,
    //         eval: &trace_extension,
    //     }));
    // }

    // // Get quotient as a polynomial.
    // let quotient_poly = constraint_tree.ifft(Evaluation::new(constraint_coset, quotient_eval));

    // // Check out of domain.
    // let point = CIRCLE_GEN;
    // let expected = trace_info.eval_quotient(EvalByPoly {
    //     point,
    //     poly: &trace_poly,
    // });
    // let sampled = quotient_poly.eval(point);
    // assert!(
    //     expected == sampled,
    //     "expected: {expected:?}, got: {sampled:?}",
    // );
    // println!("Verified successfully!");
}
