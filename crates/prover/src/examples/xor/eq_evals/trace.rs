use std::array;

use num_traits::One;
use tracing::instrument;

use crate::constraint_framework::constant_cols::{gen_is_first, gen_is_step_multiple};
use crate::core::backend::simd::column::BaseFieldVec;
use crate::core::backend::simd::m31::PackedM31;
use crate::core::backend::simd::SimdBackend;
use crate::core::backend::Backend;
use crate::core::fields::m31::BaseField;
use crate::core::fields::qm31::SecureField;
use crate::core::fields::secure_column::SECURE_EXTENSION_DEGREE;
use crate::core::lookups::gkr_prover::GkrOps;
use crate::core::poly::circle::{CanonicCoset, CircleEvaluation};
use crate::core::poly::BitReversedOrder;
use crate::core::ColumnVec;

#[instrument(skip_all)]
pub fn gen_evals_trace(
    eval_point: &[SecureField],
) -> ColumnVec<CircleEvaluation<SimdBackend, BaseField, BitReversedOrder>> {
    let eq_evals = SimdBackend::gen_eq_evals(eval_point, SecureField::one()).into_evals();

    // Currently have SecureField eq_evals.
    // Separate into SECURE_EXTENSION_DEGREE many BaseField columns.
    let mut eq_evals_cols: [Vec<PackedM31>; SECURE_EXTENSION_DEGREE] =
        array::from_fn(|_| Vec::new());

    for secure_vec in &eq_evals.data {
        let [v0, v1, v2, v3] = secure_vec.into_packed_m31s();
        eq_evals_cols[0].push(v0);
        eq_evals_cols[1].push(v1);
        eq_evals_cols[2].push(v2);
        eq_evals_cols[3].push(v3);
    }

    let n_variables = eval_point.len();
    let domain = CanonicCoset::new(n_variables as u32).circle_domain();
    let length = domain.size();
    eq_evals_cols
        .map(|col| BaseFieldVec { data: col, length })
        .map(|col| CircleEvaluation::new(domain, col))
        .into()
}

#[instrument]
pub fn gen_constants_trace<B: Backend>(
    n_variables: usize,
) -> Vec<CircleEvaluation<B, BaseField, BitReversedOrder>> {
    let mut constants_trace = Vec::new();

    let log_size = n_variables as u32;
    constants_trace.push(gen_is_first(log_size));

    // TODO: Last constant column actually equal to gen_is_first but makes the prototype easier.
    for log_step in 1..n_variables as u32 {
        constants_trace.push(gen_is_step_multiple(log_size, log_step + 1))
    }

    constants_trace
}
