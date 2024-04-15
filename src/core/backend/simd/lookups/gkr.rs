use cpu::lookups::gkr::gen_eq_evals as cpu_gen_eq_evals;
use num_traits::{One, Zero};

use crate::core::backend::simd::column::SecureFieldVec;
use crate::core::backend::simd::m31::{LOG_N_LANES, N_LANES};
use crate::core::backend::simd::qm31::PackedSecureField;
use crate::core::backend::simd::SimdBackend;
use crate::core::backend::{cpu, Column};
use crate::core::fields::qm31::SecureField;
use crate::core::lookups::gkr::GkrOps;
use crate::core::lookups::mle::Mle;
use crate::core::lookups::utils::eq;

impl GkrOps for SimdBackend {
    fn gen_eq_evals(y: &[SecureField]) -> Mle<Self, SecureField> {
        Mle::new(match y {
            [] => [SecureField::one()].into_iter().collect(),
            &[y_1, ref y @ ..] => gen_eq_evals(y, eq(&[y_1], &[SecureField::zero()])),
        })
    }
}

/// Computes `eq(x, y) * v` for all `x` in `{0, 1}^n`.
///
/// Values are returned in bit-reversed order.
#[allow(clippy::uninit_vec)]
fn gen_eq_evals(y: &[SecureField], v: SecureField) -> SecureFieldVec {
    if y.len() < LOG_N_LANES {
        return cpu_gen_eq_evals(y, v).into_iter().collect();
    }

    // Start DP with CPU backend to prevent dealing with instances smaller than [PackedSecureField].
    let (y_initial, y_rem) = y.split_last_chunk::<LOG_N_LANES>().unwrap();
    let initial = SecureFieldVec::from_iter(cpu_gen_eq_evals(y_initial, v));
    assert_eq!(initial.len(), N_LANES);

    let packed_len = 1 << y_rem.len();
    let mut data = initial.data;

    data.reserve(packed_len - data.len());
    unsafe { data.set_len(packed_len) };

    for (i, &y_j) in y_rem.iter().rev().enumerate() {
        let y_j = PackedSecureField::broadcast(y_j);

        let (lhs_evals, rhs_evals) = data.split_at_mut(1 << i);

        for i in 0..1 << i {
            let rhs_eval = lhs_evals[i] * y_j;
            lhs_evals[i] -= rhs_eval;
            rhs_evals[i] = rhs_eval;
        }
    }

    let length = packed_len * N_LANES;
    SecureFieldVec { data, length }
}
