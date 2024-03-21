// use std::time::Instant;

use crate::core::air::evaluation::SecureColumn;
use crate::core::backend::avx512::cm31::PackedCM31;
use crate::core::backend::avx512::qm31::PackedQM31;
use crate::core::backend::avx512::{AVX512Backend, BaseFieldVec, K_BLOCK_SIZE};
use crate::core::backend::CPUBackend;
use crate::core::fields::qm31::SecureField;
use crate::core::lookups::gkr::GkrOps;

impl GkrOps for AVX512Backend {
    type EqEvals = SecureColumn<Self>;

    fn gen_eq_evals(y: &[SecureField]) -> SecureColumn<Self> {
        match y {
            // Handle all base cases where the number of evals don't overflow a [PackedSecureField].
            // These base cases are offloaded to the [CPUBackend] for simplification with negligible
            // performance impact.
            [_, _, _, _] | [_, _, _] | [_, _] | [_] | [] => {
                CPUBackend::gen_eq_evals(y).into_iter().collect()
            }
            &[ref y_rem @ .., y_4, y_3, y_2, y_1] => {
                let initial = Self::gen_eq_evals(&[y_4, y_3, y_2, y_1]);
                assert_eq!(initial.len() / K_BLOCK_SIZE, 1);

                let packed_len = 1 << y_rem.len();
                let [mut col0, mut col1, mut col2, mut col3] = initial.cols.map(|c| c.data);

                // Reserve all the capacity we need upfront.
                col0.reserve(packed_len - col0.len());
                col1.reserve(packed_len - col0.len());
                col2.reserve(packed_len - col0.len());
                col3.reserve(packed_len - col0.len());

                unsafe {
                    col0.set_len(packed_len);
                    col1.set_len(packed_len);
                    col2.set_len(packed_len);
                    col3.set_len(packed_len);
                }

                for (i, &y_j) in y_rem.iter().rev().enumerate() {
                    let y_j = PackedQM31::broadcast(y_j);

                    let (col0_lhs, col0_rhs) = col0.split_at_mut(1 << i);
                    let (col1_lhs, col1_rhs) = col1.split_at_mut(1 << i);
                    let (col2_lhs, col2_rhs) = col2.split_at_mut(1 << i);
                    let (col3_lhs, col3_rhs) = col3.split_at_mut(1 << i);

                    for i in 0..1 << i {
                        let PackedQM31([PackedCM31([rhs0, rhs1]), PackedCM31([rhs2, rhs3])]) =
                            PackedQM31([
                                PackedCM31([col0_lhs[i], col1_lhs[i]]),
                                PackedCM31([col2_lhs[i], col3_lhs[i]]),
                            ]) * y_j;

                        col0_lhs[i] -= rhs0;
                        col1_lhs[i] -= rhs1;
                        col2_lhs[i] -= rhs2;
                        col3_lhs[i] -= rhs3;

                        col0_rhs[i] = rhs0;
                        col1_rhs[i] = rhs1;
                        col2_rhs[i] = rhs2;
                        col3_rhs[i] = rhs3;
                    }
                }

                let length = packed_len * K_BLOCK_SIZE;

                SecureColumn {
                    cols: [
                        BaseFieldVec { data: col0, length },
                        BaseFieldVec { data: col1, length },
                        BaseFieldVec { data: col2, length },
                        BaseFieldVec { data: col3, length },
                    ],
                }
            }
        }
    }
}
