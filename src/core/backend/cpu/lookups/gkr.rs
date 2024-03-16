use num_traits::One;

use crate::core::backend::CPUBackend;
use crate::core::fields::qm31::SecureField;
use crate::core::lookups::gkr::GkrOps;

impl GkrOps for CPUBackend {
    fn gen_eq_evals(y: &[SecureField]) -> Self::Column {
        match y {
            [] => vec![SecureField::one()],
            &[y_1] => vec![SecureField::one() - y_1, y_1],
            &[y_j, ref y @ ..] => {
                let mut c = Self::gen_eq_evals(y);
                for i in 0..c.len() {
                    // `lhs[i] = eq(0, y_j) * c[i]`
                    // `rhs[i] = eq(1, y_j) * c[i]`
                    let tmp = c[i] * y_j;
                    c.push(tmp);
                    c[i] -= tmp;
                }
                c
            }
        }
    }
}
