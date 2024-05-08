use num_traits::{One, Zero};

use crate::core::backend::CpuBackend;
use crate::core::fields::qm31::SecureField;
use crate::core::lookups::gkr::GkrOps;
use crate::core::lookups::mle::Mle;
use crate::core::lookups::utils::eq;

impl GkrOps for CpuBackend {
    fn gen_eq_evals(y: &[SecureField]) -> Mle<Self, SecureField> {
        Mle::new(match y {
            [] => vec![SecureField::one()],
            &[y_1, ref y @ ..] => gen_eq_evals(y, eq(&[y_1], &[SecureField::zero()])),
        })
    }
}

/// Computes `eq(x, y) * v` for all `x` in `{0, 1}^n`.
///
/// Values are returned in bit-reversed order.
fn gen_eq_evals(y: &[SecureField], v: SecureField) -> Vec<SecureField> {
    match y {
        [] => vec![v],
        &[y_i, ref y @ ..] => {
            let mut c = gen_eq_evals(y, v);
            for i in 0..c.len() {
                // `lhs[i] = eq(0, y_i) * c[i]`
                // `rhs[i] = eq(1, y_i) * c[i]`
                let tmp = c[i] * y_i;
                c.push(tmp);
                c[i] -= tmp;
            }
            c
        }
    }
}

#[cfg(test)]
mod tests {
    use num_traits::{One, Zero};

    use crate::core::backend::CpuBackend;
    use crate::core::fields::m31::BaseField;
    use crate::core::fields::qm31::SecureField;
    use crate::core::lookups::gkr::GkrOps;
    use crate::core::lookups::utils::eq;

    #[test]
    fn gen_eq_evals() {
        let zero = SecureField::zero();
        let one = SecureField::one();
        let y = [7, 4, 3].map(|v| BaseField::from(v).into());

        let eq_evals = CpuBackend::gen_eq_evals(&y);

        assert_eq!(
            **eq_evals,
            [
                eq(&[zero, zero, zero], &y),
                eq(&[zero, zero, one], &y),
                eq(&[zero, one, zero], &y),
                eq(&[zero, one, one], &y)
            ]
        );
    }
}
