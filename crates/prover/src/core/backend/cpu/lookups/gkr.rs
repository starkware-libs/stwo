use crate::core::backend::CpuBackend;
use crate::core::fields::qm31::SecureField;
use crate::core::lookups::gkr::GkrOps;
use crate::core::lookups::mle::Mle;

impl GkrOps for CpuBackend {
    fn gen_eq_evals(y: &[SecureField], v: SecureField) -> Mle<Self, SecureField> {
        Mle::new(gen_eq_evals(y, v))
    }
}

/// Returns evaluations `eq(x, y) * v` for all `x` in `{0, 1}^n`.
///
/// Evaluations are returned in bit-reversed order.
fn gen_eq_evals(y: &[SecureField], v: SecureField) -> Vec<SecureField> {
    let mut evals = Vec::with_capacity(1 << y.len());
    evals.push(v);

    for &y_i in y.iter().rev() {
        for j in 0..evals.len() {
            // `lhs[j] = eq(0, y_i) * c[i]`
            // `rhs[j] = eq(1, y_i) * c[i]`
            let tmp = evals[j] * y_i;
            evals.push(tmp);
            evals[j] -= tmp;
        }
    }

    evals
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
        let y = [7, 3].map(|v| BaseField::from(v).into());

        let eq_evals = CpuBackend::gen_eq_evals(&y, one);

        assert_eq!(
            **eq_evals,
            [
                eq(&[zero, zero], &y),
                eq(&[zero, one], &y),
                eq(&[one, zero], &y),
                eq(&[one, one], &y),
            ]
        );
    }
}
