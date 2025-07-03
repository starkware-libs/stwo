use num_traits::One;
use stwo::core::fields::m31::BaseField;
use stwo::core::poly::circle::CanonicCoset;
use stwo::prover::backend::simd::SimdBackend;
use stwo::prover::backend::{Col, Column};
use stwo::prover::poly::circle::CircleEvaluation;
use stwo::prover::poly::BitReversedOrder;
use stwo_constraint_framework::preprocessed_columns::PreProcessedColumnId;

pub mod accumulation;
pub mod mle_eval;
pub mod preprocessed_columns;

/// A column with `1` at the first position, and `0` elsewhere.
#[derive(Debug, Clone)]
pub struct IsFirst {
    pub log_size: u32,
}
impl IsFirst {
    pub const fn new(log_size: u32) -> Self {
        Self { log_size }
    }

    pub fn gen_column_simd(&self) -> CircleEvaluation<SimdBackend, BaseField, BitReversedOrder> {
        let mut col = Col::<SimdBackend, BaseField>::zeros(1 << self.log_size);
        col.set(0, BaseField::one());
        CircleEvaluation::new(CanonicCoset::new(self.log_size).circle_domain(), col)
    }

    pub fn id(&self) -> PreProcessedColumnId {
        PreProcessedColumnId {
            id: format!("preprocessed_is_first_{}", self.log_size).to_string(),
        }
    }
}

#[cfg(test)]
mod test {
    use stwo::core::fields::qm31::SecureField;
    use stwo::core::fields::{ExtensionOf, Field};
    use stwo::prover::backend::Column;
    use stwo::prover::lookups::mle::{Mle, MleOps};

    /// Evaluates the multilinear polynomial at `point`.
    pub(crate) fn mle_eval_at_point<B, F>(
        evaluation: &Mle<B, F>,
        point: &[SecureField],
    ) -> SecureField
    where
        F: Field,
        SecureField: ExtensionOf<F>,
        B: MleOps<F>,
    {
        pub fn eval(mle_evals: &[SecureField], p: &[SecureField]) -> SecureField {
            match p {
                [] => mle_evals[0],
                &[p_i, ref p @ ..] => {
                    let (lhs, rhs) = mle_evals.split_at(mle_evals.len() / 2);
                    let lhs_eval = eval(lhs, p);
                    let rhs_eval = eval(rhs, p);
                    // Equivalent to `eq(0, p_i) * lhs_eval + eq(1, p_i) * rhs_eval`.
                    p_i * (rhs_eval - lhs_eval) + lhs_eval
                }
            }
        }

        let mle_evals = evaluation
            .clone()
            .into_evals()
            .to_cpu()
            .into_iter()
            .map(|v| v.into())
            .collect::<Vec<_>>();

        eval(&mle_evals, point)
    }
}
