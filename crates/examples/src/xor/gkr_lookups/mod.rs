pub mod accumulation;
pub mod mle_eval;
pub mod preprocessed_columns;

#[cfg(test)]
mod test {
    use stwo_prover::core::backend::Column;
    use stwo_prover::core::fields::qm31::SecureField;
    use stwo_prover::core::fields::{ExtensionOf, Field};
    use stwo_prover::core::lookups::mle::{Mle, MleOps};

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
