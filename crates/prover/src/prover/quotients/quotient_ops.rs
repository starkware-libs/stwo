use std::cmp::Reverse;
use std::iter::zip;

use itertools::Itertools;
use tracing::{span, Level};

use crate::core::fields::m31::BaseField;
use crate::core::fields::qm31::SecureField;
use crate::core::pcs::quotients::{ColumnSampleBatch, PointSample};
use crate::core::poly::circle::{CanonicCoset, CircleDomain};
use crate::prover::poly::circle::{CircleEvaluation, PolyOps, SecureEvaluation};
use crate::prover::poly::BitReversedOrder;

pub trait QuotientOps: PolyOps {
    /// Accumulates the quotients of the columns at the given domain.
    /// For a column f(x), and a point sample (p,v), the quotient is
    ///   (f(x) - V0(x))/V1(x)
    /// where V0(p)=v, V0(conj(p))=conj(v), and V1 is a vanishing polynomial for p,conj(p).
    /// This ensures that if f(p)=v, then the quotient is a polynomial.
    /// The result is a linear combination of the quotients using powers of random_coeff.
    fn accumulate_quotients(
        domain: CircleDomain,
        columns: &[&CircleEvaluation<Self, BaseField, BitReversedOrder>],
        random_coeff: SecureField,
        sample_batches: &[ColumnSampleBatch],
        log_blowup_factor: u32,
    ) -> SecureEvaluation<Self, BitReversedOrder>;
}

pub fn compute_fri_quotients<B: QuotientOps>(
    columns: &[&CircleEvaluation<B, BaseField, BitReversedOrder>],
    samples: &[Vec<PointSample>],
    random_coeff: SecureField,
    log_blowup_factor: u32,
) -> Vec<SecureEvaluation<B, BitReversedOrder>> {
    let _span = span!(Level::INFO, "Compute FRI quotients", class = "FRIQuotients").entered();
    zip(columns, samples)
        .sorted_by_key(|(c, _)| Reverse(c.domain.log_size()))
        .group_by(|(c, _)| c.domain.log_size())
        .into_iter()
        .map(|(log_size, tuples)| {
            let (columns, samples): (Vec<_>, Vec<_>) = tuples.unzip();
            let domain = CanonicCoset::new(log_size).circle_domain();
            // TODO: slice.
            let sample_batches = ColumnSampleBatch::new_vec(&samples);
            B::accumulate_quotients(
                domain,
                &columns,
                random_coeff,
                &sample_batches,
                log_blowup_factor,
            )
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use crate::core::circle::SECURE_FIELD_CIRCLE_GEN;
    use crate::core::pcs::quotients::PointSample;
    use crate::core::poly::circle::CanonicCoset;
    use crate::prover::backend::cpu::{CpuCircleEvaluation, CpuCirclePoly};
    use crate::prover::quotients::quotient_ops::compute_fri_quotients;
    use crate::{m31, qm31};

    #[test]
    fn test_quotients_are_low_degree() {
        const LOG_SIZE: u32 = 7;
        const LOG_BLOWUP_FACTOR: u32 = 1;
        let polynomial = CpuCirclePoly::new((0..1 << LOG_SIZE).map(|i| m31!(i)).collect());
        let eval_domain = CanonicCoset::new(LOG_SIZE + 1).circle_domain();
        let eval = polynomial.evaluate(eval_domain);
        let point = SECURE_FIELD_CIRCLE_GEN;
        let value = polynomial.eval_at_point(point);
        let coeff = qm31!(1, 2, 3, 4);
        let quot_eval = compute_fri_quotients(
            &[&eval],
            &[vec![PointSample { point, value }]],
            coeff,
            LOG_BLOWUP_FACTOR,
        )
        .pop()
        .unwrap();
        let quot_poly_base_field =
            CpuCircleEvaluation::new(eval_domain, quot_eval.values.columns[0].clone())
                .interpolate();
        assert!(quot_poly_base_field.is_in_fri_space(LOG_SIZE));
    }
}
