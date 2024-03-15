use std::cmp::Reverse;
use std::collections::BTreeMap;

use itertools::{izip, multiunzip, Itertools};

use crate::core::backend::CPUBackend;
use crate::core::circle::CirclePoint;
use crate::core::fields::m31::BaseField;
use crate::core::fields::qm31::SecureField;
use crate::core::fri::SparseCircleEvaluation;
use crate::core::poly::circle::{
    CanonicCoset, CircleDomain, CircleEvaluation, PolyOps, SecureEvaluation,
};
use crate::core::poly::BitReversedOrder;
use crate::core::prover::VerificationError;
use crate::core::queries::SparseSubCircleDomain;

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
        samples: &[ColumnSampleBatch],
    ) -> SecureEvaluation<Self>;
}

/// A batch of column samplings at a point.
pub struct ColumnSampleBatch {
    /// The point at which the columns are sampled.
    pub point: CirclePoint<SecureField>,
    /// The sampled column indices and their values at the point.
    pub column_indices_and_values: Vec<(usize, SecureField)>,
}
impl ColumnSampleBatch {
    /// Groups column opening by opening point.
    /// # Arguments
    /// opening: For each column, a vector of samples.
    pub fn new(samples: &[&Vec<PointSample>]) -> Vec<Self> {
        samples
            .iter()
            .enumerate()
            .flat_map(|(column_index, samples)| {
                samples.iter().map(move |opening| (column_index, opening))
            })
            .group_by(|(_, opening)| opening.point)
            .into_iter()
            .map(|(point, column_samples)| Self {
                point,
                column_indices_and_values: column_samples
                    .map(|(column_index, opening)| (column_index, opening.value))
                    .collect(),
            })
            .collect()
    }
}

pub struct PointSample {
    pub point: CirclePoint<SecureField>,
    pub value: SecureField,
}

pub fn compute_fri_quotients<B: QuotientOps>(
    columns: &[&CircleEvaluation<B, BaseField, BitReversedOrder>],
    samples: &[Vec<PointSample>],
    random_coeff: SecureField,
) -> Vec<SecureEvaluation<B>> {
    izip!(columns, samples)
        .group_by(|(c, _)| c.domain.log_size())
        .into_iter()
        .sorted_by_key(|(log_size, _)| Reverse(*log_size))
        .map(|(log_size, tuples)| {
            let (columns, samples): (Vec<_>, Vec<_>) = multiunzip(tuples);
            let domain = CanonicCoset::new(log_size).circle_domain();
            // TODO: slice.
            let batched_samples = ColumnSampleBatch::new(&samples);
            B::accumulate_quotients(domain, &columns, random_coeff, &batched_samples)
        })
        .collect()
}

pub fn fri_answers(
    column_log_sizes: Vec<u32>,
    samples: &[Vec<PointSample>],
    random_coeff: SecureField,
    query_domain_per_log_size: BTreeMap<u32, SparseSubCircleDomain>,
    queried_values_per_column: &[Vec<BaseField>],
) -> Result<Vec<SparseCircleEvaluation<SecureField>>, VerificationError> {
    izip!(column_log_sizes, samples, queried_values_per_column)
        .group_by(|(c, ..)| *c)
        .into_iter()
        .sorted_by_key(|(log_size, _)| Reverse(*log_size))
        .map(|(log_size, tuples)| {
            let (_, samples, queried_valued_per_column): (Vec<_>, Vec<_>, Vec<_>) =
                multiunzip(tuples);
            fri_answers_for_log_size(
                log_size,
                &samples,
                random_coeff,
                &query_domain_per_log_size[&log_size],
                &queried_valued_per_column,
            )
        })
        .collect()
}

pub fn fri_answers_for_log_size(
    log_size: u32,
    samples: &[&Vec<PointSample>],
    random_coeff: SecureField,
    query_domain: &SparseSubCircleDomain,
    queried_values_per_column: &[&Vec<BaseField>],
) -> Result<SparseCircleEvaluation<SecureField>, VerificationError> {
    let commitment_domain = CanonicCoset::new(log_size).circle_domain();
    let batched_samples = ColumnSampleBatch::new(samples);
    for x in queried_values_per_column {
        if x.len() != query_domain.flatten().len() {
            return Err(VerificationError::InvalidStructure);
        }
    }
    let mut queried_values_per_column = queried_values_per_column
        .iter()
        .map(|q| q.iter())
        .collect_vec();

    let res = SparseCircleEvaluation::new(
        query_domain
            .iter()
            .map(|subdomain| {
                let domain = subdomain.to_circle_domain(&commitment_domain);
                let column_evals = queried_values_per_column
                    .iter_mut()
                    .map(|q| {
                        CircleEvaluation::new(domain, q.take(domain.size()).copied().collect_vec())
                    })
                    .collect_vec();
                CPUBackend::accumulate_quotients(
                    domain,
                    &column_evals.iter().collect_vec(),
                    random_coeff,
                    &batched_samples,
                )
                .to_cpu()
            })
            .collect(),
    );
    if !queried_values_per_column.iter().all(|x| x.is_empty()) {
        return Err(VerificationError::InvalidStructure);
    }
    Ok(res)
}

#[cfg(test)]
mod tests {
    use crate::core::backend::cpu::{CPUCircleEvaluation, CPUCirclePoly};
    use crate::core::circle::SECURE_FIELD_CIRCLE_GEN;
    use crate::core::commitment_scheme::quotients::{compute_fri_quotients, PointSample};
    use crate::core::poly::circle::CanonicCoset;
    use crate::{m31, qm31};

    #[test]
    fn test_quotients_are_low_degree() {
        const LOG_SIZE: u32 = 7;
        let polynomial = CPUCirclePoly::new((0..1 << LOG_SIZE).map(|i| m31!(i)).collect());
        let eval_domain = CanonicCoset::new(LOG_SIZE + 1).circle_domain();
        let eval = polynomial.evaluate(eval_domain);
        let point = SECURE_FIELD_CIRCLE_GEN;
        let value = polynomial.eval_at_point(point);
        let coeff = qm31!(1, 2, 3, 4);
        let quot_eval =
            compute_fri_quotients(&[&eval], &[vec![PointSample { point, value }]], coeff)
                .pop()
                .unwrap();
        let quot_poly_base_field =
            CPUCircleEvaluation::new(eval_domain, quot_eval.values.columns[0].clone())
                .interpolate();
        assert!(quot_poly_base_field.is_in_fft_space(LOG_SIZE));
    }
}
