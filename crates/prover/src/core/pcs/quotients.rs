use std::cmp::Reverse;
use std::collections::BTreeMap;
use std::iter::zip;

use itertools::{izip, multiunzip, Itertools};
use tracing::{span, Level};

use crate::core::backend::cpu::quotients::{accumulate_row_quotients, quotient_constants};
use crate::core::circle::CirclePoint;
use crate::core::fields::m31::BaseField;
use crate::core::fields::qm31::SecureField;
use crate::core::poly::circle::{
    CanonicCoset, CircleDomain, CircleEvaluation, PolyOps, SecureEvaluation,
};
use crate::core::poly::BitReversedOrder;
use crate::core::prover::VerificationError;
use crate::core::utils::bit_reverse_index;
use crate::core::ColumnVec;

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

/// A batch of column samplings at a point.
pub struct ColumnSampleBatch {
    /// The point at which the columns are sampled.
    pub point: CirclePoint<SecureField>,
    /// The sampled column indices and their values at the point.
    pub columns_and_values: Vec<(usize, SecureField)>,
}

impl ColumnSampleBatch {
    /// Groups column samples by sampled point.
    /// # Arguments
    /// samples: For each column, a vector of samples.
    pub fn new_vec(samples: &[&Vec<PointSample>]) -> Vec<Self> {
        // Group samples by point, and create a ColumnSampleBatch for each point.
        // This should keep a stable ordering.
        let mut grouped_samples = BTreeMap::new();
        for (column_index, samples) in samples.iter().enumerate() {
            for sample in samples.iter() {
                grouped_samples
                    .entry(sample.point)
                    .or_insert_with(Vec::new)
                    .push((column_index, sample.value));
            }
        }
        grouped_samples
            .into_iter()
            .map(|(point, columns_and_values)| ColumnSampleBatch {
                point,
                columns_and_values,
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
    log_blowup_factor: u32,
) -> Vec<SecureEvaluation<B, BitReversedOrder>> {
    let _span = span!(Level::INFO, "Compute FRI quotients").entered();
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

pub fn fri_answers(
    column_log_sizes: Vec<u32>,
    samples: &[Vec<PointSample>],
    random_coeff: SecureField,
    query_positions_per_log_size: &BTreeMap<u32, Vec<usize>>,
    queried_values_per_column: &[Vec<BaseField>],
) -> Result<ColumnVec<Vec<SecureField>>, VerificationError> {
    izip!(column_log_sizes, samples, queried_values_per_column)
        .sorted_by_key(|(log_size, ..)| Reverse(*log_size))
        .group_by(|(log_size, ..)| *log_size)
        .into_iter()
        .map(|(log_size, tuples)| {
            let (_, samples, queried_values_per_column): (Vec<_>, Vec<_>, Vec<_>) =
                multiunzip(tuples);
            fri_answers_for_log_size(
                log_size,
                &samples,
                random_coeff,
                &query_positions_per_log_size[&log_size],
                &queried_values_per_column,
            )
        })
        .collect()
}

pub fn fri_answers_for_log_size(
    log_size: u32,
    samples: &[&Vec<PointSample>],
    random_coeff: SecureField,
    query_positions: &[usize],
    queried_values_per_column: &[&Vec<BaseField>],
) -> Result<Vec<SecureField>, VerificationError> {
    for queried_values in queried_values_per_column {
        if queried_values.len() != query_positions.len() {
            return Err(VerificationError::InvalidStructure(
                "Insufficient number of queried values".to_string(),
            ));
        }
    }

    let sample_batches = ColumnSampleBatch::new_vec(samples);
    let quotient_constants = quotient_constants(&sample_batches, random_coeff);
    let commitment_domain = CanonicCoset::new(log_size).circle_domain();
    let mut quotient_evals_at_queries = Vec::new();

    for (row, &query_position) in query_positions.iter().enumerate() {
        let domain_point = commitment_domain.at(bit_reverse_index(query_position, log_size));
        let queried_values_at_row = queried_values_per_column
            .iter()
            .map(|col| col[row])
            .collect_vec();
        quotient_evals_at_queries.push(accumulate_row_quotients(
            &sample_batches,
            &queried_values_at_row,
            &quotient_constants,
            domain_point,
        ));
    }

    Ok(quotient_evals_at_queries)
}

#[cfg(test)]
mod tests {
    use crate::core::backend::cpu::{CpuCircleEvaluation, CpuCirclePoly};
    use crate::core::circle::SECURE_FIELD_CIRCLE_GEN;
    use crate::core::pcs::quotients::{compute_fri_quotients, PointSample};
    use crate::core::poly::circle::CanonicCoset;
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
