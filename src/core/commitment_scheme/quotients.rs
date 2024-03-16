use std::cmp::Reverse;
use std::collections::BTreeMap;

use itertools::{izip, multiunzip, Itertools};
use tracing::{span, Level};

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
    fn accumulate_quotients(
        domain: CircleDomain,
        columns: &[&CircleEvaluation<Self, BaseField, BitReversedOrder>],
        random_coeff: SecureField,
        openings: &[BatchedColumnOpenings],
    ) -> SecureEvaluation<Self>;
}

pub struct BatchedColumnOpenings {
    pub point: CirclePoint<SecureField>,
    pub column_indices_and_values: Vec<(usize, SecureField)>,
}
impl BatchedColumnOpenings {
    /// Groups column opening by opening point.
    /// # Arguments
    /// opening: For each column, a vector of openings.
    pub fn new(openings: &[&Vec<PointOpening>]) -> Vec<Self> {
        openings
            .iter()
            .enumerate()
            .flat_map(|(column_index, openings)| {
                openings.iter().map(move |opening| (column_index, opening))
            })
            .group_by(|(_, opening)| opening.point)
            .into_iter()
            .map(|(point, column_openings)| BatchedColumnOpenings {
                point,
                column_indices_and_values: column_openings
                    .map(|(column_index, opening)| (column_index, opening.value))
                    .collect(),
            })
            .collect()
    }
}

pub struct PointOpening {
    pub point: CirclePoint<SecureField>,
    pub value: SecureField,
}

pub fn compute_fri_quotients<B: QuotientOps>(
    columns: &[&CircleEvaluation<B, BaseField, BitReversedOrder>],
    openings: &[Vec<PointOpening>],
    random_coeff: SecureField,
) -> Vec<SecureEvaluation<B>> {
    let _span = span!(Level::INFO, "Compute FRI quotients").entered();
    izip!(columns, openings)
        .group_by(|(c, _)| c.domain.log_size())
        .into_iter()
        .sorted_by_key(|(log_size, _)| Reverse(*log_size))
        .map(|(log_size, tuples)| {
            let (columns, openings): (Vec<_>, Vec<_>) = multiunzip(tuples);
            let domain = CanonicCoset::new(log_size).circle_domain();
            // TODO: slice.
            let batched_openings = BatchedColumnOpenings::new(&openings);
            B::accumulate_quotients(domain, &columns, random_coeff, &batched_openings)
        })
        .collect()
}

pub fn fri_answers(
    column_log_sizes: Vec<u32>,
    openings: &[Vec<PointOpening>],
    random_coeff: SecureField,
    query_domain_per_log_size: BTreeMap<u32, SparseSubCircleDomain>,
    queried_values_per_column: &[Vec<BaseField>],
) -> Result<Vec<SparseCircleEvaluation<SecureField>>, VerificationError> {
    izip!(column_log_sizes, openings, queried_values_per_column)
        .group_by(|(c, ..)| *c)
        .into_iter()
        .sorted_by_key(|(log_size, _)| Reverse(*log_size))
        .map(|(log_size, tuples)| {
            let (_, openings, queried_valued_per_column): (Vec<_>, Vec<_>, Vec<_>) =
                multiunzip(tuples);
            fri_answers_for_log_size(
                log_size,
                &openings,
                random_coeff,
                &query_domain_per_log_size[&log_size],
                &queried_valued_per_column,
            )
        })
        .collect()
}

pub fn fri_answers_for_log_size(
    log_size: u32,
    openings: &[&Vec<PointOpening>],
    random_coeff: SecureField,
    query_domain: &SparseSubCircleDomain,
    queried_values_per_column: &[&Vec<BaseField>],
) -> Result<SparseCircleEvaluation<SecureField>, VerificationError> {
    let commitment_domain = CanonicCoset::new(log_size).circle_domain();
    let batched_openings = BatchedColumnOpenings::new(openings);
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
                    &batched_openings,
                )
                .to_cpu_circle_eval()
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
    use crate::core::commitment_scheme::quotients::{compute_fri_quotients, PointOpening};
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
            compute_fri_quotients(&[&eval], &[vec![PointOpening { point, value }]], coeff)
                .pop()
                .unwrap();
        let quot_poly_base_field =
            CPUCircleEvaluation::new(eval_domain, quot_eval.values.columns[0].clone())
                .interpolate();
        assert!(quot_poly_base_field.is_in_fft_space(LOG_SIZE));
    }
}
