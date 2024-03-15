use std::cmp::Reverse;
use std::collections::BTreeMap;

use itertools::{izip, multiunzip, Itertools};

use crate::core::backend::{Backend, CPUBackend};
use crate::core::circle::CirclePoint;
use crate::core::fields::m31::BaseField;
use crate::core::fields::qm31::SecureField;
use crate::core::fields::secure::SecureEvaluation;
use crate::core::fri::SparseCircleEvaluation;
use crate::core::poly::circle::{CanonicCoset, CircleDomain, CircleEvaluation};
use crate::core::poly::BitReversedOrder;
use crate::core::queries::SparseSubCircleDomain;

pub trait QuotientOps: Backend {
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
) -> Vec<SparseCircleEvaluation<SecureField>> {
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
) -> SparseCircleEvaluation<SecureField> {
    let commitment_domain = CanonicCoset::new(log_size).circle_domain();
    let batched_openings = BatchedColumnOpenings::new(openings);
    for x in queried_values_per_column {
        assert_eq!(x.len(), query_domain.flatten().len());
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
                .to_cpu()
            })
            .collect(),
    );
    assert!(queried_values_per_column.iter().all(|x| x.is_empty()));
    res
}
