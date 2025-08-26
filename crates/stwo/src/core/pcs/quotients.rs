use core::cmp::Reverse;

use itertools::{izip, multiunzip, zip_eq, Itertools};
use num_traits::{One, Zero};
use serde::{Deserialize, Serialize};
use std_shims::{BTreeMap, Vec};

use super::TreeVec;
use crate::core::circle::CirclePoint;
use crate::core::constraints::complex_conjugate_line_coeffs;
use crate::core::fields::cm31::CM31;
use crate::core::fields::m31::{BaseField, M31};
use crate::core::fields::qm31::SecureField;
use crate::core::fields::FieldExpOps;
use crate::core::fri::FriProof;
use crate::core::pcs::PcsConfig;
use crate::core::poly::circle::CanonicCoset;
use crate::core::utils::bit_reverse_index;
use crate::core::vcs::verifier::MerkleDecommitment;
use crate::core::vcs::MerkleHasher;
use crate::core::verifier::VerificationError;
use crate::core::ColumnVec;

// Used for no_std support.
pub type IndexMap<K, V> = indexmap::IndexMap<K, V, core::hash::BuildHasherDefault<fnv::FnvHasher>>;

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct CommitmentSchemeProof<H: MerkleHasher> {
    pub config: PcsConfig,
    pub commitments: TreeVec<H::Hash>,
    pub sampled_values: TreeVec<ColumnVec<Vec<SecureField>>>,
    pub decommitments: TreeVec<MerkleDecommitment<H>>,
    pub queried_values: TreeVec<Vec<BaseField>>,
    pub proof_of_work: u64,
    pub fri_proof: FriProof<H>,
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
        let mut grouped_samples = IndexMap::default();
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

pub fn fri_answers(
    column_log_sizes: TreeVec<Vec<u32>>,
    samples: TreeVec<Vec<Vec<PointSample>>>,
    random_coeff: SecureField,
    query_positions_per_log_size: &BTreeMap<u32, Vec<usize>>,
    queried_values: TreeVec<Vec<BaseField>>,
    n_columns_per_log_size: TreeVec<&BTreeMap<u32, usize>>,
) -> Result<ColumnVec<Vec<SecureField>>, VerificationError> {
    let mut queried_values = queried_values.map(|values| values.into_iter());

    izip!(column_log_sizes.flatten(), samples.flatten().iter())
        .sorted_by_key(|(log_size, ..)| Reverse(*log_size))
        .group_by(|(log_size, ..)| *log_size)
        .into_iter()
        .map(|(log_size, tuples)| {
            let (_, samples): (Vec<_>, Vec<_>) = multiunzip(tuples);
            fri_answers_for_log_size(
                log_size,
                &samples,
                random_coeff,
                &query_positions_per_log_size[&log_size],
                &mut queried_values,
                n_columns_per_log_size
                    .as_ref()
                    .map(|columns_log_sizes| *columns_log_sizes.get(&log_size).unwrap_or(&0)),
            )
        })
        .collect()
}

pub fn fri_answers_for_log_size(
    log_size: u32,
    samples: &[&Vec<PointSample>],
    random_coeff: SecureField,
    query_positions: &[usize],
    queried_values: &mut TreeVec<impl Iterator<Item = BaseField>>,
    n_columns: TreeVec<usize>,
) -> Result<Vec<SecureField>, VerificationError> {
    let sample_batches = ColumnSampleBatch::new_vec(samples);
    // TODO(ilya): Is it ok to use the same `random_coeff` for all log sizes.
    let quotient_constants = quotient_constants(&sample_batches, random_coeff);
    let commitment_domain = CanonicCoset::new(log_size).circle_domain();

    let mut quotient_evals_at_queries = Vec::new();
    for &query_position in query_positions {
        let domain_point = commitment_domain.at(bit_reverse_index(query_position, log_size));

        let queried_values_at_row = queried_values
            .as_mut()
            .zip_eq(n_columns.as_ref())
            .map(|(queried_values, n_columns)| queried_values.take(*n_columns).collect())
            .flatten();

        quotient_evals_at_queries.push(accumulate_row_quotients(
            &sample_batches,
            &queried_values_at_row,
            &quotient_constants,
            domain_point,
        ));
    }

    Ok(quotient_evals_at_queries)
}

pub fn accumulate_row_quotients(
    sample_batches: &[ColumnSampleBatch],
    queried_values_at_row: &[BaseField],
    quotient_constants: &QuotientConstants,
    domain_point: CirclePoint<BaseField>,
) -> SecureField {
    let denominator_inverses = denominator_inverses(sample_batches, domain_point);
    let mut row_accumulator = SecureField::zero();
    for (sample_batch, line_coeffs, batch_coeff, denominator_inverse) in izip!(
        sample_batches,
        &quotient_constants.line_coeffs,
        &quotient_constants.batch_random_coeffs,
        denominator_inverses
    ) {
        let mut numerator = SecureField::zero();
        for ((column_index, _), (a, b, c)) in zip_eq(&sample_batch.columns_and_values, line_coeffs)
        {
            let value = queried_values_at_row[*column_index] * *c;
            // The numerator is a line equation passing through
            //   (sample_point.y, sample_value), (conj(sample_point), conj(sample_value))
            // evaluated at (domain_point.y, value).
            // When substituting a polynomial in this line equation, we get a polynomial with a root
            // at sample_point and conj(sample_point) if the original polynomial had the values
            // sample_value and conj(sample_value) at these points.
            let linear_term = *a * domain_point.y + *b;
            numerator += value - linear_term;
        }

        row_accumulator = row_accumulator * *batch_coeff + numerator.mul_cm31(denominator_inverse);
    }
    row_accumulator
}

/// Precomputes the complex conjugate line coefficients for each column in each sample batch.
///
/// For the `i`-th (in a sample batch) column's numerator term `alpha^i * (c * F(p) - (a * p.y +
/// b))`, we precompute and return the constants: (`alpha^i * a`, `alpha^i * b`, `alpha^i * c`).
pub fn column_line_coeffs(
    sample_batches: &[ColumnSampleBatch],
    random_coeff: SecureField,
) -> Vec<Vec<(SecureField, SecureField, SecureField)>> {
    sample_batches
        .iter()
        .map(|sample_batch| {
            let mut alpha = SecureField::one();
            sample_batch
                .columns_and_values
                .iter()
                .map(|(_, sampled_value)| {
                    alpha *= random_coeff;
                    let sample = PointSample {
                        point: sample_batch.point,
                        value: *sampled_value,
                    };
                    complex_conjugate_line_coeffs(&sample, alpha)
                })
                .collect()
        })
        .collect()
}

/// Precomputes the random coefficients used to linearly combine the batched quotients.
///
/// For each sample batch we compute random_coeff^(number of columns in the batch),
/// which is used to linearly combine the batch with the next one.
pub fn batch_random_coeffs(
    sample_batches: &[ColumnSampleBatch],
    random_coeff: SecureField,
) -> Vec<SecureField> {
    sample_batches
        .iter()
        .map(|sb| random_coeff.pow(sb.columns_and_values.len() as u128))
        .collect()
}

fn denominator_inverses(
    sample_batches: &[ColumnSampleBatch],
    domain_point: CirclePoint<M31>,
) -> Vec<CM31> {
    let mut denominators = Vec::new();

    // We want a P to be on a line that passes through a point Pr + uPi in QM31^2, and its conjugate
    // Pr - uPi. Thus, Pr - P is parallel to Pi. Or, (Pr - P).x * Pi.y - (Pr - P).y * Pi.x = 0.
    for sample_batch in sample_batches {
        // Extract Pr, Pi.
        let prx = sample_batch.point.x.0;
        let pry = sample_batch.point.y.0;
        let pix = sample_batch.point.x.1;
        let piy = sample_batch.point.y.1;
        denominators.push((prx - domain_point.x) * piy - (pry - domain_point.y) * pix);
    }

    CM31::batch_inverse(&denominators)
}

pub fn quotient_constants(
    sample_batches: &[ColumnSampleBatch],
    random_coeff: SecureField,
) -> QuotientConstants {
    QuotientConstants {
        line_coeffs: column_line_coeffs(sample_batches, random_coeff),
        batch_random_coeffs: batch_random_coeffs(sample_batches, random_coeff),
    }
}

/// Holds the precomputed constant values used in each quotient evaluation.
pub struct QuotientConstants {
    /// The line coefficients for each quotient numerator term. For more details see
    /// [self::column_line_coeffs].
    pub line_coeffs: Vec<Vec<(SecureField, SecureField, SecureField)>>,
    /// The random coefficients used to linearly combine the batched quotients For more details see
    /// [self::batch_random_coeffs].
    pub batch_random_coeffs: Vec<SecureField>,
}
