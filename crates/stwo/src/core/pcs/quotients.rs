use itertools::{izip, zip_eq, Itertools};
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
use crate::core::fri::{FriProof, FriProofAux};
use crate::core::pcs::PcsConfig;
use crate::core::poly::circle::CanonicCoset;
use crate::core::utils::bit_reverse_index;
use crate::core::vcs_lifted::merkle_hasher::MerkleHasherLifted;
use crate::core::vcs_lifted::verifier::{MerkleDecommitmentLifted, MerkleDecommitmentLiftedAux};
use crate::core::verifier::VerificationError;
use crate::core::ColumnVec;

// Used for no_std support.
pub type IndexMap<K, V> = indexmap::IndexMap<K, V, core::hash::BuildHasherDefault<fnv::FnvHasher>>;

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct CommitmentSchemeProof<H: MerkleHasherLifted> {
    pub config: PcsConfig,
    pub commitments: TreeVec<H::Hash>,
    pub sampled_values: TreeVec<ColumnVec<Vec<SecureField>>>,
    pub decommitments: TreeVec<MerkleDecommitmentLifted<H>>,
    pub queried_values: TreeVec<Vec<BaseField>>,
    pub proof_of_work: u64,
    pub fri_proof: FriProof<H>,
}

/// Auxiliary data for a [CommitmentSchemeProof].
#[derive(Clone, Debug)]
pub struct CommitmentSchemeProofAux<H: MerkleHasherLifted> {
    /// The indices of the queries in the ordered they were sampled, before sorting and
    /// deduplication.
    pub unsorted_query_locations: Vec<usize>,
    /// For each trace, the Merkle decommitment auxiliary data.
    pub trace_decommitment: TreeVec<MerkleDecommitmentLiftedAux<H>>,
    /// The FRI auxiliary data.
    pub fri: FriProofAux<H>,
}

pub struct ExtendedCommitmentSchemeProof<H: MerkleHasherLifted> {
    pub proof: CommitmentSchemeProof<H>,
    pub aux: CommitmentSchemeProofAux<H>,
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

/// For each query position, corresponding to a domain point `p`, compute the FRI quotients
///
///     ∑ ∑ α^{k(i, z)} * (c(i, z) * f̃ᵢ(p) - b(i, z) - a(i, z)) / line(z,conj(z))(p)
///
/// where:
/// * the outer sum is over the set of sample points `z`,
/// * the inner sum is over the set of columns (corresponding to index `i`),
/// * line(z,conj(z))(p) is the equation of the line through (z, conj(z)) evaluated at `p`
/// * f̃ᵢ is the lift of the trace poly fᵢ to the domain of maximal log size.
/// * (a(i, z), b(i, z), c(i, z)) are the coefficients of the line equation `cY - aX - b` through
///   (z.y, f̃ᵢ(z)), (conj(z.y), conj(f̃ᵢ(z)).
pub fn fri_answers(
    column_log_sizes: TreeVec<Vec<u32>>,
    samples: TreeVec<Vec<Vec<PointSample>>>,
    random_coeff: SecureField,
    query_positions: &[usize],
    queried_values: TreeVec<Vec<BaseField>>,
    n_columns_per_log_size: TreeVec<&BTreeMap<u32, usize>>,
) -> Result<ColumnVec<Vec<SecureField>>, VerificationError> {
    let mut queried_values = queried_values.map(|values| values.into_iter());
    let lifting_log_size = *column_log_sizes.0.iter().flatten().max().unwrap();
    let flattened_samples = samples.flatten();
    let flattened_columns = column_log_sizes.flatten();
    let zipped_sorted: Vec<_> = izip!(flattened_columns.iter(), flattened_samples.iter())
        .sorted_by_key(|(log_size, _)| *log_size)
        .collect();

    let mut res = Vec::with_capacity(query_positions.len());
    for position in query_positions.iter() {
        let mut curr_coeff_power = SecureField::one();
        let mut sum = SecureField::zero();
        let grouped = zipped_sorted.iter().group_by(|(log_size, _)| *log_size);
        for (log_size, group) in grouped.into_iter() {
            let samples: Vec<_> = group.map(|(_, samples)| *samples).collect();
            let n_cols = n_columns_per_log_size
                .as_ref()
                .map(|map| *map.get(log_size).unwrap_or(&0));

            let value = fri_answers_for_unlifted_log_size(
                lifting_log_size,
                &samples,
                random_coeff,
                &mut curr_coeff_power,
                *position,
                &mut queried_values,
                n_cols,
            )?;

            sum += value;
        }
        res.push(sum);
    }
    // TODO(Leo): change the output type once we change fri's API.
    Ok(vec![res])
}

pub fn fri_answers_for_unlifted_log_size(
    log_size: u32,
    samples: &[&Vec<PointSample>],
    random_coeff: SecureField,
    curr_coeff_power: &mut SecureField,
    query_position: usize,
    queried_values: &mut TreeVec<impl Iterator<Item = BaseField>>,
    n_columns: TreeVec<usize>,
) -> Result<SecureField, VerificationError> {
    let sample_batches = ColumnSampleBatch::new_vec(samples);
    // TODO(ilya): Is it ok to use the same `random_coeff` for all log sizes.
    let quotient_constants = quotient_constants(&sample_batches, random_coeff, curr_coeff_power);
    let commitment_domain = CanonicCoset::new(log_size).circle_domain();
    let domain_point = commitment_domain.at(bit_reverse_index(query_position, log_size));
    let queried_values_at_row = queried_values
        .as_mut()
        .zip_eq(n_columns.as_ref())
        .map(|(queried_values, n_columns)| queried_values.take(*n_columns).collect())
        .flatten();

    Ok(accumulate_row_quotients(
        &sample_batches,
        &queried_values_at_row,
        &quotient_constants,
        domain_point,
    ))
}

pub fn accumulate_row_quotients(
    sample_batches: &[ColumnSampleBatch],
    queried_values_at_row: &[BaseField],
    quotient_constants: &QuotientConstants,
    domain_point: CirclePoint<BaseField>,
) -> SecureField {
    let sample_points = sample_batches.iter().map(|b| b.point).collect_vec();
    let denominator_inverses = denominator_inverses(&sample_points, domain_point);
    let mut row_accumulator = SecureField::zero();
    for (sample_batch, line_coeffs, denominator_inverse) in izip!(
        sample_batches,
        &quotient_constants.line_coeffs,
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

        row_accumulator += numerator.mul_cm31(denominator_inverse);
    }
    row_accumulator
}

/// Computes the sum
///     ∑ α^{k_i} * (cᵢ * f̃ᵢ(p) - bᵢ)
/// where:
/// * i is an index into `queried_values_at_row` that runs over the columns involved in the batch.
/// * f̃ᵢ(p) is `queried_values_at_row[i]`.
pub fn accumulate_row_partial_numerators(
    batch: &ColumnSampleBatch,
    queried_values_at_row: &[BaseField],
    coeffs: &Vec<(SecureField, SecureField, SecureField)>,
) -> SecureField {
    let mut numerator = SecureField::zero();
    for ((column_index, _), (_, b, c)) in zip_eq(&batch.columns_and_values, coeffs) {
        let value = queried_values_at_row[*column_index] * *c;
        numerator += value - *b;
    }
    numerator
}

/// Precomputes the complex conjugate line coefficients for each column in each sample batch.
///
/// For the `i`-th numerator term `curr_coeff_power * alpha^i * (c * F(p) - (a * p.y + b))`,
/// we precompute and return the constants: (`curr_coeff_power * alpha^i * a`, `curr_coeff_power *
/// alpha^i * b`, `curr_coeff_power * alpha^i * c`). The index `i` is zero-based and runs
/// monotonically across all sample batches (i.e. the index of the `m`-th column in the `n`-th batch
/// is `m + Σ len(batch_k)`, for `k < n`).
pub fn column_line_coeffs(
    sample_batches: &[ColumnSampleBatch],
    random_coeff: SecureField,
    curr_coeff_power: &mut SecureField,
) -> Vec<Vec<(SecureField, SecureField, SecureField)>> {
    sample_batches
        .iter()
        .map(|sample_batch| {
            sample_batch
                .columns_and_values
                .iter()
                .map(|(_, sampled_value)| {
                    let sample = PointSample {
                        point: sample_batch.point,
                        value: *sampled_value,
                    };
                    let line_coeffs = complex_conjugate_line_coeffs(&sample, *curr_coeff_power);
                    *curr_coeff_power *= random_coeff;
                    line_coeffs
                })
                .collect()
        })
        .collect()
}

pub fn denominator_inverses(
    sample_points: &[CirclePoint<SecureField>],
    domain_point: CirclePoint<M31>,
) -> Vec<CM31> {
    let mut denominators = Vec::new();

    // We want a P to be on a line that passes through a point Pr + uPi in QM31^2, and its conjugate
    // Pr - uPi. Thus, Pr - P is parallel to Pi. Or, (Pr - P).x * Pi.y - (Pr - P).y * Pi.x = 0.
    for sample_point in sample_points {
        // Extract Pr, Pi.
        let prx = sample_point.x.0;
        let pry = sample_point.y.0;
        let pix = sample_point.x.1;
        let piy = sample_point.y.1;
        denominators.push((prx - domain_point.x) * piy - (pry - domain_point.y) * pix);
    }

    CM31::batch_inverse(&denominators)
}

pub fn quotient_constants(
    sample_batches: &[ColumnSampleBatch],
    random_coeff: SecureField,
    curr_coeff_power: &mut SecureField,
) -> QuotientConstants {
    QuotientConstants {
        line_coeffs: column_line_coeffs(sample_batches, random_coeff, curr_coeff_power),
    }
}

/// Holds the precomputed constant values used in each quotient evaluation.
pub struct QuotientConstants {
    /// The line coefficients for each quotient numerator term. For more details see
    /// [self::column_line_coeffs].
    pub line_coeffs: Vec<Vec<(SecureField, SecureField, SecureField)>>,
}
