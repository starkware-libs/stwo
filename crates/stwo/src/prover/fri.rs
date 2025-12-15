use hashbrown::HashMap;
use itertools::Itertools;
use num_traits::Zero;
use tracing::instrument;

use crate::core::channel::{Channel, MerkleChannel};
use crate::core::circle::Coset;
use crate::core::fields::m31::BaseField;
use crate::core::fields::qm31::{SecureField, QM31};
use crate::core::fri::{
    ExtendedFriLayerProof, ExtendedFriProof, FriConfig, FriLayerProof, FriLayerProofAux, FriProof,
    FriProofAux, CIRCLE_TO_LINE_FOLD_STEP, FOLD_STEP,
};
use crate::core::poly::line::{LineDomain, LinePoly};
use crate::core::queries::{draw_queries, Queries};
use crate::core::vcs_lifted::merkle_hasher::MerkleHasherLifted;
use crate::prover::backend::ColumnOps;
use crate::prover::line::LineEvaluation;
use crate::prover::poly::circle::{PolyOps, SecureEvaluation};
use crate::prover::poly::twiddles::TwiddleTree;
use crate::prover::poly::BitReversedOrder;
use crate::prover::secure_column::SecureColumnByCoords;
use crate::prover::vcs_lifted::ops::MerkleOpsLifted;
use crate::prover::vcs_lifted::prover::MerkleProverLifted;

pub trait FriOps: ColumnOps<BaseField> + PolyOps + Sized + ColumnOps<SecureField> {
    /// Folds a degree `d` polynomial into a degree `d/2` polynomial.
    ///
    /// Let `eval` be a polynomial evaluated on a [LineDomain] `E`, `alpha` be a random field
    /// element and `pi(x) = 2x^2 - 1` be the circle's x-coordinate doubling map. This function
    /// returns `f' = f0 + alpha * f1` evaluated on `pi(E)` such that `2f(x) = f0(pi(x)) + x *
    /// f1(pi(x))`.
    ///
    /// # Panics
    ///
    /// Panics if there are less than two evaluations.
    fn fold_line(
        eval: &LineEvaluation<Self>,
        alpha: SecureField,
        twiddles: &TwiddleTree<Self>,
    ) -> LineEvaluation<Self>;

    /// Folds and accumulates a degree `d` circle polynomial into a degree `d/2` univariate
    /// polynomial.
    ///
    /// Let `src` be the evaluation of a circle polynomial `f` on a
    /// [`CircleDomain`] `E`. This function computes evaluations of `f' = f0
    /// + alpha * f1` on the x-coordinates of `E` such that `2f(p) = f0(px) + py * f1(px)`. The
    /// evaluations of `f'` are accumulated into `dst` by the formula `dst = dst * alpha^2 + f'`.
    ///
    /// # Panics
    ///
    /// Panics if `src` is not double the length of `dst`.
    ///
    /// [`CircleDomain`]: crate::core::poly::circle::CircleDomain
    // TODO(andrew): Make folding factor generic.
    // TODO(andrew): Fold directly into FRI layer to prevent allocation.
    fn fold_circle_into_line(
        dst: &mut LineEvaluation<Self>,
        src: &SecureEvaluation<Self, BitReversedOrder>,
        alpha: SecureField,
        twiddles: &TwiddleTree<Self>,
    );

    /// Decomposes a FRI-space polynomial into a polynomial inside the fft-space and the
    /// remainder term.
    /// FRI-space: polynomials of total degree n/2.
    /// Based on lemma #12 from the CircleStark paper: f(P) = g(P)+ lambda * alternating(P),
    /// where lambda is the cosset diff of eval, and g is a polynomial in the fft-space.
    fn decompose(
        eval: &SecureEvaluation<Self, BitReversedOrder>,
    ) -> (SecureEvaluation<Self, BitReversedOrder>, SecureField);
}

pub struct FriDecommitResult<H: MerkleHasherLifted> {
    pub fri_proof: ExtendedFriProof<H>,
    pub query_positions: Vec<usize>,
    pub unsorted_query_locations: Vec<usize>,
}

/// A FRI prover that applies the FRI protocol to prove a set of polynomials are of low degree.
pub struct FriProver<'a, B: FriOps + MerkleOpsLifted<MC::H>, MC: MerkleChannel> {
    config: FriConfig,
    first_layer: FriFirstLayerProver<'a, B, MC::H>,
    inner_layers: Vec<FriInnerLayerProver<B, MC::H>>,
    last_layer_poly: LinePoly,
}
impl<'a, B: FriOps + MerkleOpsLifted<MC::H>, MC: MerkleChannel> FriProver<'a, B, MC> {
    /// Runs the commitment phase of FRI on one circle evaluation over a canonic circle domain.
    ///
    /// # Panics
    ///
    /// Panics if:
    /// * The evaluation is not from a sufficiently low degree circle polynomial.
    /// * The evaluation domain is not a canonic circle domain.
    #[instrument(skip_all)]
    pub fn commit(
        channel: &mut MC::C,
        config: FriConfig,
        column: &'a SecureEvaluation<B, BitReversedOrder>,
        twiddles: &TwiddleTree<B>,
    ) -> Self {
        assert!(column.domain.is_canonic(), "not canonic");

        let first_layer = Self::commit_first_layer(channel, column);
        let (inner_layers, last_layer_evaluation) =
            Self::commit_inner_layers(channel, config, column, twiddles);
        let last_layer_poly = Self::commit_last_layer(channel, config, last_layer_evaluation);

        Self {
            config,
            first_layer,
            inner_layers,
            last_layer_poly,
        }
    }

    /// Commits to the first FRI layer.
    fn commit_first_layer(
        channel: &mut MC::C,
        column: &'a SecureEvaluation<B, BitReversedOrder>,
    ) -> FriFirstLayerProver<'a, B, MC::H> {
        let layer = FriFirstLayerProver::new(column);
        MC::mix_root(channel, layer.merkle_tree.root());
        layer
    }

    /// Builds and commits to the inner FRI layers (all layers except the first and last).
    ///
    /// Returns all inner layers and the evaluation of the last layer.
    fn commit_inner_layers(
        channel: &mut MC::C,
        config: FriConfig,
        column: &SecureEvaluation<B, BitReversedOrder>,
        twiddles: &TwiddleTree<B>,
    ) -> (Vec<FriInnerLayerProver<B, MC::H>>, LineEvaluation<B>) {
        let first_inner_layer_log_size = column.domain.log_size() - CIRCLE_TO_LINE_FOLD_STEP;
        let first_inner_layer_domain =
            LineDomain::new(Coset::half_odds(first_inner_layer_log_size));

        let mut layer_evaluation = LineEvaluation::new_zero(first_inner_layer_domain);
        let mut layers = Vec::new();
        let folding_alpha = channel.draw_secure_felt();

        // Folding the max size column.
        B::fold_circle_into_line(&mut layer_evaluation, column, folding_alpha, twiddles);

        while layer_evaluation.len() > config.last_layer_domain_size() {
            let layer = FriInnerLayerProver::new(layer_evaluation);
            MC::mix_root(channel, layer.merkle_tree.root());
            let folding_alpha = channel.draw_secure_felt();
            layer_evaluation = B::fold_line(&layer.evaluation, folding_alpha, twiddles);
            layers.push(layer);
        }

        (layers, layer_evaluation)
    }

    /// Builds and commits to the last layer.
    ///
    /// The layer is committed to by sending the verifier all the coefficients of the remaining
    /// polynomial.
    ///
    /// # Panics
    ///
    /// Panics if:
    /// * The evaluation domain size exceeds the maximum last layer domain size.
    /// * The evaluation is not of sufficiently low degree.
    fn commit_last_layer(
        channel: &mut MC::C,
        config: FriConfig,
        evaluation: LineEvaluation<B>,
    ) -> LinePoly {
        assert_eq!(evaluation.len(), config.last_layer_domain_size());

        let evaluation = evaluation.to_cpu();
        let mut coeffs = evaluation.interpolate().into_ordered_coefficients();

        let last_layer_degree_bound = 1 << config.log_last_layer_degree_bound;
        let zeros = coeffs.split_off(last_layer_degree_bound);
        assert!(zeros.iter().all(SecureField::is_zero), "invalid degree");

        let last_layer_poly = LinePoly::from_ordered_coefficients(coeffs);
        channel.mix_felts(&last_layer_poly);

        last_layer_poly
    }

    /// Returns a FRI proof and the query positions.
    pub fn decommit(self, channel: &mut MC::C) -> FriDecommitResult<MC::H> {
        let first_layer_log_size = self.first_layer.column.domain.log_size();
        let unsorted_query_locations =
            draw_queries(channel, first_layer_log_size, self.config.n_queries);
        let queries = Queries::new(&unsorted_query_locations, first_layer_log_size);

        let fri_proof = self.decommit_on_queries(&queries);
        FriDecommitResult {
            fri_proof,
            query_positions: queries.positions,
            unsorted_query_locations,
        }
    }

    /// # Panics
    ///
    /// Panics if the queries were sampled on the wrong domain size.
    pub fn decommit_on_queries(self, queries: &Queries) -> ExtendedFriProof<MC::H> {
        let Self {
            config: _,
            first_layer,
            inner_layers,
            last_layer_poly,
        } = self;

        let first_layer_proof = first_layer.decommit(queries);

        let inner_layer_proofs = inner_layers
            .into_iter()
            .scan(
                queries.fold(CIRCLE_TO_LINE_FOLD_STEP),
                |layer_queries, layer| {
                    let layer_proof = layer.decommit(layer_queries);
                    *layer_queries = layer_queries.fold(FOLD_STEP);
                    Some(layer_proof)
                },
            )
            .collect_vec();

        let (inner_proofs, inner_layers_aux): (Vec<_>, Vec<_>) = inner_layer_proofs
            .into_iter()
            .map(|p| (p.proof, p.aux))
            .unzip();

        ExtendedFriProof {
            proof: FriProof {
                first_layer: first_layer_proof.proof,
                inner_layers: inner_proofs,
                last_layer_poly,
            },
            aux: FriProofAux {
                first_layer: first_layer_proof.aux,
                inner_layers: inner_layers_aux,
            },
        }
    }
}

/// Commitment to the first FRI layer.
struct FriFirstLayerProver<'a, B: FriOps + MerkleOpsLifted<H>, H: MerkleHasherLifted> {
    column: &'a SecureEvaluation<B, BitReversedOrder>,
    merkle_tree: MerkleProverLifted<B, H>,
}

impl<'a, B: FriOps + MerkleOpsLifted<H>, H: MerkleHasherLifted> FriFirstLayerProver<'a, B, H> {
    fn new(first_layer_column: &'a SecureEvaluation<B, BitReversedOrder>) -> Self {
        let coordinate_columns = first_layer_column.columns.iter().collect();
        let merkle_tree = MerkleProverLifted::commit(coordinate_columns);

        FriFirstLayerProver {
            column: first_layer_column,
            merkle_tree,
        }
    }

    fn decommit(self, queries: &Queries) -> ExtendedFriLayerProof<H> {
        assert_eq!(queries.log_domain_size, self.column.domain.log_size());

        let (column_decommitment_positions, column_witness, value_map) =
            compute_decommitment_positions_and_witness_evals(
                self.column,
                &queries.positions,
                CIRCLE_TO_LINE_FOLD_STEP,
            );

        let (_, decommitment) = self.merkle_tree.decommit(
            &column_decommitment_positions,
            self.column.columns.iter().collect(),
        );

        ExtendedFriLayerProof {
            proof: FriLayerProof {
                fri_witness: column_witness,
                decommitment: decommitment.decommitment,
                commitment: self.merkle_tree.root(),
            },
            aux: FriLayerProofAux {
                all_values: vec![value_map],
                decommitment: decommitment.aux,
            },
        }
    }
}

/// A FRI layer comprises of a merkle tree that commits to evaluations of a polynomial.
///
/// The polynomial evaluations are viewed as evaluation of a polynomial on multiple distinct cosets
/// of size two. Each leaf of the merkle tree commits to a single coset evaluation.
// TODO(andrew): Support different step sizes and update docs.
// TODO(andrew): The docs are wrong. Each leaf of the merkle tree commits to a single
// QM31 value. This is inefficient and should be changed.
struct FriInnerLayerProver<B: FriOps + MerkleOpsLifted<H>, H: MerkleHasherLifted> {
    evaluation: LineEvaluation<B>,
    merkle_tree: MerkleProverLifted<B, H>,
}

impl<B: FriOps + MerkleOpsLifted<H>, H: MerkleHasherLifted> FriInnerLayerProver<B, H> {
    fn new(evaluation: LineEvaluation<B>) -> Self {
        let merkle_tree =
            MerkleProverLifted::commit(evaluation.values.columns.iter().collect_vec());
        FriInnerLayerProver {
            evaluation,
            merkle_tree,
        }
    }

    fn decommit(self, queries: &Queries) -> ExtendedFriLayerProof<H> {
        let (decommitment_positions, fri_witness, value_map) =
            compute_decommitment_positions_and_witness_evals(
                &self.evaluation.values,
                queries,
                FOLD_STEP,
            );

        let (_evals, decommitment) = self.merkle_tree.decommit(
            &decommitment_positions,
            self.evaluation.values.columns.iter().collect_vec(),
        );

        let commitment = self.merkle_tree.root();

        ExtendedFriLayerProof {
            proof: FriLayerProof {
                fri_witness,
                decommitment: decommitment.decommitment,
                commitment,
            },
            aux: FriLayerProofAux {
                all_values: vec![value_map],
                decommitment: decommitment.aux,
            },
        }
    }
}

/// Returns a column's merkle tree decommitment positions and the evals the verifier can't
/// deduce from previous computations but requires for decommitment and folding.
///
/// Returns a map from leaf index to its value.
fn compute_decommitment_positions_and_witness_evals(
    column: &SecureColumnByCoords<impl PolyOps>,
    query_positions: &[usize],
    fold_step: u32,
) -> (Vec<usize>, Vec<QM31>, HashMap<usize, QM31>) {
    let mut decommitment_positions = Vec::new();
    let mut witness_evals = Vec::new();
    let mut value_map = HashMap::new();

    // Group queries by the folding coset they reside in.
    for subset_queries in query_positions.chunk_by(|a, b| a >> fold_step == b >> fold_step) {
        let subset_start = (subset_queries[0] >> fold_step) << fold_step;
        let subset_decommitment_positions = subset_start..subset_start + (1 << fold_step);
        let mut subset_queries_iter = subset_queries.iter().peekable();

        for position in subset_decommitment_positions {
            // Add decommitment position.
            decommitment_positions.push(position);

            let eval = column.at(position);
            value_map.insert(position, eval);

            // Only add evals the verifier can't calculate.
            if subset_queries_iter.next_if_eq(&&position).is_none() {
                witness_evals.push(eval);
            }
        }
    }

    (decommitment_positions, witness_evals, value_map)
}

#[cfg(test)]
mod tests {

    use num_traits::One;

    use crate::core::circle::{CirclePointIndex, Coset};
    use crate::core::fields::m31::BaseField;
    use crate::core::fields::qm31::SecureField;
    use crate::core::fri::FriConfig;
    use crate::core::poly::circle::CircleDomain;
    use crate::core::test_utils::test_channel;
    use crate::core::vcs_lifted::blake2_merkle::Blake2sMerkleChannel;
    use crate::prover::backend::cpu::CpuCirclePoly;
    use crate::prover::backend::CpuBackend;
    use crate::prover::poly::circle::{PolyOps, SecureEvaluation};
    use crate::prover::poly::BitReversedOrder;

    /// Default blowup factor used for tests.
    const LOG_BLOWUP_FACTOR: u32 = 2;

    type FriProver<'a> = super::FriProver<'a, CpuBackend, Blake2sMerkleChannel>;

    #[test]
    #[should_panic = "invalid degree"]
    fn committing_high_degree_polynomial_fails() {
        const LOG_EXPECTED_BLOWUP_FACTOR: u32 = LOG_BLOWUP_FACTOR;
        const LOG_INVALID_BLOWUP_FACTOR: u32 = LOG_BLOWUP_FACTOR - 1;
        let config = FriConfig::new(2, LOG_EXPECTED_BLOWUP_FACTOR, 3);
        let column = polynomial_evaluation(6, LOG_INVALID_BLOWUP_FACTOR);
        let twiddles = CpuBackend::precompute_twiddles(column.domain.half_coset);

        FriProver::commit(&mut test_channel(), config, &column, &twiddles);
    }

    #[test]
    #[should_panic = "not canonic"]
    fn committing_column_from_invalid_domain_fails() {
        let invalid_domain = CircleDomain::new(Coset::new(CirclePointIndex::generator(), 3));
        assert!(!invalid_domain.is_canonic(), "must be an invalid domain");
        let config = FriConfig::new(2, 2, 3);
        let column = SecureEvaluation::new(
            invalid_domain,
            [SecureField::one(); 1 << 4].into_iter().collect(),
        );
        let twiddles = CpuBackend::precompute_twiddles(column.domain.half_coset);

        FriProver::commit(&mut test_channel(), config, &column, &twiddles);
    }

    /// Returns an evaluation of a random polynomial with degree `2^log_degree`.
    ///
    /// The evaluation domain size is `2^(log_degree + log_blowup_factor)`.
    fn polynomial_evaluation(
        log_degree: u32,
        log_blowup_factor: u32,
    ) -> SecureEvaluation<CpuBackend, BitReversedOrder> {
        let poly = CpuCirclePoly::new(vec![BaseField::one(); 1 << log_degree]);
        let coset = Coset::half_odds(log_degree + log_blowup_factor - 1);
        let domain = CircleDomain::new(coset);
        let values = poly.evaluate(domain);
        SecureEvaluation::new(domain, values.into_iter().map(SecureField::from).collect())
    }
}
