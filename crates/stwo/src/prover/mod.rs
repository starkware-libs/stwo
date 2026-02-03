use thiserror::Error;
use tracing::{info, instrument, span, Level};

use crate::core::channel::{Channel, MerkleChannel};
use crate::core::circle::CirclePoint;
use crate::core::fields::qm31::{SecureField, SECURE_EXTENSION_DEGREE};
use crate::core::proof::{ExtendedStarkProof, StarkProof};
use crate::core::verifier::{COMPOSITION_LOG_SPLIT, PREPROCESSED_TRACE_IDX};
use crate::prover::backend::BackendForChannel;

mod air;
pub use air::component_prover::{ComponentProver, ComponentProvers, Trace};
pub use air::{AccumulationOps, ColumnAccumulator, DomainEvaluationAccumulator};
mod pcs;
pub use pcs::quotient_ops::QuotientOps;
pub use pcs::{CommitmentSchemeProver, CommitmentTreeProver, TreeBuilder};
pub mod backend;
pub mod channel;
pub mod fri;
pub mod line;
pub mod lookups;
pub mod poly;
pub mod secure_column;
pub mod vcs;
pub mod vcs_lifted;

pub fn prove<B: BackendForChannel<MC>, MC: MerkleChannel>(
    components: &[&dyn ComponentProver<B>],
    channel: &mut MC::C,
    commitment_scheme: CommitmentSchemeProver<'_, B, MC>,
) -> Result<StarkProof<MC::H>, ProvingError> {
    Ok(prove_ex(components, channel, commitment_scheme, false)?.proof)
}

#[instrument(skip_all)]
pub fn prove_ex<B: BackendForChannel<MC>, MC: MerkleChannel>(
    components: &[&dyn ComponentProver<B>],
    channel: &mut MC::C,
    mut commitment_scheme: CommitmentSchemeProver<'_, B, MC>,
    include_all_preprocessed_columns: bool,
) -> Result<ExtendedStarkProof<MC::H>, ProvingError> {
    let n_preprocessed_columns = commitment_scheme.trees[PREPROCESSED_TRACE_IDX]
        .polynomials
        .len();
    let component_provers = ComponentProvers {
        components: components.to_vec(),
        n_preprocessed_columns,
    };
    let trace = commitment_scheme.trace();

    // Evaluate and commit on composition polynomial.
    let random_coeff = channel.draw_secure_felt();

    let span = span!(Level::INFO, "Composition", class = "Composition").entered();
    let span1 = span!(
        Level::INFO,
        "Generation",
        class = "CompositionPolynomialGeneration"
    )
    .entered();
    let composition_poly = component_provers.compute_composition_polynomial(random_coeff, &trace);
    let composition_log_size = composition_poly.log_size();
    span1.exit();

    // Commit on the Composition Polynomial by splitting its coeffs to two polynomialsof degree
    // half the size of the original polynomial, and commit on each half separately.
    let mut tree_builder = commitment_scheme.tree_builder();
    let (left_comp_poly_half, right_comp_poly_half) = composition_poly.split_at_mid();

    tree_builder.extend_polys(left_comp_poly_half.into_coordinate_polys());
    tree_builder.extend_polys(right_comp_poly_half.into_coordinate_polys());
    tree_builder.commit(channel);
    span.exit();

    // Draw OODS point.
    let oods_point = CirclePoint::<SecureField>::get_random_point(channel);
    // The max degree of a committed polynomial is equal to the degree of the composition poly /
    // 2^COMPOSITION_LOG_SPLIT.
    let max_log_degree_bound = composition_log_size - COMPOSITION_LOG_SPLIT;
    // Get mask sample points relative to oods point.
    let mut sample_points = component_provers.components().mask_points(
        oods_point,
        max_log_degree_bound,
        include_all_preprocessed_columns,
    );

    // Add the composition polynomial mask points.
    sample_points.push(vec![vec![oods_point]; 2 * SECURE_EXTENSION_DEGREE]);

    // Prove the trace and composition OODS values, and retrieve them.
    let commitment_scheme_proof = commitment_scheme.prove_values(sample_points, channel);
    let proof = StarkProof(commitment_scheme_proof.proof);
    info!(proof_size_estimate = proof.size_estimate());

    // Evaluate composition polynomial at OODS point and check that it matches the trace OODS
    // values. This is a sanity check.
    if proof
        .extract_composition_oods_eval(oods_point, composition_log_size)
        .unwrap()
        != component_provers
            .components()
            .eval_composition_polynomial_at_point(
                oods_point,
                &proof.sampled_values,
                random_coeff,
                max_log_degree_bound,
            )
    {
        return Err(ProvingError::ConstraintsNotSatisfied);
    }

    Ok(ExtendedStarkProof {
        proof,
        aux: commitment_scheme_proof.aux,
    })
}

#[derive(Clone, Copy, Debug, Error)]
pub enum ProvingError {
    #[error("Constraints not satisfied.")]
    ConstraintsNotSatisfied,
}
