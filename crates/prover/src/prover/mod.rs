use thiserror::Error;
use tracing::{info, instrument, span, Level};

use crate::core::air::{ComponentProver, ComponentProvers};
use crate::core::backend::BackendForChannel;
use crate::core::channel::{Channel, MerkleChannel};
use crate::core::circle::CirclePoint;
use crate::core::fields::qm31::{SecureField, SECURE_EXTENSION_DEGREE};
use crate::core::pcs::CommitmentSchemeProver;
use crate::core::proof::StarkProof;

pub const PREPROCESSED_TRACE_IDX: usize = 0;

#[instrument(skip_all)]
pub fn prove<B: BackendForChannel<MC>, MC: MerkleChannel>(
    components: &[&dyn ComponentProver<B>],
    channel: &mut MC::C,
    mut commitment_scheme: CommitmentSchemeProver<'_, B, MC>,
) -> Result<StarkProof<MC::H>, ProvingError> {
    let n_preprocessed_columns = commitment_scheme.trees[PREPROCESSED_TRACE_IDX]
        .polynomials
        .len();
    let component_provers = ComponentProvers {
        components: components.to_vec(),
        n_preprocessed_columns,
    };
    let trace = commitment_scheme.trace();

    // Evaluate and commit on composition polynomial.
    let random_coeff = channel.draw_felt();

    let span = span!(Level::INFO, "Composition", class = "Composition").entered();
    let span1 = span!(
        Level::INFO,
        "Generation",
        class = "CompositionPolynomialGeneration"
    )
    .entered();
    let composition_poly = component_provers.compute_composition_polynomial(random_coeff, &trace);
    span1.exit();

    let mut tree_builder = commitment_scheme.tree_builder();
    tree_builder.extend_polys(composition_poly.into_coordinate_polys());
    tree_builder.commit(channel);
    span.exit();

    // Draw OODS point.
    let oods_point = CirclePoint::<SecureField>::get_random_point(channel);

    // Get mask sample points relative to oods point.
    let mut sample_points = component_provers.components().mask_points(oods_point);

    // Add the composition polynomial mask points.
    sample_points.push(vec![vec![oods_point]; SECURE_EXTENSION_DEGREE]);

    // Prove the trace and composition OODS values, and retrieve them.
    let commitment_scheme_proof = commitment_scheme.prove_values(sample_points, channel);
    let proof = StarkProof(commitment_scheme_proof);
    info!(proof_size_estimate = proof.size_estimate());

    // Evaluate composition polynomial at OODS point and check that it matches the trace OODS
    // values. This is a sanity check.
    if proof.extract_composition_oods_eval().unwrap()
        != component_provers
            .components()
            .eval_composition_polynomial_at_point(oods_point, &proof.sampled_values, random_coeff)
    {
        return Err(ProvingError::ConstraintsNotSatisfied);
    }

    Ok(proof)
}

#[derive(Clone, Copy, Debug, Error)]
pub enum ProvingError {
    #[error("Constraints not satisfied.")]
    ConstraintsNotSatisfied,
}
