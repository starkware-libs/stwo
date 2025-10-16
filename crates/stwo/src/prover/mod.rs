use thiserror::Error;
use tracing::{info, instrument, span, Level};

use crate::core::channel::{Channel, MerkleChannel};
use crate::core::circle::CirclePoint;
use crate::core::fields::qm31::{SecureField, SECURE_EXTENSION_DEGREE};
use crate::core::proof::StarkProof;
use crate::core::verifier::PREPROCESSED_TRACE_IDX;
use crate::prover::backend::BackendForChannel;
use crate::prover::poly::circle::SecureCirclePoly;

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
    let random_coeff = channel.draw_secure_felt();

    let span = span!(Level::INFO, "Composition", class = "Composition").entered();
    let span1 = span!(
        Level::INFO,
        "Generation",
        class = "CompositionPolynomialGeneration"
    )
    .entered();
    let composition_poly = component_provers.compute_composition_polynomial(random_coeff, &trace);
    println!("composition_poly: {:?}", composition_poly);
    span1.exit();
    let (left_composition_poly, right_composition_poly) = composition_poly.split_at_mid();
    println!("left_composition_poly: {:?}", left_composition_poly);
    println!("right_composition_poly: {:?}", right_composition_poly);
    let mut tree_builder = commitment_scheme.tree_builder();
    
    let composition_poly_clone = composition_poly.clone();
    tree_builder.extend_polys(composition_poly.into_coordinate_polys());
    tree_builder.commit(channel);
    span.exit();

    // Draw OODS point.
    // let oods_point = CirclePoint::<SecureField>::get_random_point(channel);
    let oods_point = CirclePoint {
        x: SecureField::from_u32_unchecked(221714253, 601556545, 2021102783, 1712754591),
        y: SecureField::from_u32_unchecked(1736151795, 1429543180, 862074930, 782307515),
    };

    // evaluate left composition poly at oods_point
    let left_eval = left_composition_poly.eval_at_point(oods_point);
    let right_eval = right_composition_poly.eval_at_point(oods_point);
    let log_size = composition_poly_clone[0].log_size();
    let comp_poly_clone = SecureCirclePoly(composition_poly_clone);
    let comp_eval = comp_poly_clone.eval_at_point(oods_point);
    // use repeated double to get pi^{log_size-2}(oods_point.x)
    println!("left_eval: {:?}", left_eval);
    println!("right_eval: {:?}", right_eval);
    println!("comp_eval: {:?}", comp_eval);
    let pi_eval = oods_point.repeated_double(log_size - 2).x;
    assert_eq!(left_eval + pi_eval * right_eval, comp_eval);
    // println!("left + pi*right: {:?}", left_eval + pi_eval * right_eval);

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
