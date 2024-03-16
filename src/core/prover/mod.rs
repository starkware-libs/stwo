mod utils;

use itertools::Itertools;

use self::utils::component_wise_to_tree_wise;
use super::backend::Backend;
use super::commitment_scheme::{CommitmentSchemeProof, TreeVec};
use super::ColumnVec;
use crate::commitment_scheme::blake2_hash::Blake2sHasher;
use crate::commitment_scheme::hasher::Hasher;
use crate::core::air::{Air, AirExt};
use crate::core::backend::CPUBackend;
use crate::core::channel::{Blake2sChannel, Channel as ChannelTrait};
use crate::core::circle::CirclePoint;
use crate::core::commitment_scheme::{CommitmentSchemeProver, CommitmentSchemeVerifier};
use crate::core::fields::m31::BaseField;
use crate::core::fields::qm31::SecureField;
use crate::core::fields::secure::combine_secure_value;
use crate::core::poly::circle::CircleEvaluation;
use crate::core::poly::BitReversedOrder;
use crate::core::prover::utils::tree_wise_to_component_wise;

type Channel = Blake2sChannel;
type MerkleHasher = Blake2sHasher;

pub const LOG_BLOWUP_FACTOR: u32 = 1;
pub const LOG_LAST_LAYER_DEGREE_BOUND: u32 = 0;
pub const PROOF_OF_WORK_BITS: u32 = 12;
pub const N_QUERIES: usize = 3;

pub struct StarkProof {
    pub commitments: TreeVec<<MerkleHasher as Hasher>::Hash>,
    pub commitment_scheme_proof: CommitmentSchemeProof,
}

pub struct AdditionalProofData {
    pub composition_polynomial_oods_value: SecureField,
    pub composition_polynomial_random_coeff: SecureField,
    pub oods_point: CirclePoint<SecureField>,
    pub oods_quotients: Vec<CircleEvaluation<CPUBackend, SecureField, BitReversedOrder>>,
}

pub fn prove<B: Backend>(
    air: &impl Air<B>,
    channel: &mut Channel,
    trace: ColumnVec<CircleEvaluation<B, BaseField, BitReversedOrder>>,
) -> StarkProof {
    // Evaluate and commit on trace.
    // TODO(spapini): Commit on trace outside.
    let trace_polys = trace.into_iter().map(|poly| poly.interpolate()).collect();
    let mut commitment_scheme = CommitmentSchemeProver::new(LOG_BLOWUP_FACTOR);
    commitment_scheme.commit(trace_polys, channel);

    // Evaluate and commit on composition polynomial.
    let random_coeff = channel.draw_felt();
    let composition_polynomial_poly = air.compute_composition_polynomial(
        random_coeff,
        &air.component_traces(&commitment_scheme.trees[0].polynomials),
    );
    commitment_scheme.commit(composition_polynomial_poly.to_vec(), channel);

    // Evaluate the trace mask and the composition polynomial on the OODS point.
    let oods_point = CirclePoint::<SecureField>::get_random_point(channel);
    // TODO(spapini): This values are evaluted twice right now.
    let (open_points, _trace_oods_values) = air.mask_points_and_values(
        oods_point,
        &air.component_traces(&commitment_scheme.trees[0].polynomials),
    );

    // Add composiion poly points.
    let mut open_points = component_wise_to_tree_wise(air, open_points);
    open_points.0.push(vec![
        vec![oods_point],
        vec![oods_point],
        vec![oods_point],
        vec![oods_point],
    ]);

    let commitment_scheme_proof = commitment_scheme.open_values(open_points, channel);

    StarkProof {
        commitments: commitment_scheme.roots(),
        commitment_scheme_proof,
    }
}

pub fn verify(proof: StarkProof, air: &impl Air<CPUBackend>, channel: &mut Channel) -> bool {
    // Read trace commitment.
    let mut commitment_scheme = CommitmentSchemeVerifier::new();
    commitment_scheme.commit(proof.commitments[0], air.column_log_sizes(), channel);
    let random_coeff = channel.draw_felt();

    // Read composition polynomial commitment.
    commitment_scheme.commit(
        proof.commitments[1],
        vec![air.max_constraint_log_degree_bound(); 4],
        channel,
    );

    // Calculate the composition polynomial value on the OODS point using the trace values sent and
    // verify it is equal to the composition polynomial value sent.
    let oods_point = CirclePoint::<SecureField>::get_random_point(channel);
    let mut open_points = component_wise_to_tree_wise(air, air.mask_points(oods_point));
    open_points.0.push(vec![
        vec![oods_point],
        vec![oods_point],
        vec![oods_point],
        vec![oods_point],
    ]);

    // TODO(spapini): Save clone.
    let mut opened_values = proof.commitment_scheme_proof.opened_values.clone();
    let composition_oods_values = opened_values.0.pop().unwrap();
    let trace_oods_values = tree_wise_to_component_wise(air, opened_values);

    let composition_polynomial_oods_value =
        air.eval_composition_polynomial_at_point(oods_point, &trace_oods_values, random_coeff);
    assert_eq!(
        composition_polynomial_oods_value,
        combine_secure_value(
            composition_oods_values
                .iter()
                .flatten()
                .cloned()
                .collect_vec()
                .try_into()
                .unwrap()
        )
    );

    commitment_scheme.verify_opening(open_points, proof.commitment_scheme_proof, channel)
}
