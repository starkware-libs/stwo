use std::iter::zip;

use itertools::enumerate;

use super::ColumnVec;
use crate::commitment_scheme::blake2_hash::Blake2sHasher;
use crate::commitment_scheme::hasher::Hasher;
use crate::core::air::evaluation::SECURE_EXTENSION_DEGREE;
use crate::core::air::{Air, AirExt};
use crate::core::backend::cpu::CPUCircleEvaluation;
use crate::core::backend::CPUBackend;
use crate::core::channel::{Blake2sChannel, Channel as ChannelTrait};
use crate::core::circle::CirclePoint;
use crate::core::commitment_scheme::{CommitmentSchemeProver, Decommitments, OpenedValues};
use crate::core::fields::m31::BaseField;
use crate::core::fields::qm31::SecureField;
use crate::core::fri::{FriConfig, FriProof, FriProver};
use crate::core::oods::get_pair_oods_quotient;
use crate::core::poly::circle::CircleEvaluation;
use crate::core::poly::BitReversedOrder;
use crate::core::proof_of_work::{ProofOfWork, ProofOfWorkProof};
use crate::core::ComponentVec;

type Channel = Blake2sChannel;
type MerkleHasher = Blake2sHasher;

pub const LOG_BLOWUP_FACTOR: u32 = 1;
// TODO(Andrew): Change to 0 once related bug is fixed.
pub const LOG_LAST_LAYER_DEGREE_BOUND: u32 = 1;
pub const PROOF_OF_WORK_BITS: u32 = 12;
pub const N_QUERIES: usize = 3;

pub struct StarkProof {
    pub commitments: Vec<<MerkleHasher as Hasher>::Hash>,
    pub decommitments: Decommitments,
    pub trace_oods_values: ComponentVec<Vec<SecureField>>,
    pub composition_polynomial_column_oods_values: [SecureField; SECURE_EXTENSION_DEGREE],
    pub opened_values: OpenedValues,
    pub proof_of_work: ProofOfWorkProof,
    pub fri_proof: FriProof<MerkleHasher>,
    pub additional_proof_data: AdditionalProofData,
}

pub struct AdditionalProofData {
    pub composition_polynomial_oods_value: SecureField,
    pub composition_polynomial_random_coeff: SecureField,
    pub oods_point: CirclePoint<SecureField>,
    pub oods_quotients: Vec<CircleEvaluation<CPUBackend, SecureField, BitReversedOrder>>,
}

pub fn prove(
    air: &impl Air<CPUBackend>,
    channel: &mut Channel,
    trace: ColumnVec<CPUCircleEvaluation<BaseField, BitReversedOrder>>,
) -> StarkProof {
    // Evaluate and commit on trace.
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
    let (trace_oods_points, trace_oods_values) = air.mask_points_and_values(
        oods_point,
        &air.component_traces(&commitment_scheme.trees[0].polynomials),
    );
    let composition_polynomial_oods_value = composition_polynomial_poly.eval_at_point(oods_point);

    // Calculate a quotient polynomial for each trace mask item and one for the composition
    // polynomial.
    let mut oods_quotients = Vec::with_capacity(trace_oods_points.len() + SECURE_EXTENSION_DEGREE);
    let composition_polynomial_column_oods_values =
        composition_polynomial_poly.eval_columns_at_point(oods_point);
    for (evaluation, value) in zip(
        &commitment_scheme.trees[1].evaluations,
        composition_polynomial_column_oods_values,
    ) {
        oods_quotients.push(get_pair_oods_quotient(oods_point, value, evaluation).bit_reverse());
    }
    for (component_points, component_values) in zip(&trace_oods_points, &trace_oods_values) {
        for (i, (column_points, column_values)) in
            enumerate(zip(component_points, component_values))
        {
            for (point, value) in zip(column_points, column_values) {
                oods_quotients.push(
                    get_pair_oods_quotient(
                        *point,
                        *value,
                        &commitment_scheme.trees[0].evaluations[i],
                    )
                    .bit_reverse(),
                );
            }
        }
    }

    let fri_config = FriConfig::new(LOG_LAST_LAYER_DEGREE_BOUND, LOG_BLOWUP_FACTOR, N_QUERIES);
    let fri_prover = FriProver::commit(channel, fri_config, &oods_quotients);

    let proof_of_work = ProofOfWork::new(PROOF_OF_WORK_BITS).prove(channel);
    let (fri_proof, fri_opening_positions) = fri_prover.decommit(channel);

    let (opened_values, decommitments) = commitment_scheme.decommit(fri_opening_positions);

    StarkProof {
        commitments: commitment_scheme.roots(),
        decommitments,
        trace_oods_values,
        composition_polynomial_column_oods_values,
        opened_values,
        proof_of_work,
        fri_proof,
        additional_proof_data: AdditionalProofData {
            composition_polynomial_oods_value,
            composition_polynomial_random_coeff: random_coeff,
            oods_point,
            oods_quotients,
        },
    }
}
