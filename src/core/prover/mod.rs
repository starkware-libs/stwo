use std::iter::zip;

use itertools::{enumerate, Itertools};

use super::poly::circle::CanonicCoset;
use super::queries::SparseSubCircleDomain;
use super::ColumnVec;
use crate::commitment_scheme::blake2_hash::Blake2sHasher;
use crate::commitment_scheme::hasher::Hasher;
use crate::core::air::evaluation::SECURE_EXTENSION_DEGREE;
use crate::core::air::{Air, AirExt};
use crate::core::backend::cpu::CPUCircleEvaluation;
use crate::core::backend::CPUBackend;
use crate::core::channel::{Blake2sChannel, Channel as ChannelTrait};
use crate::core::circle::CirclePoint;
use crate::core::commitment_scheme::{
    CommitmentSchemeProver, CommitmentSchemeVerifier, Decommitments, OpenedValues,
};
use crate::core::fields::m31::BaseField;
use crate::core::fields::qm31::SecureField;
use crate::core::fri::{FriConfig, FriProof, FriProver, FriVerifier, SparseCircleEvaluation};
use crate::core::oods::get_pair_oods_quotient;
use crate::core::poly::circle::{combine_secure_value, CircleEvaluation};
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

pub fn verify(proof: StarkProof, air: &impl Air<CPUBackend>, channel: &mut Channel) -> bool {
    // Read trace commitment.
    let mut commitment_scheme = CommitmentSchemeVerifier::new();
    commitment_scheme.commit(proof.commitments[0], channel);
    let random_coeff = channel.draw_felt();

    // Read composition polynomial commitment.
    commitment_scheme.commit(proof.commitments[1], channel);

    // Calculate the composition polynomial value on the OODS point using the trace values sent and
    // verify it is equal to the composition polynomial value sent.
    let oods_point = CirclePoint::<SecureField>::get_random_point(channel);
    let trace_oods_points = air.mask_points(oods_point);
    let composition_polynomial_oods_value = air.eval_composition_polynomial_at_point(
        oods_point,
        &proof.trace_oods_values,
        random_coeff,
    );
    assert_eq!(
        composition_polynomial_oods_value,
        combine_secure_value(proof.composition_polynomial_column_oods_values)
    );

    let bounds = air.quotient_log_bounds();
    let fri_config = FriConfig::new(LOG_LAST_LAYER_DEGREE_BOUND, LOG_BLOWUP_FACTOR, N_QUERIES);
    let mut fri_verifier =
        FriVerifier::commit(channel, fri_config, proof.fri_proof, bounds).unwrap();

    ProofOfWork::new(PROOF_OF_WORK_BITS).verify(channel, &proof.proof_of_work);
    let opening_positions = fri_verifier
        .column_opening_positions(channel)
        .into_values()
        .collect_vec();
    commitment_scheme.verify(&proof.decommitments, &opening_positions);

    let commitment_domains = air.commitment_domains();
    // Prepare the quotient evaluations needed for the FRI verifier.
    let sparse_circle_evaluations = prepare_fri_evaluations(
        opening_positions,
        proof.opened_values,
        trace_oods_points,
        proof.trace_oods_values,
        proof.composition_polynomial_column_oods_values,
        commitment_domains,
        oods_point,
    );

    fri_verifier.decommit(sparse_circle_evaluations).unwrap();

    true
}

#[inline(never)]
fn prepare_fri_evaluations(
    opening_positions: Vec<SparseSubCircleDomain>,
    opened_values: OpenedValues,
    trace_oods_points: ComponentVec<Vec<CirclePoint<SecureField>>>,
    trace_oods_values: ComponentVec<Vec<SecureField>>,
    composition_polynomial_column_oods_values: [SecureField; SECURE_EXTENSION_DEGREE],
    commitment_domains: Vec<CanonicCoset>,
    oods_point: CirclePoint<SecureField>,
) -> Vec<SparseCircleEvaluation<SecureField>> {
    // TODO(AlonH): Generalize when introducing mixed degree.
    let trace_commitment_domain = commitment_domains[0];
    let composition_polynomial_commitment_domain = commitment_domains[1];
    let mut sparse_circle_evaluations = Vec::new();
    for (opened_values, oods_value) in
        zip(&opened_values[1], composition_polynomial_column_oods_values)
    {
        let mut evaluation = Vec::new();
        let mut opened_values_iter = opened_values.iter();
        for sub_circle_domain in opening_positions[1].iter() {
            let values = (&mut opened_values_iter)
                .take(1 << sub_circle_domain.log_size)
                .copied()
                .collect();
            let sub_circle_evaluation = CircleEvaluation::new(
                sub_circle_domain
                    .to_circle_domain(&composition_polynomial_commitment_domain.circle_domain()),
                values,
            );
            evaluation.push(
                get_pair_oods_quotient(oods_point, oods_value, &sub_circle_evaluation)
                    .bit_reverse(),
            );
        }
        assert!(
            opened_values_iter.next().is_none(),
            "Not all values were used."
        );
        sparse_circle_evaluations.push(SparseCircleEvaluation::new(evaluation));
    }
    for (component_points, component_values) in zip(&trace_oods_points, &trace_oods_values) {
        for (i, (column_points, column_values)) in
            enumerate(zip(component_points, component_values))
        {
            for (oods_point, oods_value) in zip(column_points, column_values) {
                let mut evaluation = Vec::new();
                let mut opened_values = opened_values[0][i].iter().copied();
                for sub_circle_domain in opening_positions[0].iter() {
                    let values = (&mut opened_values)
                        .take(1 << sub_circle_domain.log_size)
                        .collect();
                    let sub_circle_evaluation = CircleEvaluation::new(
                        sub_circle_domain
                            .to_circle_domain(&trace_commitment_domain.circle_domain()),
                        values,
                    );
                    evaluation.push(
                        get_pair_oods_quotient(*oods_point, *oods_value, &sub_circle_evaluation)
                            .bit_reverse(),
                    );
                }
                assert!(opened_values.next().is_none(), "Not all values were used.");
                sparse_circle_evaluations.push(SparseCircleEvaluation::new(evaluation));
            }
        }
    }
    sparse_circle_evaluations
}
