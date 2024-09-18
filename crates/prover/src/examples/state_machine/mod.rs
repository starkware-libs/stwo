pub mod components;
pub mod gen;

use components::{StateMachineElements, StateTransitionEval};
use gen::{gen_interaction_trace, gen_trace, State};
use itertools::Itertools;

use crate::constraint_framework::constant_columns::gen_is_first;
use crate::constraint_framework::{FrameworkComponent, TraceLocationAllocator};
use crate::core::air::Component;
use crate::core::backend::simd::m31::LOG_N_LANES;
use crate::core::backend::simd::SimdBackend;
use crate::core::channel::Blake2sChannel;
use crate::core::pcs::{CommitmentSchemeProver, CommitmentSchemeVerifier, PcsConfig, TreeVec};
use crate::core::poly::circle::{CanonicCoset, CirclePoly, PolyOps};
use crate::core::prover::{prove, verify, StarkProof, VerificationError};
use crate::core::vcs::blake2_merkle::{Blake2sMerkleChannel, Blake2sMerkleHasher};

pub type StateMachineOp0Component = FrameworkComponent<StateTransitionEval<0>>;

#[allow(unused)]
pub fn prove_state_machine(
    log_n_rows: u32,
    initial_state: State,
    config: PcsConfig,
    channel: &mut Blake2sChannel,
) -> (
    StateMachineOp0Component,
    StarkProof<Blake2sMerkleHasher>,
    TreeVec<Vec<CirclePoly<SimdBackend>>>,
) {
    assert!(log_n_rows >= LOG_N_LANES);

    // Precompute twiddles.
    let twiddles = SimdBackend::precompute_twiddles(
        CanonicCoset::new(log_n_rows + config.fri_config.log_blowup_factor + 1)
            .circle_domain()
            .half_coset,
    );

    // Setup protocol.
    let commitment_scheme =
        &mut CommitmentSchemeProver::<_, Blake2sMerkleChannel>::new(config, &twiddles);

    // Trace.
    let trace_op0 = gen_trace(log_n_rows, initial_state, 0);
    let mut tree_builder = commitment_scheme.tree_builder();
    tree_builder.extend_evals(trace_op0.clone());
    tree_builder.commit(channel);

    // Draw lookup element.
    let lookup_elements = StateMachineElements::draw(channel);

    // Interaction trace.
    let (interaction_trace_op0, total_sum_op0) =
        gen_interaction_trace(log_n_rows, &trace_op0, 0, &lookup_elements);
    let mut tree_builder = commitment_scheme.tree_builder();
    tree_builder.extend_evals(interaction_trace_op0);
    tree_builder.commit(channel);

    // Constant trace.
    let mut tree_builder = commitment_scheme.tree_builder();
    tree_builder.extend_evals(vec![gen_is_first(log_n_rows)]);
    tree_builder.commit(channel);

    let trace_polys = commitment_scheme
        .trees
        .as_ref()
        .map(|t| t.polynomials.iter().cloned().collect_vec());

    // Prove constraints.
    let component_op0 = StateMachineOp0Component::new(
        &mut TraceLocationAllocator::default(),
        StateTransitionEval {
            log_n_rows,
            lookup_elements,
            total_sum: total_sum_op0,
        },
    );

    let proof = prove(&[&component_op0], channel, commitment_scheme).unwrap();

    (component_op0, proof, trace_polys)
}

pub fn verify_state_machine(
    config: PcsConfig,
    channel: &mut Blake2sChannel,
    component: StateMachineOp0Component,
    proof: StarkProof<Blake2sMerkleHasher>,
) -> Result<(), VerificationError> {
    let commitment_scheme = &mut CommitmentSchemeVerifier::<Blake2sMerkleChannel>::new(config);

    // Decommit.
    // Retrieve the expected column sizes in each commitment interaction, from the AIR.
    let sizes = component.trace_log_degree_bounds();
    // Trace columns.
    commitment_scheme.commit(proof.commitments[0], &sizes[0], channel);
    // Interaction columns.
    commitment_scheme.commit(proof.commitments[1], &sizes[1], channel);
    // Constant columns.
    commitment_scheme.commit(proof.commitments[2], &sizes[2], channel);

    verify(&[&component], channel, commitment_scheme, proof)
}

#[cfg(test)]
mod tests {
    use num_traits::Zero;

    use super::components::STATE_SIZE;
    use super::{prove_state_machine, verify_state_machine};
    use crate::constraint_framework::{assert_constraints, FrameworkEval};
    use crate::core::channel::Blake2sChannel;
    use crate::core::fields::m31::M31;
    use crate::core::fields::qm31::QM31;
    use crate::core::pcs::PcsConfig;
    use crate::core::poly::circle::CanonicCoset;

    #[test]
    fn test_state_machine_constraints() {
        let log_n_rows = 8;
        let config = PcsConfig::default();

        // Initial and last state.
        let initial_state = [M31::zero(); STATE_SIZE];
        let last_state = [M31::from_u32_unchecked(1 << log_n_rows), M31::zero()];

        // Setup protocol.
        let channel = &mut Blake2sChannel::default();
        let (component, _, trace_polys) =
            prove_state_machine(log_n_rows, initial_state, config, channel);

        let interaction_elements = component.lookup_elements.clone();
        let initial_state_comb: QM31 = interaction_elements.combine(&initial_state);
        let last_state_comb: QM31 = interaction_elements.combine(&last_state);

        // Assert total sum is `(1 / initial_state_comb) - (1 / last_state_comb)`.
        assert_eq!(
            component.total_sum * initial_state_comb * last_state_comb,
            last_state_comb - initial_state_comb
        );

        // Assert constraints.
        assert_constraints(&trace_polys, CanonicCoset::new(log_n_rows), |eval| {
            component.evaluate(eval);
        });
    }

    #[test]
    fn test_state_machine_prove() {
        let log_n_rows = 8;
        let config = PcsConfig::default();
        let initial_state = [M31::zero(); STATE_SIZE];
        let prover_channel = &mut Blake2sChannel::default();
        let (component_op0, proof, _) =
            prove_state_machine(log_n_rows, initial_state, config, prover_channel);

        let verifier_channel = &mut Blake2sChannel::default();
        verify_state_machine(config, verifier_channel, component_op0, proof).unwrap();
    }
}
