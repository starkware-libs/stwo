pub mod components;
pub mod gen;

use components::{
    State, StateMachineComponents, StateMachineElements, StateMachineOp0Component,
    StateMachineOp1Component, StateMachineProof, StateMachineStatement0, StateMachineStatement1,
    StateTransitionEval,
};
use gen::{gen_interaction_trace, gen_trace};
use itertools::{chain, Itertools};

use crate::constraint_framework::preprocessed_columns::gen_is_first;
use crate::constraint_framework::TraceLocationAllocator;
use crate::core::backend::simd::m31::LOG_N_LANES;
use crate::core::backend::simd::SimdBackend;
use crate::core::channel::Blake2sChannel;
use crate::core::fields::m31::M31;
use crate::core::fields::qm31::QM31;
use crate::core::pcs::{CommitmentSchemeProver, CommitmentSchemeVerifier, PcsConfig};
use crate::core::poly::circle::{CanonicCoset, PolyOps};
use crate::core::prover::{prove, verify, VerificationError};
use crate::core::vcs::blake2_merkle::{Blake2sMerkleChannel, Blake2sMerkleHasher};

#[allow(unused)]
pub fn prove_state_machine(
    log_n_rows: u32,
    initial_state: State,
    config: PcsConfig,
    channel: &mut Blake2sChannel,
) -> (
    StateMachineComponents,
    StateMachineProof<Blake2sMerkleHasher>,
) {
    let (x_axis_log_rows, y_axis_log_rows) = (log_n_rows, log_n_rows - 1);
    let (x_row, y_row) = (34, 56);
    assert!(y_axis_log_rows >= LOG_N_LANES && x_axis_log_rows >= LOG_N_LANES);
    assert!(x_row < 1 << x_axis_log_rows);
    assert!(y_row < 1 << y_axis_log_rows);

    let mut intermediate_state = initial_state;
    intermediate_state[0] += M31::from_u32_unchecked(x_row);
    let mut final_state = intermediate_state;
    final_state[1] += M31::from_u32_unchecked(y_row);

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
    let trace_op0 = gen_trace(x_axis_log_rows, initial_state, 0);
    let trace_op1 = gen_trace(y_axis_log_rows, intermediate_state, 1);

    let stmt0 = StateMachineStatement0 {
        n: x_axis_log_rows,
        m: y_axis_log_rows,
    };
    stmt0.mix_into(channel);

    let mut tree_builder = commitment_scheme.tree_builder();
    tree_builder.extend_evals(chain![trace_op0.clone(), trace_op1.clone()]);
    tree_builder.commit(channel);

    // Draw lookup element.
    let lookup_elements = StateMachineElements::draw(channel);

    // Interaction trace.
    let (interaction_trace_op0, [total_sum_op0, claimed_sum_op0]) =
        gen_interaction_trace(x_row as usize - 1, &trace_op0, 0, &lookup_elements);
    let (interaction_trace_op1, [total_sum_op1, claimed_sum_op1]) =
        gen_interaction_trace(y_row as usize - 1, &trace_op1, 1, &lookup_elements);

    let stmt1 = StateMachineStatement1 {
        x_axis_claimed_sum: claimed_sum_op0,
        y_axis_claimed_sum: claimed_sum_op1,
    };
    stmt1.mix_into(channel);

    let mut tree_builder = commitment_scheme.tree_builder();
    tree_builder.extend_evals(chain![interaction_trace_op0, interaction_trace_op1].collect_vec());
    tree_builder.commit(channel);

    // Constant trace.
    let mut tree_builder = commitment_scheme.tree_builder();
    tree_builder.extend_evals(vec![
        gen_is_first(x_axis_log_rows),
        gen_is_first(y_axis_log_rows),
    ]);
    tree_builder.commit(channel);

    // Prove constraints.
    let mut tree_span_provider = &mut TraceLocationAllocator::default();
    let component0 = StateMachineOp0Component::new(
        tree_span_provider,
        StateTransitionEval {
            log_n_rows: x_axis_log_rows,
            lookup_elements: lookup_elements.clone(),
            total_sum: total_sum_op0,
            claimed_sum: (claimed_sum_op0, x_row as usize - 1),
        },
    );
    let component1 = StateMachineOp1Component::new(
        tree_span_provider,
        StateTransitionEval {
            log_n_rows: y_axis_log_rows,
            lookup_elements,
            total_sum: total_sum_op1,
            claimed_sum: (claimed_sum_op1, y_row as usize - 1),
        },
    );
    let components = StateMachineComponents {
        component0,
        component1,
    };
    let stark_proof = prove(&components.component_provers(), channel, commitment_scheme).unwrap();
    let proof = StateMachineProof {
        public_input: [initial_state, final_state],
        stmt0,
        stmt1,
        stark_proof,
    };
    (components, proof)
}

pub fn verify_state_machine(
    config: PcsConfig,
    channel: &mut Blake2sChannel,
    components: StateMachineComponents,
    proof: StateMachineProof<Blake2sMerkleHasher>,
) -> Result<(), VerificationError> {
    let commitment_scheme = &mut CommitmentSchemeVerifier::<Blake2sMerkleChannel>::new(config);
    // Decommit.
    // Retrieve the expected column sizes in each commitment interaction, from the AIR.
    let sizes = proof.stmt0.log_sizes();
    // Trace columns.
    proof.stmt0.mix_into(channel);
    commitment_scheme.commit(proof.stark_proof.commitments[0], &sizes[0], channel);

    // Assert state machine statement.
    let lookup_elements = StateMachineElements::draw(channel);
    let initial_state_comb: QM31 = lookup_elements.combine(&proof.public_input[0]);
    let final_state_comb: QM31 = lookup_elements.combine(&proof.public_input[1]);
    assert_eq!(
        (proof.stmt1.x_axis_claimed_sum + proof.stmt1.y_axis_claimed_sum)
            * initial_state_comb
            * final_state_comb,
        final_state_comb - initial_state_comb
    );

    // Interaction columns.
    proof.stmt1.mix_into(channel);
    commitment_scheme.commit(proof.stark_proof.commitments[1], &sizes[1], channel);
    // Constant columns.
    commitment_scheme.commit(proof.stark_proof.commitments[2], &sizes[2], channel);

    verify(
        &components.components(),
        channel,
        commitment_scheme,
        proof.stark_proof,
    )
}

#[cfg(test)]
mod tests {
    use num_traits::Zero;

    use super::components::{
        StateMachineElements, StateMachineOp0Component, StateTransitionEval, STATE_SIZE,
    };
    use super::gen::{gen_interaction_trace, gen_trace};
    use super::{prove_state_machine, verify_state_machine};
    use crate::constraint_framework::preprocessed_columns::gen_is_first;
    use crate::constraint_framework::{assert_constraints, FrameworkEval, TraceLocationAllocator};
    use crate::core::channel::Blake2sChannel;
    use crate::core::fields::m31::M31;
    use crate::core::fields::qm31::QM31;
    use crate::core::fields::FieldExpOps;
    use crate::core::pcs::{PcsConfig, TreeVec};
    use crate::core::poly::circle::CanonicCoset;

    #[test]
    fn test_state_machine_constraints() {
        let log_n_rows = 8;
        let initial_state = [M31::zero(); STATE_SIZE];

        let trace = gen_trace(log_n_rows, initial_state, 0);
        let lookup_elements = StateMachineElements::draw(&mut Blake2sChannel::default());

        // Interaction trace.
        let (interaction_trace, [total_sum, claimed_sum]) =
            gen_interaction_trace(1 << log_n_rows, &trace, 0, &lookup_elements);

        assert_eq!(total_sum, claimed_sum);
        let component = StateMachineOp0Component::new(
            &mut TraceLocationAllocator::default(),
            StateTransitionEval {
                log_n_rows,
                lookup_elements,
                total_sum,
                claimed_sum: (total_sum, (1 << log_n_rows) - 1),
            },
        );

        let trace = TreeVec::new(vec![
            trace,
            interaction_trace,
            vec![gen_is_first(log_n_rows)],
        ]);
        let trace_polys = trace.map_cols(|c| c.interpolate());
        assert_constraints(&trace_polys, CanonicCoset::new(log_n_rows), |eval| {
            component.evaluate(eval);
        });
    }

    #[test]
    fn test_state_machine_claimed_sum() {
        let log_n_rows = 8;
        let config = PcsConfig::default();

        // Initial and last state.
        let initial_state = [M31::zero(); STATE_SIZE];
        let last_state = [M31::from_u32_unchecked(34), M31::from_u32_unchecked(56)];

        // Setup protocol.
        let channel = &mut Blake2sChannel::default();
        let (component, _) = prove_state_machine(log_n_rows, initial_state, config, channel);

        let interaction_elements = component.component0.lookup_elements.clone();
        let initial_state_comb: QM31 = interaction_elements.combine(&initial_state);
        let last_state_comb: QM31 = interaction_elements.combine(&last_state);

        assert_eq!(
            component.component0.claimed_sum.0 + component.component1.claimed_sum.0,
            initial_state_comb.inverse() - last_state_comb.inverse()
        );
    }

    #[test]
    fn test_state_machine_prove() {
        let log_n_rows = 8;
        let config = PcsConfig::default();
        let initial_state = [M31::zero(); STATE_SIZE];
        let prover_channel = &mut Blake2sChannel::default();
        let verifier_channel = &mut Blake2sChannel::default();

        let (components, proof) =
            prove_state_machine(log_n_rows, initial_state, config, prover_channel);

        verify_state_machine(config, verifier_channel, components, proof).unwrap();
    }
}
