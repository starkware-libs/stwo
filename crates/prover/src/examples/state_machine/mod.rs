use crate::constraint_framework::relation_tracker::RelationSummary;
use crate::constraint_framework::Relation;
pub mod components;
pub mod gen;

use components::{
    track_state_machine_relations, State, StateMachineComponents, StateMachineElements,
    StateMachineOp0Component, StateMachineOp1Component, StateMachineProof, StateMachineStatement0,
    StateMachineStatement1, StateTransitionEval,
};
use gen::{gen_interaction_trace, gen_trace};
use itertools::{chain, Itertools};

use crate::constraint_framework::TraceLocationAllocator;
use crate::core::backend::simd::m31::LOG_N_LANES;
use crate::core::backend::simd::SimdBackend;
use crate::core::channel::Blake2sChannel;
use crate::core::fields::m31::M31;
use crate::core::fields::qm31::QM31;
use crate::core::pcs::{CommitmentSchemeProver, CommitmentSchemeVerifier, PcsConfig, TreeVec};
use crate::core::poly::circle::{CanonicCoset, PolyOps};
use crate::core::prover::{prove, verify, VerificationError};
use crate::core::vcs::blake2_merkle::{Blake2sMerkleChannel, Blake2sMerkleHasher};

#[allow(unused)]
pub fn prove_state_machine(
    log_n_rows: u32,
    initial_state: State,
    config: PcsConfig,
    channel: &mut Blake2sChannel,
    track_relations: bool,
) -> (
    StateMachineComponents,
    StateMachineProof<Blake2sMerkleHasher>,
    Option<RelationSummary>,
) {
    let (x_axis_log_rows, y_axis_log_rows) = (log_n_rows, log_n_rows - 1);
    assert!(y_axis_log_rows >= LOG_N_LANES && x_axis_log_rows >= LOG_N_LANES);

    let mut intermediate_state = initial_state;
    intermediate_state[0] += M31::from_u32_unchecked(1 << x_axis_log_rows);
    let mut final_state = intermediate_state;
    final_state[1] += M31::from_u32_unchecked(1 << y_axis_log_rows);

    // Precompute twiddles.
    let twiddles = SimdBackend::precompute_twiddles(
        CanonicCoset::new(log_n_rows + config.fri_config.log_blowup_factor + 1)
            .circle_domain()
            .half_coset,
    );

    // Setup protocol.
    let mut commitment_scheme =
        CommitmentSchemeProver::<_, Blake2sMerkleChannel>::new(config, &twiddles);

    // Trace.
    let trace_op0 = gen_trace(x_axis_log_rows, initial_state, 0);
    let trace_op1 = gen_trace(y_axis_log_rows, intermediate_state, 1);

    let trace = chain![trace_op0.clone(), trace_op1.clone()].collect_vec();

    let relation_summary = match track_relations {
        false => None,
        true => Some(RelationSummary::summarize_relations(
            &track_state_machine_relations(
                &TreeVec(vec![&vec![], &trace]),
                x_axis_log_rows,
                y_axis_log_rows,
            ),
        )),
    };

    // Commitments.
    let mut tree_builder = commitment_scheme.tree_builder();
    tree_builder.commit(channel);

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
    let (interaction_trace_op0, claimed_sum_op0) =
        gen_interaction_trace(&trace_op0, 0, &lookup_elements);
    let (interaction_trace_op1, claimed_sum_op1) =
        gen_interaction_trace(&trace_op1, 1, &lookup_elements);

    let stmt1 = StateMachineStatement1 {
        x_axis_claimed_sum: claimed_sum_op0,
        y_axis_claimed_sum: claimed_sum_op1,
    };
    stmt1.mix_into(channel);

    let mut tree_builder = commitment_scheme.tree_builder();
    tree_builder.extend_evals(chain![interaction_trace_op0, interaction_trace_op1].collect_vec());
    tree_builder.commit(channel);

    // Prove constraints.
    let mut tree_span_provider = &mut TraceLocationAllocator::default();
    let component0 = StateMachineOp0Component::new(
        tree_span_provider,
        StateTransitionEval {
            log_n_rows: x_axis_log_rows,
            lookup_elements: lookup_elements.clone(),
            claimed_sum: claimed_sum_op0,
        },
        claimed_sum_op0,
    );
    let component1 = StateMachineOp1Component::new(
        tree_span_provider,
        StateTransitionEval {
            log_n_rows: y_axis_log_rows,
            lookup_elements,
            claimed_sum: claimed_sum_op1,
        },
        claimed_sum_op1,
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
    (components, proof, relation_summary)
}

pub fn verify_state_machine(
    channel: &mut Blake2sChannel,
    components: StateMachineComponents,
    proof: StateMachineProof<Blake2sMerkleHasher>,
) -> Result<(), VerificationError> {
    let commitment_scheme =
        &mut CommitmentSchemeVerifier::<Blake2sMerkleChannel>::new(proof.stark_proof.config);
    // Decommit.
    // Retrieve the expected column sizes in each commitment interaction, from the AIR.
    let sizes = proof.stmt0.log_sizes();

    // Preprocessed columns.
    commitment_scheme.commit(proof.stark_proof.commitments[0], &sizes[0], channel);
    // Trace columns.
    proof.stmt0.mix_into(channel);
    commitment_scheme.commit(proof.stark_proof.commitments[1], &sizes[1], channel);

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
    use crate::constraint_framework::expr::ExprEvaluator;
    use crate::constraint_framework::{
        assert_constraints, FrameworkEval, Relation, TraceLocationAllocator,
    };
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
        let (interaction_trace, claimed_sum) = gen_interaction_trace(&trace, 0, &lookup_elements);

        let component = StateMachineOp0Component::new(
            &mut TraceLocationAllocator::default(),
            StateTransitionEval {
                log_n_rows,
                lookup_elements,
                claimed_sum,
            },
            claimed_sum,
        );

        let trace = TreeVec::new(vec![vec![], trace, interaction_trace]);
        let trace_polys = trace.map_cols(|c| c.interpolate());
        assert_constraints(
            &trace_polys,
            CanonicCoset::new(log_n_rows),
            |eval| {
                component.evaluate(eval);
            },
            claimed_sum,
        );
    }

    #[test]
    fn test_state_machine_claimed_sum() {
        let log_n_rows = 8;
        let config = PcsConfig::default();

        // Initial and last state.
        let initial_state = [M31::zero(); STATE_SIZE];
        let last_state = [
            M31::from_u32_unchecked(1 << log_n_rows),
            M31::from_u32_unchecked(1 << (log_n_rows - 1)),
        ];

        // Setup protocol.
        let channel = &mut Blake2sChannel::default();
        let (component, ..) =
            prove_state_machine(log_n_rows, initial_state, config, channel, false);

        let interaction_elements = component.component0.lookup_elements.clone();
        let initial_state_comb: QM31 = interaction_elements.combine(&initial_state);
        let last_state_comb: QM31 = interaction_elements.combine(&last_state);

        assert_eq!(
            component.component0.claimed_sum + component.component1.claimed_sum,
            initial_state_comb.inverse() - last_state_comb.inverse()
        );
    }

    #[test]
    fn test_relation_tracker() {
        let log_n_rows = 8;
        let config = PcsConfig::default();
        let initial_state = [M31::zero(); STATE_SIZE];
        let final_state = [
            M31::from_u32_unchecked(1 << log_n_rows),
            M31::from_u32_unchecked(1 << (log_n_rows - 1)),
        ];

        // Summarize `StateMachineElements`.
        let (_, _, summary) = prove_state_machine(
            log_n_rows,
            initial_state,
            config,
            &mut Blake2sChannel::default(),
            true,
        );
        let summary = summary.unwrap();
        let relation_info = summary.get_relation_info("StateMachineElements").unwrap();

        // Check the final state inferred from the summary.
        let mut curr_state = initial_state;
        for entry in relation_info {
            let (x_step, y_step) = match entry.0.len() {
                2 => (entry.0[0], entry.0[1]),
                1 => (entry.0[0], M31::zero()),
                0 => (M31::zero(), M31::zero()),
                _ => unreachable!(),
            };
            let mult = entry.1;
            let next_state = [curr_state[0] - x_step * mult, curr_state[1] - y_step * mult];

            curr_state = next_state;
        }

        assert_eq!(curr_state, final_state);
    }

    #[test]
    fn test_state_machine_prove() {
        let log_n_rows = 8;
        let config = PcsConfig::default();
        let initial_state = [M31::zero(); STATE_SIZE];
        let prover_channel = &mut Blake2sChannel::default();
        let verifier_channel = &mut Blake2sChannel::default();

        let (components, proof, _) =
            prove_state_machine(log_n_rows, initial_state, config, prover_channel, false);

        verify_state_machine(verifier_channel, components, proof).unwrap();
    }

    #[test]
    fn test_state_machine_constraint_repr() {
        let log_n_rows = 8;
        let initial_state = [M31::zero(); STATE_SIZE];

        let trace = gen_trace(log_n_rows, initial_state, 0);
        let lookup_elements = StateMachineElements::draw(&mut Blake2sChannel::default());

        let (_, claimed_sum) = gen_interaction_trace(&trace, 0, &lookup_elements);

        let component = StateMachineOp0Component::new(
            &mut TraceLocationAllocator::default(),
            StateTransitionEval {
                log_n_rows,
                lookup_elements,
                claimed_sum,
            },
            claimed_sum,
        );

        let eval = component.evaluate(ExprEvaluator::new());
        let expected = "let intermediate0 = (StateMachineElements_alpha0) * (trace_1_column_0_offset_0) \
            + (StateMachineElements_alpha1) * (trace_1_column_1_offset_0) \
            - (StateMachineElements_z);

\
        let intermediate1 = (StateMachineElements_alpha0) * (trace_1_column_0_offset_0 + m31(1).into()) \
            + (StateMachineElements_alpha1) * (trace_1_column_1_offset_0) \
            - (StateMachineElements_z);

\
        let constraint_0 = (QM31Impl::from_partial_evals([trace_2_column_2_offset_0, trace_2_column_3_offset_0, trace_2_column_4_offset_0, trace_2_column_5_offset_0]) \
            - (QM31Impl::from_partial_evals([trace_2_column_2_offset_neg_1, trace_2_column_3_offset_neg_1, trace_2_column_4_offset_neg_1, trace_2_column_5_offset_neg_1])) \
                + (claimed_sum) * (1 / (column_size))\
            ) \
            * ((intermediate0) * (intermediate1)) \
            - (intermediate1 - (intermediate0));"
            .to_string();

        assert_eq!(eval.format_constraints(), expected);
    }
}
