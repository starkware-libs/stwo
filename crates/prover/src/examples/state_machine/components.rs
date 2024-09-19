use num_traits::{One, Zero};

use crate::constraint_framework::logup::{LogupAtRow, LookupElements};
use crate::constraint_framework::{EvalAtRow, FrameworkComponent, FrameworkEval, InfoEvaluator};
use crate::core::air::{Component, ComponentProver};
use crate::core::backend::simd::SimdBackend;
use crate::core::channel::Channel;
use crate::core::fields::m31::M31;
use crate::core::fields::qm31::{SecureField, QM31};
use crate::core::lookups::utils::Fraction;
use crate::core::pcs::TreeVec;
use crate::core::prover::StarkProof;
use crate::core::vcs::ops::MerkleHasher;

pub const N_STATE: usize = 2;
pub type StateMachineElements = LookupElements<N_STATE>;
pub type State = [M31; N_STATE];

pub type StateMachineOp0Component = FrameworkComponent<StateTransitionEval<0>>;
pub type StateMachineOp1Component = FrameworkComponent<StateTransitionEval<1>>;

/// State machine with state of size `N_STATE`.
/// Transition `INDEX` of state increments the state by 1 at that offset.
#[derive(Clone)]
pub struct StateTransitionEval<const INDEX: usize> {
    pub log_n_rows: u32,
    pub lookup_elements: StateMachineElements,
    pub total_sum: QM31,
}

impl<const INDEX: usize> FrameworkEval for StateTransitionEval<INDEX> {
    fn log_size(&self) -> u32 {
        self.log_n_rows
    }
    fn max_constraint_log_degree_bound(&self) -> u32 {
        self.log_n_rows + 1
    }
    fn evaluate<E: EvalAtRow>(&self, mut eval: E) -> E {
        let [is_first] = eval.next_interaction_mask(2, [0]);
        let mut logup: LogupAtRow<E> = LogupAtRow::new(1, self.total_sum, None, is_first);

        let input_state: [_; N_STATE] = std::array::from_fn(|_| eval.next_trace_mask());
        let q0: E::EF = self.lookup_elements.combine(&input_state);

        let mut output_state = input_state;
        output_state[INDEX] += E::F::one();
        let q1: E::EF = self.lookup_elements.combine(&output_state);

        // Add to the total sum (1/q0 - 1/q1).
        logup.write_frac(&mut eval, Fraction::new(q1 - q0, q1 * q0));

        logup.finalize(&mut eval);
        eval
    }
}

pub struct StateMachineStatement0 {
    pub n: u32,
    pub m: u32,
}
impl StateMachineStatement0 {
    pub fn log_sizes(&self) -> TreeVec<Vec<u32>> {
        let sizes = vec![
            state_transition_info::<0>()
                .mask_offsets
                .as_cols_ref()
                .map_cols(|_| self.n),
            state_transition_info::<1>()
                .mask_offsets
                .as_cols_ref()
                .map_cols(|_| self.m),
        ];
        TreeVec::concat_cols(sizes.into_iter())
    }
    pub fn mix_into(&self, channel: &mut impl Channel) {
        channel.mix_u64(self.n as u64);
        channel.mix_u64(self.m as u64);
    }
}

pub struct StateMachineStatement1 {
    pub x_axis_claimed_sum: SecureField,
    pub y_axis_claimed_sum: SecureField,
}
impl StateMachineStatement1 {
    pub fn mix_into(&self, channel: &mut impl Channel) {
        channel.mix_felts(&[self.x_axis_claimed_sum, self.y_axis_claimed_sum])
    }
}

fn state_transition_info<const INDEX: usize>() -> InfoEvaluator {
    let component = StateTransitionEval::<INDEX> {
        log_n_rows: 1,
        lookup_elements: StateMachineElements::dummy(),
        total_sum: QM31::zero(),
    };
    component.evaluate(InfoEvaluator::default())
}

pub struct StateMachineComponents {
    pub component0: StateMachineOp0Component,
    pub component1: StateMachineOp1Component,
}

impl StateMachineComponents {
    pub fn components(&self) -> Vec<&dyn Component> {
        vec![
            &self.component0 as &dyn Component,
            &self.component1 as &dyn Component,
        ]
    }

    pub fn component_provers(&self) -> Vec<&dyn ComponentProver<SimdBackend>> {
        vec![
            &self.component0 as &dyn ComponentProver<SimdBackend>,
            &self.component1 as &dyn ComponentProver<SimdBackend>,
        ]
    }
}

pub struct StateMachineProof<H: MerkleHasher> {
    pub public_input: [State; 2], // Initial and final state.
    pub stmt0: StateMachineStatement0,
    pub stmt1: StateMachineStatement1,
    pub stark_proof: StarkProof<H>,
}
