//! Dual Component Circuit - Computing & Scheduling
//!
//! Computing: Generates inputs and computes x^5+1 with constraints
//! Scheduling: Copies data from Computing via LogUp

use itertools::chain;
use num_traits::One;
use stwo::core::air::Component;
use stwo::core::channel::Channel;
use stwo::core::fields::m31::M31;
use stwo::core::fields::qm31::SecureField;
use stwo::core::fields::FieldExpOps;
use stwo::core::pcs::TreeVec;
use stwo::core::poly::circle::CanonicCoset;
use stwo::prover::backend::simd::column::BaseColumn;
use stwo::prover::backend::simd::m31::LOG_N_LANES;
use stwo::prover::backend::simd::qm31::PackedSecureField;
use stwo::prover::backend::simd::SimdBackend;
use stwo::prover::poly::circle::CircleEvaluation;
use stwo::prover::poly::BitReversedOrder;
use stwo::prover::ComponentProver;
use stwo_constraint_framework::{
    relation, EvalAtRow, FrameworkComponent, FrameworkEval, InfoEvaluator, LogupTraceGenerator,
    Relation, RelationEntry, TraceLocationAllocator, PREPROCESSED_TRACE_IDX,
};

const CONSTRAINT_EVAL_BLOWUP_FACTOR: u32 = 1;

relation!(ComputationLookupElements, 1);

pub type SchedulingComponent = FrameworkComponent<SchedulingEval>;
pub type ComputingComponent = FrameworkComponent<ComputingEval>;

pub struct Components {
    scheduling_component: SchedulingComponent,
    computing_component: ComputingComponent,
}

impl Components {
    pub fn new(
        statement0: &ComponentsStatement0,
        lookup_elements: &ComputationLookupElements,
        statement1: &ComponentsStatement1,
    ) -> Self {
        let tree_span_provider = &mut TraceLocationAllocator::default();

        let scheduling_component = SchedulingComponent::new(
            tree_span_provider,
            SchedulingEval {
                log_size: statement0.log_size,
                lookup_elements: lookup_elements.clone(),
            },
            statement1.scheduling_claimed_sum,
        );

        let computing_component = ComputingComponent::new(
            tree_span_provider,
            ComputingEval {
                log_size: statement0.log_size,
                lookup_elements: lookup_elements.clone(),
            },
            statement1.computing_claimed_sum,
        );

        Self {
            scheduling_component,
            computing_component,
        }
    }

    pub fn components(&self) -> Vec<&dyn Component> {
        chain![[
            &self.scheduling_component as &dyn Component,
            &self.computing_component as &dyn Component
        ]]
        .collect()
    }

    pub fn component_provers(&self) -> Vec<&dyn ComponentProver<SimdBackend>> {
        chain![[
            &self.scheduling_component as &dyn ComponentProver<SimdBackend>,
            &self.computing_component as &dyn ComponentProver<SimdBackend>
        ]]
        .collect()
    }
}

#[derive(Debug, Clone)]
pub struct ComponentsStatement0 {
    pub log_size: u32,
}

impl ComponentsStatement0 {
    pub fn mix_into(&self, channel: &mut impl Channel) {
        channel.mix_u64(self.log_size as u64);
    }

    pub fn log_sizes(&self) -> TreeVec<Vec<u32>> {
        let mut log_sizes = vec![];

        log_sizes.push(
            scheduling_info()
                .mask_offsets
                .as_cols_ref()
                .map_cols(|_| self.log_size),
        );

        log_sizes.push(
            computing_info()
                .mask_offsets
                .as_cols_ref()
                .map_cols(|_| self.log_size),
        );

        let mut tree_vec = TreeVec::concat_cols(log_sizes.into_iter());
        tree_vec[PREPROCESSED_TRACE_IDX] = vec![];
        tree_vec
    }
}

#[derive(Debug, Clone)]
pub struct ComponentsStatement1 {
    pub scheduling_claimed_sum: SecureField,
    pub computing_claimed_sum: SecureField,
}

impl ComponentsStatement1 {
    pub fn mix_into(&self, channel: &mut impl Channel) {
        channel.mix_felts(&[self.scheduling_claimed_sum, self.computing_claimed_sum]);
    }
}

fn scheduling_info() -> InfoEvaluator {
    let component = SchedulingEval {
        log_size: 1,
        lookup_elements: ComputationLookupElements::dummy(),
    };
    component.evaluate(InfoEvaluator::empty())
}

fn computing_info() -> InfoEvaluator {
    let component = ComputingEval {
        log_size: 1,
        lookup_elements: ComputationLookupElements::dummy(),
    };
    component.evaluate(InfoEvaluator::empty())
}

#[derive(Clone)]
pub struct SchedulingEval {
    pub log_size: u32,
    pub lookup_elements: ComputationLookupElements,
}

impl FrameworkEval for SchedulingEval {
    fn log_size(&self) -> u32 {
        self.log_size
    }

    fn max_constraint_log_degree_bound(&self) -> u32 {
        self.log_size + CONSTRAINT_EVAL_BLOWUP_FACTOR
    }

    fn evaluate<E: EvalAtRow>(&self, mut eval: E) -> E {
        let input_col = eval.next_trace_mask();
        let output_col = eval.next_trace_mask();

        eval.add_to_relation(RelationEntry::new(
            &self.lookup_elements,
            E::EF::one(),
            &[input_col],
        ));

        eval.add_to_relation(RelationEntry::new(
            &self.lookup_elements,
            -E::EF::one(),
            &[output_col],
        ));

        eval.finalize_logup_in_pairs();
        eval
    }
}

#[derive(Clone)]
pub struct ComputingEval {
    pub log_size: u32,
    pub lookup_elements: ComputationLookupElements,
}

impl FrameworkEval for ComputingEval {
    fn log_size(&self) -> u32 {
        self.log_size
    }

    fn max_constraint_log_degree_bound(&self) -> u32 {
        self.log_size + CONSTRAINT_EVAL_BLOWUP_FACTOR
    }

    fn evaluate<E: EvalAtRow>(&self, mut eval: E) -> E {
        let input_col = eval.next_trace_mask();
        let intermediate_col = eval.next_trace_mask();
        let output_col = eval.next_trace_mask();

        // Constraint 1: intermediate = input^3
        eval.add_constraint(
            intermediate_col.clone() - input_col.clone() * input_col.clone() * input_col.clone(),
        );

        // Constraint 2: output = input^5 + 1
        eval.add_constraint(
            output_col.clone()
                - intermediate_col.clone() * input_col.clone() * input_col.clone()
                - E::F::one(),
        );

        eval.add_to_relation(RelationEntry::new(
            &self.lookup_elements,
            -E::EF::one(),
            &[input_col],
        ));

        eval.add_to_relation(RelationEntry::new(
            &self.lookup_elements,
            E::EF::one(),
            &[output_col],
        ));

        eval.finalize_logup_in_pairs();
        eval
    }
}

pub struct ComputationLookupData {
    pub inputs: CircleEvaluation<SimdBackend, M31, BitReversedOrder>,
    pub outputs: CircleEvaluation<SimdBackend, M31, BitReversedOrder>,
}

pub fn gen_computing_trace(
    log_size: u32,
) -> (
    Vec<CircleEvaluation<SimdBackend, M31, BitReversedOrder>>,
    ComputationLookupData,
) {
    let input_col = BaseColumn::from_iter((0..(1 << log_size)).map(|_| M31::from(10)));
    let intermediate_col = BaseColumn::from_iter(input_col.as_slice().iter().map(|&v| v.pow(3)));
    let output_col = BaseColumn::from_iter(
        input_col
            .as_slice()
            .iter()
            .map(|&v| v.pow(5) + M31::from(1)),
    );

    let domain = CanonicCoset::new(log_size).circle_domain();

    let input_eval = CircleEvaluation::new(domain, input_col.clone());
    let intermediate_eval = CircleEvaluation::new(domain, intermediate_col);
    let output_eval = CircleEvaluation::new(domain, output_col.clone());

    let lookup_data = ComputationLookupData {
        inputs: CircleEvaluation::new(domain, input_col),
        outputs: CircleEvaluation::new(domain, output_col),
    };

    (
        vec![input_eval, intermediate_eval, output_eval],
        lookup_data,
    )
}

pub fn gen_scheduling_trace(
    _log_size: u32,
    computation_lookup_data: &ComputationLookupData,
) -> Vec<CircleEvaluation<SimdBackend, M31, BitReversedOrder>> {
    vec![
        computation_lookup_data.inputs.clone(),
        computation_lookup_data.outputs.clone(),
    ]
}

pub fn gen_scheduling_logup_trace(
    log_size: u32,
    scheduling_col_1: &CircleEvaluation<SimdBackend, M31, BitReversedOrder>,
    scheduling_col_2: &CircleEvaluation<SimdBackend, M31, BitReversedOrder>,
    lookup_elements: &ComputationLookupElements,
) -> (
    Vec<CircleEvaluation<SimdBackend, M31, BitReversedOrder>>,
    SecureField,
) {
    let mut logup_gen = LogupTraceGenerator::new(log_size);

    let mut col_gen = logup_gen.new_col();
    for row in 0..(1 << (log_size - LOG_N_LANES)) {
        let scheduling_input: PackedSecureField =
            lookup_elements.combine(&[scheduling_col_1.data[row]]);
        let scheduling_output: PackedSecureField =
            lookup_elements.combine(&[scheduling_col_2.data[row]]);
        col_gen.write_frac(
            row,
            scheduling_output - scheduling_input,
            scheduling_input * scheduling_output,
        );
    }
    col_gen.finalize_col();

    logup_gen.finalize_last()
}

pub fn gen_computing_logup_trace(
    log_size: u32,
    computing_col_1: &CircleEvaluation<SimdBackend, M31, BitReversedOrder>,
    computing_col_3: &CircleEvaluation<SimdBackend, M31, BitReversedOrder>,
    lookup_elements: &ComputationLookupElements,
) -> (
    Vec<CircleEvaluation<SimdBackend, M31, BitReversedOrder>>,
    SecureField,
) {
    let mut logup_gen = LogupTraceGenerator::new(log_size);

    let mut col_gen = logup_gen.new_col();
    for row in 0..(1 << (log_size - LOG_N_LANES)) {
        let computing_input: PackedSecureField =
            lookup_elements.combine(&[computing_col_1.data[row]]);
        let computing_output: PackedSecureField =
            lookup_elements.combine(&[computing_col_3.data[row]]);
        col_gen.write_frac(
            row,
            computing_input - computing_output,
            computing_input * computing_output,
        );
    }
    col_gen.finalize_col();

    logup_gen.finalize_last()
}
