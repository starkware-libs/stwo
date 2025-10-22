use itertools::chain;
use num_traits::identities::Zero;
use num_traits::One;
use rand::rngs::SmallRng;
use rand::{Rng, SeedableRng};
use stwo::core::air::Component;
use stwo::core::channel::{Blake2sChannel, Channel};
use stwo::core::fields::m31::M31;
use stwo::core::fields::qm31::SecureField;
use stwo::core::fields::FieldExpOps;
use stwo::core::pcs::{CommitmentSchemeVerifier, PcsConfig, TreeVec};
use stwo::core::poly::circle::CanonicCoset;
use stwo::core::proof::StarkProof;
use stwo::core::vcs::blake2_merkle::Blake2sMerkleChannel;
use stwo::core::vcs::MerkleHasher;
use stwo::core::verifier::verify;
use stwo::prover::backend::simd::column::BaseColumn;
use stwo::prover::backend::simd::m31::LOG_N_LANES;
use stwo::prover::backend::simd::qm31::PackedSecureField;
use stwo::prover::backend::simd::SimdBackend;
use stwo::prover::poly::circle::{CircleEvaluation, PolyOps};
use stwo::prover::poly::BitReversedOrder;
use stwo::prover::{prove, CommitmentSchemeProver, ComponentProver};
use stwo_constraint_framework::{
    relation, EvalAtRow, FrameworkComponent, FrameworkEval, InfoEvaluator, LogupTraceGenerator,
    Relation, RelationEntry, TraceLocationAllocator,
};

#[allow(dead_code)]
struct ComponentsProof<H: MerkleHasher> {
    statement0: ComponentsStatement0,
    statement1: ComponentsStatement1,
    stark_proof: StarkProof<H>,
}

pub struct Components {
    scheduling_component: FrameworkComponent<SchedulingEval>,
    computing_component: FrameworkComponent<ComputingEval>,
}

impl Components {
    pub fn new(
        statement0: &ComponentsStatement0,
        lookup_elements: &ComputationLookupElements,
        statement1: &ComponentsStatement1,
    ) -> Self {
        let tree_span_provider =
            &mut TraceLocationAllocator::new_with_preprocessed_columns(&vec![]);

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

pub struct ComponentsStatement0 {
    log_size: u32,
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

        TreeVec::concat_cols(log_sizes.into_iter())
    }
}

pub struct ComponentsStatement1 {
    scheduling_claimed_sum: SecureField,
    computing_claimed_sum: SecureField,
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

pub type SchedulingComponent = FrameworkComponent<SchedulingEval>;
pub type ComputingComponent = FrameworkComponent<ComputingEval>;

pub struct SchedulingEval {
    log_size: u32,
    lookup_elements: ComputationLookupElements,
}

// ANCHOR: scheduling_eval_start
impl FrameworkEval for SchedulingEval {
    // ANCHOR_END: scheduling_eval_start
    fn log_size(&self) -> u32 {
        self.log_size
    }

    fn max_constraint_log_degree_bound(&self) -> u32 {
        self.log_size + CONSTRAINT_EVAL_BLOWUP_FACTOR
    }

    // ANCHOR: scheduling_eval_evaluate
    // --snip--

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
// ANCHOR_END: scheduling_eval_evaluate

pub struct ComputingEval {
    log_size: u32,
    lookup_elements: ComputationLookupElements,
}

// ANCHOR: computing_eval_start
impl FrameworkEval for ComputingEval {
    // ANCHOR_END: computing_eval_start
    fn log_size(&self) -> u32 {
        self.log_size
    }

    fn max_constraint_log_degree_bound(&self) -> u32 {
        self.log_size + CONSTRAINT_EVAL_BLOWUP_FACTOR
    }

    // ANCHOR: computing_eval_evaluate
    // --snip--

    fn evaluate<E: EvalAtRow>(&self, mut eval: E) -> E {
        let input_col = eval.next_trace_mask();
        let intermediate_col = eval.next_trace_mask();
        let output_col = eval.next_trace_mask();

        eval.add_constraint(
            intermediate_col.clone() - input_col.clone() * input_col.clone() * input_col.clone(),
        );
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
// ANCHOR_END: computing_eval_evaluate

const CONSTRAINT_EVAL_BLOWUP_FACTOR: u32 = 1;

relation!(ComputationLookupElements, 1);

fn gen_scheduling_trace(
    log_size: u32,
) -> Vec<CircleEvaluation<SimdBackend, M31, BitReversedOrder>> {
    // Create a table with random values
    let mut rng = SmallRng::from_seed([42; 32]);
    let scheduling_col_1 =
        BaseColumn::from_iter((0..(1 << log_size)).map(|_| M31::from(rng.gen_range(0..16))));
    let scheduling_col_2 = BaseColumn::from_iter(
        scheduling_col_1
            .as_slice()
            .iter()
            .map(|&v| v.pow(5) + M31::from(1)),
    );

    // Convert table to trace polynomials
    let domain = CanonicCoset::new(log_size).circle_domain();

    vec![scheduling_col_1, scheduling_col_2]
        .into_iter()
        .map(|col| CircleEvaluation::new(domain, col))
        .collect()
}

fn gen_computing_trace(
    log_size: u32,
    scheduling_col_1: &CircleEvaluation<SimdBackend, M31, BitReversedOrder>,
    scheduling_col_2: &CircleEvaluation<SimdBackend, M31, BitReversedOrder>,
) -> Vec<CircleEvaluation<SimdBackend, M31, BitReversedOrder>> {
    let intermediate_values = scheduling_col_1
        .as_slice()
        .iter()
        .map(|&v| v.pow(3))
        .collect::<Vec<_>>();
    let intermediate_trace = CircleEvaluation::new(
        CanonicCoset::new(log_size).circle_domain(),
        BaseColumn::from_iter(intermediate_values),
    );

    vec![
        scheduling_col_1.clone(),
        intermediate_trace,
        scheduling_col_2.clone(),
    ]
}

// ANCHOR: gen_scheduling_logup_trace_start
fn gen_scheduling_logup_trace(
    log_size: u32,
    scheduling_col_1: &CircleEvaluation<SimdBackend, M31, BitReversedOrder>,
    scheduling_col_2: &CircleEvaluation<SimdBackend, M31, BitReversedOrder>,
    lookup_elements: &ComputationLookupElements,
) -> (
    Vec<CircleEvaluation<SimdBackend, M31, BitReversedOrder>>,
    SecureField,
) {
    // ANCHOR_END: gen_scheduling_logup_trace_start
    let mut logup_gen = LogupTraceGenerator::new(log_size);

    let mut col_gen = logup_gen.new_col();
    for row in 0..(1 << (log_size - LOG_N_LANES)) {
        // ANCHOR: gen_scheduling_logup_trace_row
        // --snip--

        let scheduling_input: PackedSecureField =
            lookup_elements.combine(&[scheduling_col_1.data[row]]);
        let scheduling_output: PackedSecureField =
            lookup_elements.combine(&[scheduling_col_2.data[row]]);
        col_gen.write_frac(
            row,
            scheduling_output - scheduling_input,
            scheduling_input * scheduling_output,
        );

        // --snip--
        // ANCHOR_END: gen_scheduling_logup_trace_row
    }
    col_gen.finalize_col();

    logup_gen.finalize_last()
    // ANCHOR_END: gen_scheduling_logup_trace_end
}
// ANCHOR_END: gen_scheduling_logup_trace_end

// ANCHOR: gen_computing_logup_trace_start
fn gen_computing_logup_trace(
    log_size: u32,
    computing_col_1: &CircleEvaluation<SimdBackend, M31, BitReversedOrder>,
    computing_col_3: &CircleEvaluation<SimdBackend, M31, BitReversedOrder>,
    lookup_elements: &ComputationLookupElements,
) -> (
    Vec<CircleEvaluation<SimdBackend, M31, BitReversedOrder>>,
    SecureField,
) {
    // ANCHOR_END: gen_computing_logup_trace_start
    let mut logup_gen = LogupTraceGenerator::new(log_size);

    let mut col_gen = logup_gen.new_col();
    for row in 0..(1 << (log_size - LOG_N_LANES)) {
        // ANCHOR: gen_computing_logup_trace_row
        // --snip--

        let computing_input: PackedSecureField =
            lookup_elements.combine(&[computing_col_1.data[row]]);
        let computing_output: PackedSecureField =
            lookup_elements.combine(&[computing_col_3.data[row]]);
        col_gen.write_frac(
            row,
            computing_input - computing_output,
            computing_input * computing_output,
        );

        // --snip--
        // ANCHOR_END: gen_computing_logup_trace_row
    }
    col_gen.finalize_col();

    logup_gen.finalize_last()
    // ANCHOR: gen_computing_logup_trace_end
}
// ANCHOR_END: gen_computing_logup_trace_end

// ANCHOR: main_start
pub fn main() {
    // ANCHOR_END: main_start
    let log_size = LOG_N_LANES;

    // Config for FRI and PoW
    let config = PcsConfig::default();

    // Precompute twiddles for evaluating and interpolating the trace
    let twiddles = SimdBackend::precompute_twiddles(
        CanonicCoset::new(
            log_size + CONSTRAINT_EVAL_BLOWUP_FACTOR + config.fri_config.log_blowup_factor,
        )
        .circle_domain()
        .half_coset,
    );

    // Create the channel and commitment scheme
    let channel = &mut Blake2sChannel::default();
    let mut commitment_scheme =
        CommitmentSchemeProver::<SimdBackend, Blake2sMerkleChannel>::new(config, &twiddles);

    // Create and commit to the preprocessed columns
    let mut tree_builder = commitment_scheme.tree_builder();
    tree_builder.extend_evals(vec![]);
    tree_builder.commit(channel);

    // ANCHOR: main_prove
    // --snip--

    // Create trace columns
    let scheduling_trace = gen_scheduling_trace(log_size);
    let computing_trace = gen_computing_trace(log_size, &scheduling_trace[0], &scheduling_trace[1]);

    // Statement 0
    let statement0 = ComponentsStatement0 { log_size };
    statement0.mix_into(channel);

    // Commit to the trace columns
    let mut tree_builder = commitment_scheme.tree_builder();
    tree_builder.extend_evals([scheduling_trace.clone(), computing_trace.clone()].concat());
    tree_builder.commit(channel);

    // Draw random elements to use when creating the random linear combination of lookup values in
    // the LogUp columns
    let lookup_elements = ComputationLookupElements::draw(channel);

    // Create LogUp columns
    let (scheduling_logup_cols, scheduling_claimed_sum) = gen_scheduling_logup_trace(
        log_size,
        &scheduling_trace[0],
        &scheduling_trace[1],
        &lookup_elements,
    );
    let (computing_logup_cols, computing_claimed_sum) = gen_computing_logup_trace(
        log_size,
        &computing_trace[0],
        &computing_trace[2],
        &lookup_elements,
    );

    // Statement 1
    let statement1 = ComponentsStatement1 {
        scheduling_claimed_sum,
        computing_claimed_sum,
    };
    statement1.mix_into(channel);

    // Commit to the LogUp columns
    let mut tree_builder = commitment_scheme.tree_builder();
    tree_builder.extend_evals([scheduling_logup_cols, computing_logup_cols].concat());
    tree_builder.commit(channel);

    let components = Components::new(&statement0, &lookup_elements, &statement1);

    let stark_proof = prove(&components.component_provers(), channel, commitment_scheme).unwrap();

    let proof = ComponentsProof {
        statement0,
        statement1,
        stark_proof,
    };

    // --snip--
    // ANCHOR_END: main_prove

    // ANCHOR: main_verify
    // --snip--

    // Verify claimed sums
    assert_eq!(
        scheduling_claimed_sum + computing_claimed_sum,
        SecureField::zero()
    );

    // Unpack proof
    let statement0 = proof.statement0;
    let statement1 = proof.statement1;
    let stark_proof = proof.stark_proof;

    // Create channel and commitment scheme
    let channel = &mut Blake2sChannel::default();
    let commitment_scheme = &mut CommitmentSchemeVerifier::<Blake2sMerkleChannel>::new(config);
    let log_sizes = statement0.log_sizes();

    // Preprocessed columns.
    commitment_scheme.commit(stark_proof.commitments[0], &log_sizes[0], channel);

    // Commit to statement 0
    statement0.mix_into(channel);

    // Trace columns.
    commitment_scheme.commit(stark_proof.commitments[1], &log_sizes[1], channel);

    // Draw lookup element.
    let lookup_elements = ComputationLookupElements::draw(channel);

    // Commit to statement 1
    statement1.mix_into(channel);

    // Interaction columns.
    commitment_scheme.commit(stark_proof.commitments[2], &log_sizes[2], channel);

    // Create components
    let components = Components::new(&statement0, &lookup_elements, &statement1);

    verify(
        &components.components(),
        channel,
        commitment_scheme,
        stark_proof,
    )
    .unwrap();
    // ANCHOR_END: main_verify

    // ANCHOR: main_end
}
// ANCHOR_END: main_end
