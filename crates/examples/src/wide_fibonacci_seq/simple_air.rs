use itertools::{Itertools, zip_eq};
use num_traits::One;
use stwo::core::ColumnVec;
use stwo::core::channel::Blake2sM31Channel;
use stwo::core::fields::FieldExpOps;
use stwo::core::fields::m31::BaseField;
use stwo::core::fields::qm31::{SecureField};
use stwo::core::pcs::PcsConfig;
use stwo::core::poly::circle::CanonicCoset;
use stwo::core::proof::ExtendedStarkProof;
use stwo::core::vcs_lifted::blake2_merkle::{Blake2sM31MerkleChannel, Blake2sM31MerkleHasher};
use stwo::prover::backend::{Col, CpuBackend};
use stwo::prover::backend::Column;
use stwo::prover::backend::simd::SimdBackend;
use stwo::prover::backend::simd::column::BaseColumn;
use stwo::prover::backend::simd::m31::{LOG_N_LANES, PackedBaseField};
use stwo::prover::backend::simd::qm31::PackedSecureField;
use stwo::prover::poly::BitReversedOrder;
use stwo::prover::poly::circle::{CircleEvaluation, PolyOps};
use stwo::prover::{CommitmentSchemeProver, prove_ex};
use stwo_constraint_framework::logup::LookupElements;
use stwo_constraint_framework::preprocessed_columns::PreProcessedColumnId;
use stwo_constraint_framework::{
    EvalAtRow, FrameworkComponent, FrameworkEval, LogupTraceGenerator, RelationEntry,
    TraceLocationAllocator, relation,
};

#[cfg(test)]
#[path = "simple_air_test.rs"]
pub mod simple_air_test;

pub const FIB_SEQUENCE_LENGTH: usize = 4;
pub const LOG_N_INSTANCES: u32 = 4;
const _: () = assert!(LOG_N_INSTANCES >= LOG_N_LANES);

relation!(SimpleRelation, 2);

pub type SimpleComponent = FrameworkComponent<Eval>;

pub struct FibInput {
    a: PackedBaseField,
    b: PackedBaseField,
}

/// A component that enforces a variation of the Fibonacci sequence:
///    `a_{i + 2} = a_{i + 1}^2 + a_{i}^2 + row_const`.
///
/// The `row_const` is a different constant for each row.
///
/// The last two elements of the sequence are added to the relation.
/// Each row contains a separate sequence of length `N`.
#[derive(Clone)]
pub struct Eval {
    pub lookup_elements: SimpleRelation,
}
impl FrameworkEval for Eval {
    fn log_size(&self) -> u32 {
        LOG_N_INSTANCES
    }
    fn max_constraint_log_degree_bound(&self) -> u32 {
        LOG_N_INSTANCES + 1
    }
    fn evaluate<E: EvalAtRow>(&self, mut eval: E) -> E {
        let row_const =
            eval.get_preprocessed_column(PreProcessedColumnId { id: "row_const".into() });
        let mut a = eval.next_trace_mask();
        let mut b = eval.next_trace_mask();
        for _ in 2..FIB_SEQUENCE_LENGTH {
            let c = eval.next_trace_mask();
            eval.add_constraint(c.clone() - (a.square() + b.square() + row_const.clone()));
            a = b;
            b = c;
        }

        eval.add_to_relation(RelationEntry::new(&self.lookup_elements, E::EF::one(), &[a, b]));
        eval.finalize_logup();

        eval
    }
}

/// Generates a trace for the test.
fn generate_trace() -> ColumnVec<CircleEvaluation<SimdBackend, BaseField, BitReversedOrder>> {
    let inputs = (0..(1 << (LOG_N_INSTANCES - LOG_N_LANES)))
        .map(|i| FibInput {
            a: PackedBaseField::one(),
            b: PackedBaseField::from_array(std::array::from_fn(|j| {
                BaseField::from_u32_unchecked((i * 16 + j) as u32)
            })),
        })
        .collect_vec();

    let mut trace = (0..FIB_SEQUENCE_LENGTH)
        .map(|_| Col::<SimdBackend, BaseField>::zeros(1 << LOG_N_INSTANCES))
        .collect_vec();
    let row_const: BaseColumn = (0..(1 << LOG_N_INSTANCES)).map(|i| i.into()).collect();
    for (vec_index, (input, row_const)) in zip_eq(inputs, row_const.data).enumerate() {
        let mut a = input.a;
        let mut b = input.b;
        trace[0].data[vec_index] = a;
        trace[1].data[vec_index] = b;
        trace.iter_mut().skip(2).for_each(|col| {
            (a, b) = (b, a.square() + b.square() + row_const);
            col.data[vec_index] = b;
        });
    }
    let domain = CanonicCoset::new(LOG_N_INSTANCES).circle_domain();
    trace
        .into_iter()
        .map(|eval| CircleEvaluation::<SimdBackend, _, BitReversedOrder>::new(domain, eval))
        .collect_vec()
}

/// Generates the interaction trace for the test.
pub fn generate_interaction_trace(
    trace: &ColumnVec<CircleEvaluation<SimdBackend, BaseField, BitReversedOrder>>,
    lookup_elements: &LookupElements<2>,
) -> (ColumnVec<CircleEvaluation<SimdBackend, BaseField, BitReversedOrder>>, SecureField) {
    let mut logup_gen = LogupTraceGenerator::new(LOG_N_INSTANCES);

    let mut col_gen = logup_gen.new_col();
    for vec_row in 0..(1 << (LOG_N_INSTANCES - LOG_N_LANES)) {
        let denom: PackedSecureField = lookup_elements.combine(&[
            trace[FIB_SEQUENCE_LENGTH - 2].values.data[vec_row],
            trace[FIB_SEQUENCE_LENGTH - 1].values.data[vec_row],
        ]);
        col_gen.write_frac(vec_row, PackedSecureField::one(), denom);
    }
    col_gen.finalize_col();

    logup_gen.finalize_last()
}

/// Creates a proof for the simple AIR. See documentation in [Eval].
pub fn create_proof() -> (SimpleComponent, ExtendedStarkProof<Blake2sM31MerkleHasher>) {
    let config = PcsConfig::default();
    // Precompute twiddles.
    let twiddles = CpuBackend::precompute_twiddles(
        CanonicCoset::new(LOG_N_INSTANCES + 1 + config.fri_config.log_blowup_factor)
            .circle_domain()
            .half_coset,
    );


    // Setup protocol.
    let prover_channel = &mut Blake2sM31Channel::default();
    let mut commitment_scheme =
        CommitmentSchemeProver::<CpuBackend, Blake2sM31MerkleChannel>::new(config, &twiddles);

    // Preprocessed trace
    let domain = CanonicCoset::new(LOG_N_INSTANCES).circle_domain();
    let domain_big = CanonicCoset::new(LOG_N_INSTANCES + 1).circle_domain();
    let mut tree_builder = commitment_scheme.tree_builder();
    let preprocessed_column: BaseColumn =
        (0..2_u32.pow(LOG_N_INSTANCES)).map(|i| i.into()).collect();
    let preprocessed_column_big: BaseColumn =
        (0..2_u32.pow(LOG_N_INSTANCES + 1)).map(|i| i.into()).collect();
    let tmp2 = preprocessed_column_big.to_cpu();
    let tmp = preprocessed_column.to_cpu();
    let preprocessed_column_eval = CircleEvaluation::<CpuBackend, _, _>::new(domain, tmp);
    let preprocessed_column_eval_big = CircleEvaluation::<CpuBackend, _, _>::new(domain_big, tmp2);
    
    tree_builder.extend_evals([preprocessed_column_eval, preprocessed_column_eval_big]);
    tree_builder.commit(prover_channel);

    // let mut tree_builder = commitment_scheme.tree_builder();
    // Trace.
    let trace = generate_trace();
    let mut tree_builder = commitment_scheme.tree_builder();
    tree_builder.extend_evals(trace.iter().map(|col| col.to_cpu()));
    tree_builder.commit(prover_channel);

    // TODO(lior): Add proof of work before drawing the lookup elements.

    // Draw lookup element.
    let lookup_elements = SimpleRelation::draw(prover_channel);

    // Interaction trace.
    let (interaction_trace, claimed_sum) = generate_interaction_trace(&trace, &lookup_elements.0);
    let mut tree_builder = commitment_scheme.tree_builder();
    tree_builder.extend_evals(interaction_trace.iter().map(|col| col.to_cpu()));
    tree_builder.commit(prover_channel);

    let mut trace_location_allocator = TraceLocationAllocator::new_with_preprocessed_columns(&[PreProcessedColumnId { id: "row_const".into() }, PreProcessedColumnId { id: "row_const_big".into() }]);
    // Prove constraints.
    let component = SimpleComponent::new(
        &mut trace_location_allocator,
        Eval { lookup_elements },
        claimed_sum,
    );

    let proof = prove_ex::<CpuBackend, Blake2sM31MerkleChannel>(
        &[&component],
        prover_channel,
        commitment_scheme,
    )
    .unwrap();

    (component, proof)
}
