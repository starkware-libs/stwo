use itertools::Itertools;
use stwo::core::fields::m31::BaseField;
use stwo::core::fields::FieldExpOps;
use stwo::core::poly::circle::CanonicCoset;
use stwo::core::ColumnVec;
use stwo::prover::backend::simd::m31::PackedBaseField;
use stwo::prover::backend::simd::SimdBackend;
use stwo::prover::backend::{Backend, Col, Column};
use stwo::prover::poly::circle::CircleEvaluation;
use stwo::prover::poly::BitReversedOrder;
use stwo_constraint_framework::{EvalAtRow, FrameworkComponent, FrameworkEval};

pub type WideFibonacciComponent<const N: usize> = FrameworkComponent<WideFibonacciEval<N>>;

mod fib_with_preprocessed;

pub struct FibInput {
    pub a: BaseField,
    pub b: BaseField,
}

pub struct FibInputSimd {
    a: PackedBaseField,
    b: PackedBaseField,
}

pub fn generate_trace<const N: usize, B: Backend>(
    inputs: &[FibInput],
) -> ColumnVec<CircleEvaluation<B, BaseField, BitReversedOrder>> {
    assert!(inputs.len().is_power_of_two());
    let log_size = inputs.len().ilog2();
    let mut trace = (0..N)
        .map(|_| Col::<B, BaseField>::zeros(1 << log_size))
        .collect_vec();
    for (vec_index, input) in inputs.iter().enumerate() {
        let mut a = input.a;
        let mut b = input.b;
        trace[0].set(vec_index, a);
        trace[1].set(vec_index, b);
        trace.iter_mut().skip(2).for_each(|col| {
            (a, b) = (b, a.square() + b.square());
            col.set(vec_index, b);
        });
    }
    let domain = CanonicCoset::new(log_size).circle_domain();
    trace
        .into_iter()
        .map(|eval| CircleEvaluation::<B, _, BitReversedOrder>::new(domain, eval))
        .collect_vec()
}

/// Same as [`generate_trace`] but optimized for simd.
pub fn generate_trace_simd<const N: usize>(
    log_size: u32,
    inputs: &[FibInputSimd],
) -> ColumnVec<CircleEvaluation<SimdBackend, BaseField, BitReversedOrder>> {
    let mut trace = (0..N)
        .map(|_| Col::<SimdBackend, BaseField>::zeros(1 << log_size))
        .collect_vec();
    for (vec_index, input) in inputs.iter().enumerate() {
        let mut a = input.a;
        let mut b = input.b;
        trace[0].data[vec_index] = a;
        trace[1].data[vec_index] = b;
        trace.iter_mut().skip(2).for_each(|col| {
            (a, b) = (b, a.square() + b.square());
            col.data[vec_index] = b;
        });
    }
    let domain = CanonicCoset::new(log_size).circle_domain();
    trace
        .into_iter()
        .map(|eval| CircleEvaluation::<SimdBackend, _, BitReversedOrder>::new(domain, eval))
        .collect_vec()
}

/// A component that enforces the Fibonacci sequence.
/// Each row contains a separate Fibonacci sequence of length `N`.
#[derive(Clone)]
pub struct WideFibonacciEval<const N: usize> {
    pub log_n_rows: u32,
}
impl<const N: usize> FrameworkEval for WideFibonacciEval<N> {
    fn log_size(&self) -> u32 {
        self.log_n_rows
    }
    fn max_constraint_log_degree_bound(&self) -> u32 {
        self.log_n_rows + 1
    }
    fn evaluate<E: EvalAtRow>(&self, mut eval: E) -> E {
        let mut a = eval.next_trace_mask();
        let mut b = eval.next_trace_mask();
        for _ in 2..N {
            let c = eval.next_trace_mask();
            eval.add_constraint(c.clone() - (a.square() + b.square()));
            a = b;
            b = c;
        }
        eval
    }
}

#[cfg(test)]
mod tests {
    use itertools::Itertools;
    use num_traits::{One, Zero};
    use stwo::core::air::Component;
    use stwo::core::channel::Blake2sM31Channel;
    #[cfg(not(target_arch = "wasm32"))]
    use stwo::core::channel::Poseidon252Channel;
    use stwo::core::fields::m31::BaseField;
    use stwo::core::fields::qm31::SecureField;
    use stwo::core::pcs::{CommitmentSchemeVerifier, PcsConfig, TreeVec};
    use stwo::core::poly::circle::CanonicCoset;
    use stwo::core::vcs_lifted::blake2_merkle::Blake2sM31MerkleChannel;
    #[cfg(not(target_arch = "wasm32"))]
    use stwo::core::vcs_lifted::poseidon252_merkle::Poseidon252MerkleChannel;
    use stwo::core::verifier::verify;
    use stwo::prover::backend::simd::SimdBackend;
    use stwo::prover::backend::{Column, CpuBackend};
    use stwo::prover::poly::circle::PolyOps;
    use stwo::prover::{prove, CommitmentSchemeProver};
    use stwo_constraint_framework::{
        assert_constraints_on_polys, AssertEvaluator, FrameworkEval, TraceLocationAllocator,
    };

    use super::WideFibonacciEval;
    use crate::wide_fibonacci::{generate_trace, FibInput, WideFibonacciComponent};

    const FIB_SEQUENCE_LENGTH: usize = 100;

    fn generate_test_inputs(log_n_instances: u32) -> Vec<FibInput> {
        (0..1 << log_n_instances)
            .map(|i| FibInput {
                a: BaseField::one(),
                b: BaseField::from_u32_unchecked(i as u32),
            })
            .collect_vec()
    }

    fn fibonacci_constraint_evaluator<const N: u32>(eval: AssertEvaluator<'_>) {
        WideFibonacciEval::<FIB_SEQUENCE_LENGTH> { log_n_rows: N }.evaluate(eval);
    }

    #[test]
    fn test_wide_fibonacci_constraints() {
        const LOG_N_INSTANCES: u32 = 6;
        let traces = TreeVec::new(vec![
            vec![],
            generate_trace::<FIB_SEQUENCE_LENGTH, SimdBackend>(&generate_test_inputs(
                LOG_N_INSTANCES,
            )),
        ]);
        let trace_polys =
            traces.map(|trace| trace.into_iter().map(|c| c.interpolate()).collect_vec());

        assert_constraints_on_polys(
            &trace_polys,
            CanonicCoset::new(LOG_N_INSTANCES),
            fibonacci_constraint_evaluator::<LOG_N_INSTANCES>,
            SecureField::zero(),
        );
    }

    #[test]
    #[should_panic]
    fn test_wide_fibonacci_constraints_fails() {
        const LOG_N_INSTANCES: u32 = 6;

        let mut trace = generate_trace::<FIB_SEQUENCE_LENGTH, SimdBackend>(&generate_test_inputs(
            LOG_N_INSTANCES,
        ));
        // Modify the trace such that a constraint fail.
        trace[17].values.set(2, BaseField::one());
        let traces = TreeVec::new(vec![vec![], trace]);
        let trace_polys =
            traces.map(|trace| trace.into_iter().map(|c| c.interpolate()).collect_vec());

        assert_constraints_on_polys(
            &trace_polys,
            CanonicCoset::new(LOG_N_INSTANCES),
            fibonacci_constraint_evaluator::<LOG_N_INSTANCES>,
            SecureField::zero(),
        );
    }

    #[test_log::test]
    fn test_wide_fib_prove_with_blake() {
        for log_n_instances in 4..=8 {
            let config = PcsConfig::default();
            // Precompute twiddles.
            let twiddles = SimdBackend::precompute_twiddles(
                CanonicCoset::new(log_n_instances + 1 + config.fri_config.log_blowup_factor)
                    .circle_domain()
                    .half_coset,
            );

            // Setup protocol.
            let prover_channel = &mut Blake2sM31Channel::default();
            let mut commitment_scheme = CommitmentSchemeProver::<
                SimdBackend,
                Blake2sM31MerkleChannel,
            >::new(config, &twiddles);

            // Preprocessed trace
            let mut tree_builder = commitment_scheme.tree_builder();
            tree_builder.extend_evals(vec![]);
            tree_builder.commit(prover_channel);

            // Trace.
            let trace =
                generate_trace::<FIB_SEQUENCE_LENGTH, _>(&generate_test_inputs(log_n_instances));
            let mut tree_builder = commitment_scheme.tree_builder();
            tree_builder.extend_evals(trace);
            tree_builder.commit(prover_channel);

            // Prove constraints.
            let component = WideFibonacciComponent::new(
                &mut TraceLocationAllocator::default(),
                WideFibonacciEval::<FIB_SEQUENCE_LENGTH> {
                    log_n_rows: log_n_instances,
                },
                SecureField::zero(),
            );

            let proof = prove::<SimdBackend, Blake2sM31MerkleChannel>(
                &[&component],
                prover_channel,
                commitment_scheme,
            )
            .unwrap();

            // Verify.
            let verifier_channel = &mut Blake2sM31Channel::default();
            let commitment_scheme =
                &mut CommitmentSchemeVerifier::<Blake2sM31MerkleChannel>::new(config);

            // Retrieve the expected column sizes in each commitment interaction, from the AIR.
            let sizes = component.trace_log_degree_bounds();
            commitment_scheme.commit(proof.commitments[0], &sizes[0], verifier_channel);
            commitment_scheme.commit(proof.commitments[1], &sizes[1], verifier_channel);
            verify(&[&component], verifier_channel, commitment_scheme, proof).unwrap();
        }
    }

    #[test]
    #[cfg(not(target_arch = "wasm32"))]
    fn test_wide_fib_prove_with_poseidon() {
        const LOG_N_INSTANCES: u32 = 6;
        let config = PcsConfig::default();
        // Precompute twiddles.
        let twiddles = SimdBackend::precompute_twiddles(
            CanonicCoset::new(LOG_N_INSTANCES + 1 + config.fri_config.log_blowup_factor)
                .circle_domain()
                .half_coset,
        );

        // Setup protocol.
        let prover_channel = &mut Poseidon252Channel::default();
        let mut commitment_scheme =
            CommitmentSchemeProver::<SimdBackend, Poseidon252MerkleChannel>::new(config, &twiddles);

        // TODO(ilya): remove the following once preprocessed columns are not mandatory.
        // Preprocessed trace
        let mut tree_builder = commitment_scheme.tree_builder();
        tree_builder.extend_evals(vec![]);
        tree_builder.commit(prover_channel);

        // Trace.
        let trace =
            generate_trace::<FIB_SEQUENCE_LENGTH, _>(&generate_test_inputs(LOG_N_INSTANCES));
        let mut tree_builder = commitment_scheme.tree_builder();
        tree_builder.extend_evals(trace);
        tree_builder.commit(prover_channel);

        // Prove constraints.
        let component = WideFibonacciComponent::new(
            &mut TraceLocationAllocator::default(),
            WideFibonacciEval::<FIB_SEQUENCE_LENGTH> {
                log_n_rows: LOG_N_INSTANCES,
            },
            SecureField::zero(),
        );
        let proof = prove::<SimdBackend, Poseidon252MerkleChannel>(
            &[&component],
            prover_channel,
            commitment_scheme,
        )
        .unwrap();

        // Verify.
        let verifier_channel = &mut Poseidon252Channel::default();
        let commitment_scheme =
            &mut CommitmentSchemeVerifier::<Poseidon252MerkleChannel>::new(proof.config);

        // Retrieve the expected column sizes in each commitment interaction, from the AIR.
        let sizes = component.trace_log_degree_bounds();
        commitment_scheme.commit(proof.commitments[0], &sizes[0], verifier_channel);
        commitment_scheme.commit(proof.commitments[1], &sizes[1], verifier_channel);
        verify(&[&component], verifier_channel, commitment_scheme, proof).unwrap();
    }

    #[test]
    fn test_e2e_lifted_fib_prove() {
        const LOG_SIZE_SHORT: u32 = 3;
        const LOG_SIZE_LONG: u32 = 9;

        const N_ROWS_LONG_COMPONENT: usize = 4;
        const N_ROWS_SHORT_COMPONENT: usize = 5;

        let config = PcsConfig::default();
        // Precompute twiddles.
        let twiddles = CpuBackend::precompute_twiddles(
            CanonicCoset::new(LOG_SIZE_LONG + config.fri_config.log_blowup_factor)
                .circle_domain()
                .half_coset,
        );

        // Setup protocol.
        let prover_channel = &mut Blake2sM31Channel::default();
        let mut commitment_scheme =
            CommitmentSchemeProver::<CpuBackend, Blake2sM31MerkleChannel>::new(config, &twiddles);
        commitment_scheme.set_store_polynomials_coefficients();
        // Preprocessed trace
        let mut tree_builder = commitment_scheme.tree_builder();
        tree_builder.extend_evals(vec![]);
        tree_builder.commit(prover_channel);

        // Trace.
        let trace = [
            generate_trace::<N_ROWS_LONG_COMPONENT, _>(&generate_test_inputs(LOG_SIZE_LONG)),
            generate_trace::<N_ROWS_SHORT_COMPONENT, _>(&generate_test_inputs(LOG_SIZE_SHORT)),
        ]
        .concat();

        let mut tree_builder = commitment_scheme.tree_builder();
        tree_builder.extend_evals(trace);
        tree_builder.commit(prover_channel);

        // Generate components.
        let mut trace_alloc = TraceLocationAllocator::default();
        let component0 = WideFibonacciComponent::new(
            &mut trace_alloc,
            WideFibonacciEval::<N_ROWS_LONG_COMPONENT> {
                log_n_rows: LOG_SIZE_LONG,
            },
            SecureField::zero(),
        );
        let component1 = WideFibonacciComponent::new(
            &mut trace_alloc,
            WideFibonacciEval::<N_ROWS_SHORT_COMPONENT> {
                log_n_rows: LOG_SIZE_SHORT,
            },
            SecureField::zero(),
        );

        // Prove.
        let proof = prove::<CpuBackend, Blake2sM31MerkleChannel>(
            &[&component0, &component1],
            prover_channel,
            commitment_scheme,
        )
        .unwrap();

        // Verify.
        let verifier_channel = &mut Blake2sM31Channel::default();
        let commitment_scheme =
            &mut CommitmentSchemeVerifier::<Blake2sM31MerkleChannel>::new(config);

        let trace_sizes = [
            vec![LOG_SIZE_LONG; N_ROWS_LONG_COMPONENT],
            vec![LOG_SIZE_SHORT; N_ROWS_SHORT_COMPONENT],
        ]
        .concat();
        // Retrieve the expected column sizes in each commitment interaction, from the AIR.
        let sizes = TreeVec::new(vec![vec![], trace_sizes]);
        commitment_scheme.commit(proof.commitments[0], &sizes[0], verifier_channel);
        commitment_scheme.commit(proof.commitments[1], &sizes[1], verifier_channel);

        assert!(verify(
            &[&component0, &component1],
            verifier_channel,
            commitment_scheme,
            proof,
        )
        .is_ok());
    }
}
