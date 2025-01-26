use itertools::Itertools;

use crate::constraint_framework::{EvalAtRow, FrameworkComponent, FrameworkEval};
use crate::core::backend::simd::m31::PackedBaseField;
use crate::core::backend::simd::SimdBackend;
use crate::core::backend::{Col, Column};
use crate::core::fields::m31::BaseField;
use crate::core::fields::FieldExpOps;
use crate::core::poly::circle::{CanonicCoset, CircleEvaluation};
use crate::core::poly::BitReversedOrder;
use crate::core::ColumnVec;

pub type WideFibonacciComponent<const N: usize> = FrameworkComponent<WideFibonacciEval<N>>;

pub struct FibInput {
    a: PackedBaseField,
    b: PackedBaseField,
}

/// A component that enforces the Fibonacci sequence.
/// Each row contains a seperate Fibonacci sequence of length `N`.
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

pub fn generate_trace<const N: usize>(
    log_size: u32,
    inputs: &[FibInput],
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

#[cfg(test)]
mod tests {
    use itertools::Itertools;
    use num_traits::{One, Zero};

    use super::WideFibonacciEval;
    use crate::constraint_framework::{
        assert_constraints, AssertEvaluator, FrameworkEval, TraceLocationAllocator,
    };
    use crate::core::air::Component;
    use crate::core::backend::simd::m31::{PackedBaseField, LOG_N_LANES};
    use crate::core::backend::simd::SimdBackend;
    use crate::core::backend::Column;
    use crate::core::channel::Blake2sChannel;
    #[cfg(not(target_arch = "wasm32"))]
    use crate::core::channel::Poseidon252Channel;
    use crate::core::fields::m31::BaseField;
    use crate::core::fields::qm31::SecureField;
    use crate::core::pcs::{CommitmentSchemeProver, CommitmentSchemeVerifier, PcsConfig, TreeVec};
    use crate::core::poly::circle::{CanonicCoset, CircleEvaluation, PolyOps};
    use crate::core::poly::BitReversedOrder;
    use crate::core::prover::{prove, verify};
    use crate::core::vcs::blake2_merkle::Blake2sMerkleChannel;
    #[cfg(not(target_arch = "wasm32"))]
    use crate::core::vcs::poseidon252_merkle::Poseidon252MerkleChannel;
    use crate::core::ColumnVec;
    use crate::examples::wide_fibonacci::{generate_trace, FibInput, WideFibonacciComponent};

    const FIB_SEQUENCE_LENGTH: usize = 100;

    fn generate_test_trace(
        log_n_instances: u32,
    ) -> ColumnVec<CircleEvaluation<SimdBackend, BaseField, BitReversedOrder>> {
        if log_n_instances < LOG_N_LANES {
            let n_instances = 1 << log_n_instances;
            let inputs = vec![FibInput {
                a: PackedBaseField::from_array(std::array::from_fn(|j| {
                    if j < n_instances {
                        BaseField::one()
                    } else {
                        BaseField::zero()
                    }
                })),
                b: PackedBaseField::from_array(std::array::from_fn(|j| {
                    if j < n_instances {
                        BaseField::from_u32_unchecked((j) as u32)
                    } else {
                        BaseField::zero()
                    }
                })),
            }];
            return generate_trace::<FIB_SEQUENCE_LENGTH>(log_n_instances, &inputs);
        }
        let inputs = (0..(1 << (log_n_instances - LOG_N_LANES)))
            .map(|i| FibInput {
                a: PackedBaseField::one(),
                b: PackedBaseField::from_array(std::array::from_fn(|j| {
                    BaseField::from_u32_unchecked((i * 16 + j) as u32)
                })),
            })
            .collect_vec();
        generate_trace::<FIB_SEQUENCE_LENGTH>(log_n_instances, &inputs)
    }

    fn fibonacci_constraint_evaluator<const N: u32>(eval: AssertEvaluator<'_>) {
        WideFibonacciEval::<FIB_SEQUENCE_LENGTH> { log_n_rows: N }.evaluate(eval);
    }

    #[test]
    fn test_wide_fibonacci_constraints() {
        const LOG_N_INSTANCES: u32 = 6;
        let traces = TreeVec::new(vec![vec![], generate_test_trace(LOG_N_INSTANCES)]);
        let trace_polys =
            traces.map(|trace| trace.into_iter().map(|c| c.interpolate()).collect_vec());

        assert_constraints(
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

        let mut trace = generate_test_trace(LOG_N_INSTANCES);
        // Modify the trace such that a constraint fail.
        trace[17].values.set(2, BaseField::one());
        let traces = TreeVec::new(vec![vec![], trace]);
        let trace_polys =
            traces.map(|trace| trace.into_iter().map(|c| c.interpolate()).collect_vec());

        assert_constraints(
            &trace_polys,
            CanonicCoset::new(LOG_N_INSTANCES),
            fibonacci_constraint_evaluator::<LOG_N_INSTANCES>,
            SecureField::zero(),
        );
    }

    #[test_log::test]
    fn test_wide_fib_prove_with_blake() {
        for log_n_instances in 2..=6 {
            let config = PcsConfig::default();
            // Precompute twiddles.
            let twiddles = SimdBackend::precompute_twiddles(
                CanonicCoset::new(log_n_instances + 1 + config.fri_config.log_blowup_factor)
                    .circle_domain()
                    .half_coset,
            );

            // Setup protocol.
            let prover_channel = &mut Blake2sChannel::default();
            let mut commitment_scheme =
                CommitmentSchemeProver::<SimdBackend, Blake2sMerkleChannel>::new(config, &twiddles);

            // Preprocessed trace
            let mut tree_builder = commitment_scheme.tree_builder();
            tree_builder.extend_evals([]);
            tree_builder.commit(prover_channel);

            // Trace.
            let trace = generate_test_trace(log_n_instances);
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

            let proof = prove::<SimdBackend, Blake2sMerkleChannel>(
                &[&component],
                prover_channel,
                commitment_scheme,
            )
            .unwrap();

            // Verify.
            let verifier_channel = &mut Blake2sChannel::default();
            let commitment_scheme =
                &mut CommitmentSchemeVerifier::<Blake2sMerkleChannel>::new(config);

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

        // TODO(ilya): remove the following once preproccessed columns are not mandatory.
        // Preprocessed trace
        let mut tree_builder = commitment_scheme.tree_builder();
        tree_builder.extend_evals([]);
        tree_builder.commit(prover_channel);

        // Trace.
        let trace = generate_test_trace(LOG_N_INSTANCES);
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
}
