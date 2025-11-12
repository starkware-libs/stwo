use itertools::Itertools;
use stwo::core::fields::m31::BaseField;
use stwo::core::fields::FieldExpOps;
use stwo::core::poly::circle::CanonicCoset;
use stwo::core::ColumnVec;
use stwo::prover::backend::{Backend, Col, Column};
use stwo::prover::poly::circle::CircleEvaluation;
use stwo::prover::poly::BitReversedOrder;

pub struct FibInput {
    pub a: BaseField,
    pub b: BaseField,
}

#[allow(dead_code)]
fn generate_trace<const N: usize, B: Backend>(
    log_size: u32,
    inputs: &[FibInput],
) -> ColumnVec<CircleEvaluation<B, BaseField, BitReversedOrder>> {
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

#[cfg(test)]
mod tests {
    use itertools::{chain, Itertools};
    use num_traits::{One, Zero};
    use stwo::core::channel::Blake2sM31Channel;
    #[cfg(not(target_arch = "wasm32"))]
    use stwo::core::fields::m31::BaseField;
    use stwo::core::fields::qm31::SecureField;
    use stwo::core::pcs::{CommitmentSchemeVerifier, PcsConfig, TreeVec};
    use stwo::core::poly::circle::CanonicCoset;
    use stwo::core::vcs_lifted::blake2_merkle::Blake2sM31MerkleChannel;
    use stwo::core::verifier::verify;
    #[cfg(not(target_arch = "wasm32"))]
    use stwo::core::ColumnVec;
    use stwo::prover::backend::CpuBackend;
    use stwo::prover::poly::circle::{CircleEvaluation, PolyOps};
    use stwo::prover::poly::BitReversedOrder;
    use stwo::prover::{prove, CommitmentSchemeProver};
    use stwo_constraint_framework::TraceLocationAllocator;

    use super::{generate_trace, FibInput};
    use crate::wide_fibonacci::{WideFibonacciComponent, WideFibonacciEval};

    // Consts must by >= 2;
    const N_ROWS_SHORT_COMPONENT: usize = 3;
    const N_ROWS_LONG_COMPONENT: usize = 5;

    fn generate_test_trace_mixed(
        log_sizes: (u32, u32),
    ) -> ColumnVec<CircleEvaluation<CpuBackend, BaseField, BitReversedOrder>> {
        assert!(log_sizes.0 <= log_sizes.1);
        let input_0 = (0..1 << log_sizes.0)
            .map(|i| FibInput {
                a: BaseField::one(),
                b: BaseField::from_u32_unchecked(i as u32),
            })
            .collect_vec();
        let input_1 = (0..1 << log_sizes.1)
            .map(|i| FibInput {
                a: BaseField::one(),
                b: BaseField::from_u32_unchecked(100 * i as u32),
            })
            .collect_vec();
        chain![
            generate_trace::<N_ROWS_SHORT_COMPONENT, CpuBackend>(log_sizes.0, &input_0),
            generate_trace::<N_ROWS_LONG_COMPONENT, CpuBackend>(log_sizes.1, &input_1)
        ]
        .collect_vec()
    }

    #[test_log::test]
    fn test_mixed_wide_fib_prove_with_blake() {
        const LOG_SIZE_SHORT: u32 = 3;
        const LOG_SIZE_LONG: u32 = 6;

        let config = PcsConfig::default();
        // Precompute twiddles.
        let twiddles = CpuBackend::precompute_twiddles(
            CanonicCoset::new(LOG_SIZE_LONG + 1 + config.fri_config.log_blowup_factor)
                .circle_domain()
                .half_coset,
        );

        // Setup protocol.
        let prover_channel = &mut Blake2sM31Channel::default();
        let mut commitment_scheme =
            CommitmentSchemeProver::<CpuBackend, Blake2sM31MerkleChannel>::new(config, &twiddles);

        // Preprocessed trace
        let mut tree_builder = commitment_scheme.tree_builder();
        tree_builder.extend_evals([]);
        tree_builder.commit(prover_channel);

        // Trace.
        let trace = generate_test_trace_mixed((LOG_SIZE_SHORT, LOG_SIZE_LONG));

        let mut tree_builder = commitment_scheme.tree_builder();
        tree_builder.extend_evals(trace);
        tree_builder.commit(prover_channel);

        // Generate components.
        let mut trace_alloc = TraceLocationAllocator::default();
        let component0 = WideFibonacciComponent::new(
            &mut trace_alloc,
            WideFibonacciEval::<N_ROWS_SHORT_COMPONENT> {
                log_n_rows: LOG_SIZE_SHORT,
            },
            SecureField::zero(),
        );
        let component1 = WideFibonacciComponent::new(
            &mut trace_alloc,
            WideFibonacciEval::<N_ROWS_LONG_COMPONENT> {
                log_n_rows: LOG_SIZE_LONG,
            },
            SecureField::zero(),
        );

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

        // Retrieve the expected column sizes in each commitment interaction, from the AIR.
        let sizes = TreeVec::new(vec![
            vec![],
            vec![LOG_SIZE_LONG; N_ROWS_SHORT_COMPONENT + N_ROWS_LONG_COMPONENT],
        ]);
        commitment_scheme.commit(proof.commitments[0], &sizes[0], verifier_channel);
        commitment_scheme.commit(proof.commitments[1], &sizes[1], verifier_channel);
        verify(
            &[&component0, &component1],
            verifier_channel,
            commitment_scheme,
            proof,
        )
        .unwrap();
    }
}
