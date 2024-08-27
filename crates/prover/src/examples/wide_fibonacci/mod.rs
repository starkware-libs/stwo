use itertools::Itertools;

use crate::constraint_framework::{EvalAtRow, FrameworkComponent};
use crate::core::backend::simd::m31::{PackedBaseField, LOG_N_LANES};
use crate::core::backend::simd::SimdBackend;
use crate::core::backend::{Col, Column};
use crate::core::fields::m31::BaseField;
use crate::core::fields::FieldExpOps;
use crate::core::poly::circle::{CanonicCoset, CircleEvaluation};
use crate::core::poly::BitReversedOrder;
use crate::core::ColumnVec;

const N_COLUMNS: usize = 100;

pub struct FibInput {
    a: PackedBaseField,
    b: PackedBaseField,
}

#[derive(Clone)]
pub struct WideFibonacciComponent {
    pub log_n_rows: u32,
}
impl FrameworkComponent for WideFibonacciComponent {
    fn log_size(&self) -> u32 {
        self.log_n_rows
    }
    fn max_constraint_log_degree_bound(&self) -> u32 {
        self.log_n_rows + 1
    }
    fn evaluate<E: EvalAtRow>(&self, eval: E) -> E {
        let poseidon_eval = WideFibonacciEval { eval };
        poseidon_eval.eval()
    }
}

struct WideFibonacciEval<E: EvalAtRow> {
    eval: E,
}

impl<E: EvalAtRow> WideFibonacciEval<E> {
    fn eval(mut self) -> E {
        let mut a = self.eval.next_trace_mask();
        let mut b = self.eval.next_trace_mask();
        for _ in 2..N_COLUMNS {
            let c = self.eval.next_trace_mask();
            self.eval.add_constraint(c - (a.square() + b.square()));
            a = b;
            b = c;
        }
        self.eval
    }
}

pub fn generate_trace(
    log_size: u32,
    inputs: &[FibInput],
) -> ColumnVec<CircleEvaluation<SimdBackend, BaseField, BitReversedOrder>> {
    assert!(log_size >= LOG_N_LANES);
    assert_eq!(inputs.len(), 1 << (log_size - LOG_N_LANES));
    let mut trace = (0..N_COLUMNS)
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
    use num_traits::One;

    use crate::constraint_framework::{assert_constraints, FrameworkComponent};
    use crate::core::air::Component;
    use crate::core::backend::simd::m31::{PackedBaseField, LOG_N_LANES};
    use crate::core::backend::simd::SimdBackend;
    use crate::core::channel::Poseidon252Channel;
    use crate::core::fields::m31::BaseField;
    use crate::core::pcs::{CommitmentSchemeProver, CommitmentSchemeVerifier, PcsConfig, TreeVec};
    use crate::core::poly::circle::{CanonicCoset, PolyOps};
    use crate::core::prover::{prove, verify};
    use crate::core::vcs::poseidon252_merkle::Poseidon252MerkleChannel;
    use crate::core::InteractionElements;
    use crate::examples::wide_fibonacci::{
        generate_trace, FibInput, WideFibonacciComponent, WideFibonacciEval,
    };

    #[test]
    fn test_wide_fibonacci_constraints() {
        const LOG_N_ROWS: u32 = 8;
        let inputs = (0..(1 << (LOG_N_ROWS - LOG_N_LANES)))
            .map(|i| FibInput {
                a: PackedBaseField::one(),
                b: PackedBaseField::from_array(std::array::from_fn(|j| {
                    BaseField::from_u32_unchecked((i * 16 + j) as u32)
                })),
            })
            .collect_vec();

        // Trace.
        let trace = generate_trace(LOG_N_ROWS, &inputs);

        let traces = TreeVec::new(vec![trace]);
        let trace_polys =
            traces.map(|trace| trace.into_iter().map(|c| c.interpolate()).collect_vec());
        assert_constraints(&trace_polys, CanonicCoset::new(LOG_N_ROWS), |eval| {
            WideFibonacciEval { eval }.eval();
        });
    }

    #[test_log::test]
    fn test_single_instance_wide_fib_prove_with_poseidon() {
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
        let commitment_scheme =
            &mut CommitmentSchemeProver::<SimdBackend, Poseidon252MerkleChannel>::new(
                config, &twiddles,
            );
        let component = WideFibonacciComponent {
            log_n_rows: LOG_N_INSTANCES,
        };

        // Trace.
        let inputs = (0..(1 << (LOG_N_INSTANCES - LOG_N_LANES)))
            .map(|i| FibInput {
                a: PackedBaseField::one(),
                b: PackedBaseField::from_array(std::array::from_fn(|j| {
                    BaseField::from_u32_unchecked((i * 16 + j) as u32)
                })),
            })
            .collect_vec();
        let trace = generate_trace(component.log_size(), &inputs);
        let mut tree_builder = commitment_scheme.tree_builder();
        tree_builder.extend_evals(trace);
        tree_builder.commit(prover_channel);

        // Prove constraints.
        let component = WideFibonacciComponent {
            log_n_rows: LOG_N_INSTANCES,
        };
        let proof = prove::<SimdBackend, Poseidon252MerkleChannel>(
            &[&component],
            prover_channel,
            &InteractionElements::default(),
            commitment_scheme,
        )
        .unwrap();

        // Verify.
        let verifier_channel = &mut Poseidon252Channel::default();
        let commitment_scheme =
            &mut CommitmentSchemeVerifier::<Poseidon252MerkleChannel>::new(config);

        // Retrieve the expected column sizes in each commitment interaction, from the AIR.
        let sizes = component.trace_log_degree_bounds();
        commitment_scheme.commit(proof.commitments[0], &sizes[0], verifier_channel);
        verify(
            &[&component],
            verifier_channel,
            &InteractionElements::default(),
            commitment_scheme,
            proof,
        )
        .unwrap();
    }
}
