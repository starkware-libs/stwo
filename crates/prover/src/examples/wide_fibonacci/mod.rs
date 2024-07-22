pub mod component;
pub mod constraint_eval;
pub mod simd;
pub mod trace_gen;

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;

    use itertools::Itertools;
    use num_traits::{One, Zero};
    #[cfg(not(target_arch = "wasm32"))]
    use starknet_ff::FieldElement as FieldElement252;

    use super::component::{Input, WideFibAir, WideFibComponent, LOG_N_COLUMNS};
    use super::constraint_eval::gen_trace;
    use crate::core::air::accumulation::DomainEvaluationAccumulator;
    use crate::core::air::{Component, ComponentProver, ComponentTrace};
    use crate::core::backend::cpu::CpuCircleEvaluation;
    #[cfg(not(target_arch = "wasm32"))]
    use crate::core::channel::Poseidon252Channel;
    use crate::core::channel::{Blake2sChannel, Channel};
    use crate::core::fields::m31::BaseField;
    use crate::core::fields::qm31::SecureField;
    use crate::core::fields::IntoSlice;
    use crate::core::pcs::TreeVec;
    use crate::core::poly::circle::CanonicCoset;
    use crate::core::prover::StarkProof;
    use crate::core::utils::{
        bit_reverse, circle_domain_order_to_coset_order, shifted_secure_combination,
    };
    use crate::core::vcs::blake2_hash::Blake2sHasher;
    use crate::core::vcs::blake2_merkle::Blake2sMerkleHasher;
    #[cfg(not(target_arch = "wasm32"))]
    use crate::core::vcs::poseidon252_merkle::Poseidon252MerkleHasher;
    use crate::core::InteractionElements;
    use crate::examples::wide_fibonacci::trace_gen::write_lookup_column;
    use crate::trace_generation::{commit_and_prove, commit_and_verify, ComponentTraceGenerator};
    use crate::{m31, qm31};

    pub fn assert_constraints_on_row(row: &[BaseField]) {
        for i in 2..row.len() {
            assert_eq!(
                (row[i] - (row[i - 1] * row[i - 1] + row[i - 2] * row[i - 2])),
                BaseField::zero()
            );
        }
    }

    pub fn assert_constraints_on_lookup_column(
        column: &[SecureField],
        input_trace: &[Vec<BaseField>],
        alpha: SecureField,
        z: SecureField,
    ) {
        let n_columns = input_trace.len();
        let column_length = column.len();
        assert_eq!(column_length, input_trace[0].len());
        let mut prev_value = SecureField::one();
        for (i, cell) in column.iter().enumerate() {
            assert_eq!(
                *cell
                    * shifted_secure_combination(
                        &[input_trace[n_columns - 2][i], input_trace[n_columns - 1][i]],
                        alpha,
                        z,
                    ),
                shifted_secure_combination(&[input_trace[0][i], input_trace[1][i]], alpha, z)
                    * prev_value
            );
            prev_value = *cell;
        }

        // Assert the last cell in the column is equal to the combination of the first two values
        // divided by the combination of the last two values in the sequence (all other values
        // should cancel out).
        assert_eq!(
            column[column_length - 1]
                * shifted_secure_combination(
                    &[input_trace[n_columns - 2][1], input_trace[n_columns - 1][1]],
                    alpha,
                    z,
                ),
            (shifted_secure_combination(&[input_trace[0][0], input_trace[1][0]], alpha, z))
        );
    }

    #[test]
    fn test_trace_row_constraints() {
        let wide_fib = WideFibComponent {
            log_fibonacci_size: LOG_N_COLUMNS as u32,
            log_n_instances: 1,
        };
        let input = Input {
            a: m31!(0x76),
            b: m31!(0x483),
        };

        let trace = gen_trace(&wide_fib, vec![input, input]);
        let row_0 = trace.iter().map(|col| col[0]).collect_vec();
        let row_1 = trace.iter().map(|col| col[1]).collect_vec();

        assert_constraints_on_row(&row_0);
        assert_constraints_on_row(&row_1);
    }

    #[test]
    fn test_lookup_column_constraints() {
        let wide_fib = WideFibComponent {
            log_fibonacci_size: 4 + LOG_N_COLUMNS as u32,
            log_n_instances: 0,
        };
        let input = Input {
            a: m31!(1),
            b: m31!(1),
        };

        let alpha = qm31!(7, 1, 3, 4);
        let z = qm31!(11, 1, 2, 3);
        let mut trace = gen_trace(&wide_fib, vec![input]);
        let input_trace = trace.iter().map(|values| &values[..]).collect_vec();
        let lookup_column = write_lookup_column(&input_trace, alpha, z);

        trace = trace
            .iter_mut()
            .map(|column| {
                bit_reverse(column);
                circle_domain_order_to_coset_order(column)
            })
            .collect_vec();
        assert_constraints_on_lookup_column(&lookup_column, &trace, alpha, z)
    }

    #[test]
    fn test_composition_is_low_degree() {
        let wide_fib = WideFibComponent {
            log_fibonacci_size: 3 + LOG_N_COLUMNS as u32,
            log_n_instances: 0,
        };
        let random_coeff = qm31!(1, 2, 3, 4);
        let mut acc = DomainEvaluationAccumulator::new(
            random_coeff,
            wide_fib.max_constraint_log_degree_bound(),
            wide_fib.n_constraints(),
        );
        let inputs = (0..1 << wide_fib.log_n_instances)
            .map(|i| Input {
                a: m31!(1),
                b: m31!(i + 1_u32),
            })
            .collect_vec();

        let trace_values = gen_trace(&wide_fib, inputs);

        let trace_domain = CanonicCoset::new(wide_fib.log_column_size());
        let trace = trace_values
            .into_iter()
            .map(|eval| CpuCircleEvaluation::new_canonical_ordered(trace_domain, eval))
            .collect_vec();
        let trace_polys = trace
            .clone()
            .into_iter()
            .map(|eval| eval.interpolate())
            .collect_vec();
        let eval_domain =
            CanonicCoset::new(wide_fib.max_constraint_log_degree_bound()).circle_domain();
        let trace_evals = trace_polys
            .iter()
            .map(|poly| poly.evaluate(eval_domain))
            .collect_vec();

        let interaction_elements = InteractionElements::new(BTreeMap::from_iter(
            wide_fib
                .interaction_element_ids()
                .iter()
                .cloned()
                .enumerate()
                .map(|(i, id)| (id, qm31!(43 + i as u32, 1, 2, 3))),
        ));
        let interaction_poly = wide_fib
            .write_interaction_trace(&trace.iter().collect(), &interaction_elements)
            .into_iter()
            .map(|eval| eval.interpolate())
            .collect_vec();

        let interaction_trace = interaction_poly
            .iter()
            .map(|poly| poly.evaluate(eval_domain))
            .collect_vec();
        let trace = ComponentTrace {
            polys: TreeVec::new(vec![
                trace_polys.iter().collect_vec(),
                interaction_poly.iter().collect_vec(),
            ]),
            evals: TreeVec::new(vec![
                trace_evals.iter().collect_vec(),
                interaction_trace.iter().collect_vec(),
            ]),
        };

        let lookup_values = wide_fib.lookup_values(&trace);
        wide_fib.evaluate_constraint_quotients_on_domain(
            &trace,
            &mut acc,
            &interaction_elements,
            &lookup_values,
        );

        let res = acc.finalize();
        let poly = res.0[0].clone();
        for coeff in poly.coeffs[(1 << wide_fib.max_constraint_log_degree_bound()) - 1..].iter() {
            assert_eq!(*coeff, BaseField::zero());
        }
    }

    #[test_log::test]
    fn test_single_instance_wide_fib_prove() {
        // Note: To see time measurement, run test with
        //   RUST_LOG_SPAN_EVENTS=enter,close RUST_LOG=info RUST_BACKTRACE=1 cargo test
        //   test_prove -- --nocapture

        const LOG_N_INSTANCES: u32 = 0;
        let component = WideFibComponent {
            log_fibonacci_size: 3 + LOG_N_COLUMNS as u32,
            log_n_instances: LOG_N_INSTANCES,
        };
        let private_input = (0..(1 << LOG_N_INSTANCES))
            .map(|i| Input {
                a: m31!(1),
                b: m31!(i),
            })
            .collect();
        let trace = gen_trace(&component, private_input);

        let trace_domain = CanonicCoset::new(component.log_column_size());
        let trace = trace
            .into_iter()
            .map(|eval| CpuCircleEvaluation::new_canonical_ordered(trace_domain, eval))
            .collect_vec();
        let air = WideFibAir { component };
        let prover_channel =
            &mut Blake2sChannel::new(Blake2sHasher::hash(BaseField::into_slice(&[])));
        let proof: StarkProof<Blake2sMerkleHasher> =
            commit_and_prove(&air, prover_channel, trace).unwrap();

        let verifier_channel =
            &mut Blake2sChannel::new(Blake2sHasher::hash(BaseField::into_slice(&[])));
        commit_and_verify(proof, &air, verifier_channel).unwrap();
    }

    #[cfg(not(target_arch = "wasm32"))]
    #[test_log::test]
    fn test_single_instance_wide_fib_prove_with_poseidon() {
        const LOG_N_INSTANCES: u32 = 0;
        let component = WideFibComponent {
            log_fibonacci_size: 3 + LOG_N_COLUMNS as u32,
            log_n_instances: LOG_N_INSTANCES,
        };
        let private_input = (0..(1 << LOG_N_INSTANCES))
            .map(|i| Input {
                a: m31!(1),
                b: m31!(i),
            })
            .collect();
        let trace = gen_trace(&component, private_input);

        let trace_domain = CanonicCoset::new(component.log_column_size());
        let trace = trace
            .into_iter()
            .map(|eval| CpuCircleEvaluation::new_canonical_ordered(trace_domain, eval))
            .collect_vec();
        let air = WideFibAir { component };
        let prover_channel = &mut Poseidon252Channel::new(FieldElement252::default());
        let proof: StarkProof<Poseidon252MerkleHasher> =
            commit_and_prove(&air, prover_channel, trace).unwrap();

        let verifier_channel = &mut Poseidon252Channel::new(FieldElement252::default());
        commit_and_verify(proof, &air, verifier_channel).unwrap();
    }
}
