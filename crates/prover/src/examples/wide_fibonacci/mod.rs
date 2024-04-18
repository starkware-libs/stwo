#[cfg(target_arch = "x86_64")]
pub mod avx;
pub mod component;
pub mod constraint_eval;
pub mod trace_gen;

#[cfg(test)]
mod tests {
    use itertools::Itertools;
    use num_traits::{One, Zero};

    use super::component::{gen_trace, Input, WideFibAir, WideFibComponent, LOG_N_COLUMNS};
    use crate::commitment_scheme::blake2_hash::Blake2sHasher;
    use crate::commitment_scheme::hasher::Hasher;
    use crate::core::air::accumulation::DomainEvaluationAccumulator;
    use crate::core::air::{Component, ComponentTrace};
    use crate::core::backend::cpu::CPUCircleEvaluation;
    use crate::core::backend::CPUBackend;
    use crate::core::channel::{Blake2sChannel, Channel};
    use crate::core::fields::m31::BaseField;
    use crate::core::fields::qm31::QM31;
    use crate::core::fields::IntoSlice;
    use crate::core::poly::circle::CanonicCoset;
    use crate::core::poly::BitReversedOrder;
    use crate::core::prover::{prove, verify};
    use crate::examples::wide_fibonacci::trace_gen::write_lookup_column;
    use crate::m31;

    pub fn assert_constraints_on_row(row: &[BaseField]) {
        for i in 2..row.len() {
            assert_eq!(
                (row[i] - (row[i - 1] * row[i - 1] + row[i - 2] * row[i - 2])),
                BaseField::zero()
            );
        }
    }

    pub fn assert_constraints_on_lookup_column(
        column: &[BaseField],
        input_trace: &[Vec<BaseField>],
        alpha: BaseField,
        z: BaseField,
    ) {
        let n_columns = input_trace.len();
        let column_length = column.len();
        let mut prev_value = BaseField::one();
        for (i, cell) in column.iter().enumerate() {
            assert_eq!(
                (*cell
                    - ((input_trace[0][i] + alpha * input_trace[1][i] - z)
                        / (input_trace[n_columns - 2][i] + alpha * input_trace[n_columns - 1][i]
                            - z))
                        * prev_value),
                BaseField::zero()
            );
            prev_value = *cell;
        }

        assert_eq!(
            column[column_length - 1]
                * ((input_trace[n_columns - 2][column_length - 1]
                    + alpha * input_trace[n_columns - 1][column_length - 1]
                    - z)
                    / (input_trace[0][0] + alpha * input_trace[1][0] - z)),
            BaseField::one()
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

        let alpha = m31!(7);
        let z = m31!(11);
        let trace = gen_trace(&wide_fib, vec![input]);
        let lookup_column = write_lookup_column(&trace, alpha, z);
        dbg!(lookup_column.clone());

        assert_constraints_on_lookup_column(&lookup_column, &trace, alpha, z)
    }

    #[test]
    fn test_composition_is_low_degree() {
        let wide_fib = WideFibComponent {
            log_fibonacci_size: LOG_N_COLUMNS as u32,
            log_n_instances: 7,
        };
        let mut acc = DomainEvaluationAccumulator::new(
            QM31::from_u32_unchecked(1, 2, 3, 4),
            Component::<CPUBackend>::max_constraint_log_degree_bound(&wide_fib),
            Component::<CPUBackend>::n_constraints(&wide_fib),
        );
        let inputs = (0..1 << wide_fib.log_n_instances)
            .map(|i| Input {
                a: m31!(1),
                b: m31!(i as u32),
            })
            .collect_vec();

        let trace = gen_trace(&wide_fib, inputs);

        let trace_domain = CanonicCoset::new(wide_fib.log_column_size());
        let trace = trace
            .into_iter()
            .map(|col| CPUCircleEvaluation::new_canonical_ordered(trace_domain, col))
            .collect_vec();
        let trace_polys = trace
            .into_iter()
            .map(|eval| eval.interpolate())
            .collect_vec();
        let eval_domain = CanonicCoset::new(wide_fib.log_column_size() + 1).circle_domain();
        let trace_evals = trace_polys
            .iter()
            .map(|poly| poly.evaluate(eval_domain))
            .collect_vec();

        let trace = ComponentTrace {
            polys: trace_polys.iter().collect(),
            evals: trace_evals.iter().collect(),
        };

        wide_fib.evaluate_constraint_quotients_on_domain(&trace, &mut acc);

        let res = acc.finalize();
        let poly = res.0[0].clone();
        for coeff in poly.coeffs
            [(1 << (Component::<CPUBackend>::max_constraint_log_degree_bound(&wide_fib) - 1)) + 1..]
            .iter()
        {
            assert_eq!(*coeff, BaseField::zero());
        }
    }

    #[test_log::test]
    fn test_prove() {
        // Note: To see time measurement, run test with
        //   RUST_LOG_SPAN_EVENTS=enter,close RUST_LOG=info RUST_BACKTRACE=1 cargo test
        //   test_prove -- --nocapture

        const LOG_N_INSTANCES: u32 = 3;
        let component = WideFibComponent {
            log_fibonacci_size: LOG_N_COLUMNS as u32,
            log_n_instances: LOG_N_INSTANCES,
        };
        let private_input = (0..(1 << LOG_N_INSTANCES))
            .map(|i| Input {
                a: m31!(1),
                b: m31!(i),
            })
            .collect();
        let trace = gen_trace(&component, private_input);

        let trace_domain = CanonicCoset::new(component.log_column_size()).circle_domain();
        let trace = trace
            .into_iter()
            .map(|eval| CPUCircleEvaluation::<_, BitReversedOrder>::new(trace_domain, eval))
            .collect_vec();
        let air = WideFibAir { component };
        let prover_channel =
            &mut Blake2sChannel::new(Blake2sHasher::hash(BaseField::into_slice(&[])));
        let proof = prove::<CPUBackend>(&air, prover_channel, trace).unwrap();

        let verifier_channel =
            &mut Blake2sChannel::new(Blake2sHasher::hash(BaseField::into_slice(&[])));
        verify(proof, &air, verifier_channel).unwrap();
    }
}
