#[cfg(target_arch = "x86_64")]
pub mod avx;
pub mod constraint_eval;
pub mod structs;
pub mod trace_asserts;
pub mod trace_gen;

#[cfg(test)]
mod tests {
    use itertools::Itertools;
    use num_traits::Zero;

    use super::structs::{Input, WideFibComponent, LOG_N_COLUMNS, LOG_N_ROWS};
    use super::trace_asserts::{assert_constraints_on_lookup_column, assert_constraints_on_row};
    use crate::core::air::accumulation::DomainEvaluationAccumulator;
    use crate::core::air::{Component, ComponentTrace};
    use crate::core::backend::cpu::CPUCircleEvaluation;
    use crate::core::backend::CPUBackend;
    use crate::core::fields::m31::BaseField;
    use crate::core::fields::qm31::QM31;
    use crate::core::poly::circle::CanonicCoset;
    use crate::m31;

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

        let trace = wide_fib.fill_initial_trace(vec![input, input]);
        let trace0 = trace.iter().map(|col| col[0]).collect_vec();
        let trace1 = trace.iter().map(|col| col[1]).collect_vec();

        assert_constraints_on_row(&trace0);
        assert_constraints_on_row(&trace1);
    }

    #[test]
    fn test_lookup_column_constraints() {
        let wide_fib = WideFibComponent {
            log_fibonacci_size: (LOG_N_ROWS + LOG_N_COLUMNS) as u32,
            log_n_instances: 0,
        };
        let input = Input {
            a: m31!(1),
            b: m31!(1),
        };

        let alpha = m31!(1);
        let z = m31!(2);
        let trace = wide_fib.fill_initial_trace(vec![input]);
        let lookup_trace = wide_fib.lookup_columns(&trace, alpha, z);

        assert_constraints_on_lookup_column(&lookup_trace, &trace, alpha, z)
    }

    #[test]
    fn test_constraints_polynomial_is_low_degree() {
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

        let trace = wide_fib.fill_initial_trace(inputs);

        let trace_domain = CanonicCoset::new(wide_fib.log_column_size());
        let trace = trace
            .into_iter()
            .map(|col| CPUCircleEvaluation::new_canonical_ordered(trace_domain, col))
            .collect_vec();
        let trace_polys = trace
            .into_iter()
            .map(|eval| eval.interpolate())
            .collect_vec();

        let trace = ComponentTrace {
            columns: trace_polys.iter().collect(),
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
}
