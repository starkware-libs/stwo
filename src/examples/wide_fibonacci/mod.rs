pub mod avx;
pub mod constraint_eval;
pub mod structs;
pub mod trace_asserts;
pub mod trace_gen;

#[cfg(test)]
mod tests {
    use itertools::Itertools;
    use num_traits::{One, Zero};

    use super::structs::{Input, WideFibComponent};
    use super::trace_asserts::assert_constraints_on_row;
    use super::trace_gen::write_trace_row;
    use crate::core::air::accumulation::DomainEvaluationAccumulator;
    use crate::core::air::{Component, ComponentTrace};
    use crate::core::backend::cpu::CPUCircleEvaluation;
    use crate::core::backend::CPUBackend;
    use crate::core::fields::m31::BaseField;
    use crate::core::fields::qm31::QM31;
    use crate::core::poly::circle::CanonicCoset;

    fn fill_trace(private_input: &[Input]) -> Vec<Vec<BaseField>> {
        let zero_vec = vec![BaseField::zero(); private_input.len()];
        let mut dst = vec![zero_vec; 64];
        for (offset, input) in private_input.iter().enumerate() {
            write_trace_row(&mut dst, input, offset);
        }
        dst
    }

    #[test]
    fn test_wide_fib_trace() {
        let input = Input {
            a: BaseField::from_u32_unchecked(0x76),
            b: BaseField::from_u32_unchecked(0x483),
        };

        let trace = fill_trace(&[input]);
        let flat_trace = trace.into_iter().flatten().collect_vec();
        assert_constraints_on_row(&flat_trace);
    }

    #[test]
    fn test_wide_fib_constraints() {
        let wide_fib = WideFibComponent { log_size: 7 };
        let mut acc = DomainEvaluationAccumulator::new(
            QM31::from_u32_unchecked(1, 2, 3, 4),
            wide_fib.log_size + 1,
        );
        let inputs = (0..1 << wide_fib.log_size)
            .map(|i| Input {
                a: BaseField::one(),
                b: BaseField::from_u32_unchecked(i as u32),
            })
            .collect_vec();

        let trace = fill_trace(&inputs);

        let trace_domain = CanonicCoset::new(wide_fib.log_size);
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
