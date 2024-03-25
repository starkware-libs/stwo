pub mod constraint_eval;
pub mod structs;
pub mod trace_asserts;
pub mod trace_gen;

#[cfg(test)]
mod tests {
    use itertools::Itertools;
    use num_traits::Zero;

    use super::structs::Input;
    use super::trace_asserts::assert_constraints_on_row;
    use super::trace_gen::write_trace_row;
    use crate::core::fields::m31::BaseField;

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
}
