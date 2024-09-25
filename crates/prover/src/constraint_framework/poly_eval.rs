use num_traits::One;

use super::rational::Rational;
use super::EvalAtRow;
use crate::core::fields::m31::{BaseField, M31};
use crate::core::fields::qm31::SecureField;
use crate::core::fields::secure_column::SECURE_EXTENSION_DEGREE;

pub struct PolyEvaluator {
    pub cur_var_index: usize,
    pub constraints: Vec<Rational<SecureField>>,
}

impl EvalAtRow for PolyEvaluator {
    type F = Rational<BaseField>;
    type EF = Rational<SecureField>;
    fn next_interaction_mask<const N: usize>(
        &mut self,
        _interaction: usize,
        offsets: [isize; N],
    ) -> [Self::F; N] {
        // TODO(alont) support non-zero offsets.
        assert_eq!(offsets, [0; N]);
        self.cur_var_index += 1;
        std::array::from_fn(|_| Self::F::from_var_index(self.cur_var_index - 1))
    }
    fn add_constraint<G>(&mut self, constraint: G)
    where
        Self::EF: std::ops::Mul<G, Output = Self::EF>,
    {
        self.constraints.push(Self::EF::one() * constraint)
    }
    fn combine_ef(values: [Self::F; SECURE_EXTENSION_DEGREE]) -> Self::EF {
        let [one, i, u, iu] = [
            SecureField::one(),
            SecureField::from_m31_array([M31(0), M31(1), M31(0), M31(0)]),
            SecureField::from_m31_array([M31(0), M31(0), M31(1), M31(0)]),
            SecureField::from_m31_array([M31(0), M31(0), M31(0), M31(1)]),
        ];
        values[0].clone() * one
            + values[1].clone() * i
            + values[2].clone() * u
            + values[3].clone() * iu
    }
}

#[cfg(test)]
mod tests {

    use super::PolyEvaluator;
    use crate::constraint_framework::{EvalAtRow, FrameworkEval};
    use crate::core::fields::FieldExpOps;
    struct TestStruct {}
    impl FrameworkEval for TestStruct {
        fn log_size(&self) -> u32 {
            1 << 16
        }

        fn max_constraint_log_degree_bound(&self) -> u32 {
            1 << 17
        }

        fn evaluate<E: EvalAtRow>(&self, mut eval: E) -> E {
            let x0 = eval.next_trace_mask();
            let x1 = eval.next_trace_mask();
            let x2 = eval.next_trace_mask();

            eval.add_constraint(x0.clone() * x1.clone() * x2 * (x0 + x1).inverse());
            eval
        }
    }

    #[test]
    fn test_poly_eval() {
        let test_struct = TestStruct {};
        let eval = test_struct.evaluate(PolyEvaluator {
            cur_var_index: 0,
            constraints: vec![],
        });

        assert_eq!(eval.constraints[0].to_string(), "(x₀x₁x₂) / (x₀ + x₁)");
    }
}
