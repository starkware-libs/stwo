use std::ops::{Deref, DerefMut};

use num_traits::{One, Zero};
use prover_research::core::fields::qm31::SecureField;
use prover_research::core::fields::Field;

use crate::sumcheck::SumcheckOracle;
use crate::utils::Polynomial;
/// Multi-Linear Extension.
// TODO: "Values are assumed to be in lagrange basis" - Is that correct wording?
#[derive(Debug, Clone)]
pub struct MultiLinearExtension<F> {
    num_variables: usize,
    evals: Vec<F>,
}

impl<F: Field> MultiLinearExtension<F> {
    pub fn new(evals: Vec<F>) -> Self {
        assert!(evals.len().is_power_of_two());
        let num_variables = evals.len().ilog2() as usize;
        Self {
            num_variables,
            evals,
        }
    }

    pub fn eval(&self, point: &[F]) -> F {
        fn eval_impl<F: Field>(evals: &[F], point: &[F]) -> F {
            if evals.len() == 1 {
                return evals[0];
            }

            let (lhs_evals, rhs_evals) = evals.split_at(evals.len() / 2);
            let lhs_eval = eval_impl(lhs_evals, &point[1..]);
            let rhs_eval = eval_impl(rhs_evals, &point[1..]);
            let coordinate = point[0];
            // Evaluate the factor of the lagrange polynomial corresponding to this variable.
            lhs_eval * (F::one() - coordinate) + rhs_eval * coordinate
        }

        assert_eq!(point.len(), self.num_variables);
        eval_impl(&self.evals, point)
    }

    pub fn into_evals(self) -> Vec<F> {
        self.evals
    }

    pub fn fix_first(mut self, assignment: F) -> Self {
        let n_fixed_evals = self.evals.len() / 2;

        for i in 0..n_fixed_evals {
            let lhs = self.evals[i];
            let rhs = self.evals[i + n_fixed_evals];
            self.evals[i] += assignment * (rhs - lhs);
        }

        self.evals.truncate(n_fixed_evals);
        self.num_variables -= 1;

        self
    }
    // pub fn fix_first(self) -> Self {}
}

impl<F> Deref for MultiLinearExtension<F> {
    type Target = [F];

    fn deref(&self) -> &Self::Target {
        &self.evals
    }
}

impl<F> DerefMut for MultiLinearExtension<F> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.evals
    }
}

impl SumcheckOracle for MultiLinearExtension<SecureField> {
    fn num_variables(&self) -> u32 {
        self.num_variables as u32
    }

    fn univariate_sum(&self) -> Polynomial<SecureField> {
        let x0 = SecureField::zero();
        let x1 = SecureField::one();

        let (lhs_evals, rhs_evals) = self.evals.split_at(self.evals.len() / 2);

        let y0 = lhs_evals.iter().copied().sum();
        let y1 = rhs_evals.iter().copied().sum();

        Polynomial::interpolate_lagrange(&[x0, x1], &[y0, y1])
    }

    fn fix_first(self, challenge: SecureField) -> Self {
        self.fix_first(challenge)
    }
}
