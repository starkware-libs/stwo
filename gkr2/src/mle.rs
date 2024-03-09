use std::iter::zip;
use std::ops::{Deref, DerefMut};
use std::vec;

use num_traits::{One, Zero};
use prover_research::core::fields::qm31::SecureField;
use prover_research::core::fields::{ExtensionOf, Field};

use crate::gkr::{SUMCHECK_ADDS, SUMCHECK_MULTS};
use crate::sumcheck::SumcheckOracle;
use crate::utils::Polynomial;

/// Multi-Linear extension with values represented in the lagrange basis.
#[derive(Debug, Clone)]
pub struct Mle<F> {
    evals: Vec<F>,
}

impl<F: Field> Mle<F> {
    pub fn new(evals: Vec<F>) -> Self {
        assert!(evals.len().is_power_of_two());
        Self { evals }
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

        assert_eq!(point.len(), self.num_variables());
        eval_impl(&self.evals, point)
    }

    pub fn into_evals(self) -> Vec<F> {
        self.evals
    }

    pub fn fix_first(self, assignment: SecureField) -> Mle<SecureField>
    where
        SecureField: ExtensionOf<F>,
    {
        let (lhs_evals, rhs_evals) = self.evals.split_at(self.evals.len() / 2);
        let mut evals = Vec::new();

        for (&lhs_eval, &rhs_eval) in zip(lhs_evals, rhs_evals) {
            evals.push(assignment * (rhs_eval - lhs_eval) + lhs_eval);
        }

        Mle::new(evals)
    }

    pub fn num_variables(&self) -> usize {
        self.evals.len().ilog2() as usize
    }
}

impl Mle<SecureField> {
    pub fn fix_first_mut(&mut self, assignment: SecureField) {
        let n_fixed_evals = self.evals.len() / 2;

        unsafe { SUMCHECK_ADDS += n_fixed_evals * 2 };
        unsafe { SUMCHECK_MULTS += n_fixed_evals };
        for i in 0..n_fixed_evals {
            let lhs = self.evals[i];
            let rhs = self.evals[i + n_fixed_evals];
            self.evals[i] += assignment * (rhs - lhs);
        }

        self.evals.truncate(n_fixed_evals);
    }
}

impl<F> Deref for Mle<F> {
    type Target = [F];

    fn deref(&self) -> &Self::Target {
        &self.evals
    }
}

impl<F> DerefMut for Mle<F> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.evals
    }
}

impl<F> IntoIterator for Mle<F> {
    type Item = F;

    type IntoIter = vec::IntoIter<F>;

    fn into_iter(self) -> Self::IntoIter {
        self.evals.into_iter()
    }
}

impl SumcheckOracle for Mle<SecureField> {
    fn num_variables(&self) -> usize {
        self.num_variables()
    }

    fn univariate_sum(&self, claim: SecureField) -> Polynomial<SecureField> {
        let x0 = SecureField::zero();
        let x1 = SecureField::one();

        let (lhs_evals, _rhs_evals) = self.evals.split_at(self.evals.len() / 2);

        let y0 = lhs_evals.iter().copied().sum();
        let y1 = claim - y0;

        Polynomial::interpolate_lagrange(&[x0, x1], &[y0, y1])
    }

    fn fix_first(self, challenge: SecureField) -> Self {
        self.fix_first(challenge)
    }
}

/// Rectangular trace where columns are [`Mle`]s
#[derive(Debug, Clone)]
pub struct MleTrace<F> {
    columns: Vec<Mle<F>>,
}

impl<F: Field> MleTrace<F> {
    /// # Panics
    ///
    /// Panics if columns is empty or if the columns aren't all the same size.
    pub fn new(columns: Vec<Mle<F>>) -> Self {
        let num_variables = columns[0].num_variables();
        assert!(columns.iter().all(|c| c.num_variables() == num_variables));
        Self { columns }
    }

    /// Returns the number of variables that all columns have.
    pub fn num_variables(&self) -> usize {
        self.columns[0].num_variables()
    }

    /// Evaluates each column of the trace at `point`.
    pub fn eval(&self, point: &[F]) -> Vec<F> {
        self.columns.iter().map(|c| c.eval(point)).collect()
    }
}

impl<F> Deref for MleTrace<F> {
    type Target = [Mle<F>];

    fn deref(&self) -> &Self::Target {
        &self.columns
    }
}
