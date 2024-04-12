use std::iter::zip;

use num_traits::{One, Zero};

use crate::core::air::evaluation::SecureColumn;
use crate::core::backend::CPUBackend;
use crate::core::fields::m31::BaseField;
use crate::core::fields::qm31::SecureField;
use crate::core::fields::Field;
use crate::core::lookups::mle::{Mle, MleOps};
use crate::core::lookups::sumcheck::SumcheckOracle;
use crate::core::lookups::utils::Polynomial;

/// Evaluates the multi-linear extension `mle` at point `p`.
pub(crate) fn eval_mle_at_point<F: Field>(mle: &[F], eval_range: , p: &[F]) -> F {
    match p {
        [] => mle[0],
        [p_i, p @ ..] => {
            let (lhs, rhs) = mle.split_at(mle.len() / 2);
            let lhs_eval = eval_mle_at_point(lhs, p);
            let rhs_eval = eval_mle_at_point(rhs, p);
            // `= eq(0, p_i) * lhs + eq(1, p_i) * rhs`
            *p_i * (rhs_eval - lhs_eval) + lhs_eval
        }
    }
}

// fn eval_mle_at_point<F: Field>(mle: &Mle<CPUBackend, F>, eval_range: Range<usize>, p: &[F]) -> F {
//     match p {
//         [] => mle.at(eval_range.start),
//         [p_i, p @ ..] => {
//             let midpoint = (eval_range.start + eval_range.end) / 2;
//             let lhs_eval = eval_mle_at_point(mle, eval_range.start..midpoint, p);
//             let rhs_eval = eval_mle_at_point(mle, midpoint..eval_range.end, p);
//             // Equivalent to `eq(0, p_i) * lhs + eq(1, p_i) * rhs`.
//             *p_i * (rhs_eval - lhs_eval) + lhs_eval
//         }
//     }
// }

impl MleOps<BaseField> for CPUBackend {
    fn eval_at_point(mle: &Mle<Self, BaseField>, point: &[BaseField]) -> BaseField {
        assert_eq!(point.len(), mle.num_variables());
        eval_mle_at_point(mle, point)
    }

    fn fix_first(mle: Mle<Self, BaseField>, assignment: SecureField) -> Mle<Self, SecureField> {
        let (lhs_evals, rhs_evals) = mle.split_at(mle.len() / 2);

        // Column of SecureField elements (Structure of arrays)
        // TODO: Make this better.
        let mut fixed_evals = SecureColumn::<Self> {
            cols: Default::default(),
        };

        for (&lhs_eval, &rhs_eval) in zip(lhs_evals, rhs_evals) {
            // `eval = eq(0, assignment) * lhs + eq(1, assignment) * rhs`
            let eval = assignment * (rhs_eval - lhs_eval) + lhs_eval;

            let [e0, e1, e2, e3] = eval.to_m31_array();
            fixed_evals.cols[0].push(e0);
            fixed_evals.cols[1].push(e1);
            fixed_evals.cols[2].push(e2);
            fixed_evals.cols[3].push(e3);
        }

        Mle::new(fixed_evals)
    }
}

impl MleOps<SecureField> for CPUBackend {
    fn eval_at_point(mle: &Mle<Self, SecureField>, point: &[SecureField]) -> SecureField {
        /// Evaluates the multi-linear extension `mle` at point `p`.
        ///
        /// `secure_mle` is evaluations of a [`SecureField`] multilinear polynomial stored as
        /// structure-of-arrays.
        fn eval_mle_at_point(secure_mle: [&[BaseField]; 4], p: &[SecureField]) -> SecureField {
            match p {
                [] => SecureField::from_m31_array(secure_mle.map(|mle| mle[0])),
                [p_i, p @ ..] => {
                    let mle_midpoint = secure_mle[0].len() / 2;
                    let (lhs0, rhs0) = secure_mle[0].split_at(mle_midpoint);
                    let (lhs1, rhs1) = secure_mle[1].split_at(mle_midpoint);
                    let (lhs2, rhs2) = secure_mle[2].split_at(mle_midpoint);
                    let (lhs3, rhs3) = secure_mle[3].split_at(mle_midpoint);
                    let lhs_eval = eval_mle_at_point([lhs0, lhs1, lhs2, lhs3], p);
                    let rhs_eval = eval_mle_at_point([rhs0, rhs1, rhs2, rhs3], p);
                    // Equivalent to `eq(0, p_i) * lhs + eq(1, p_i) * rhs`.
                    *p_i * (rhs_eval - lhs_eval) + lhs_eval
                }
            }
        }

        assert_eq!(point.len(), mle.num_variables());
        let secure_mle = [&*mle.cols[0], &*mle.cols[1], &*mle.cols[2], &*mle.cols[3]];
        eval_mle_at_point(secure_mle, point)
    }

    fn fix_first(mle: Mle<Self, SecureField>, assignment: SecureField) -> Mle<Self, SecureField> {
        let n_fixed_evals = mle.len() / 2;
        let mut evals = mle.into_evals();

        for i in 0..n_fixed_evals {
            let lhs = evals.at(i);
            let rhs = evals.at(i + n_fixed_evals);
            // `evals[i] = eq(0, assignment) * lhs + eq(1, assignment) * rhs`
            evals.set(i, lhs + assignment * (rhs - lhs));
        }

        evals
            .cols
            .iter_mut()
            .for_each(|col| col.truncate(n_fixed_evals));

        Mle::new(evals)
    }
}

impl SumcheckOracle for Mle<CPUBackend, SecureField> {
    fn num_variables(&self) -> usize {
        self.num_variables()
    }

    fn univariate_sum(&self, claim: SecureField) -> Polynomial<SecureField> {
        let x0 = SecureField::zero();
        let x1 = SecureField::one();

        let evals_midpoint = self.len() / 2;

        let c0 = self.cols[0][0..evals_midpoint].iter().sum();
        let c1 = self.cols[1][0..evals_midpoint].iter().sum();
        let c2 = self.cols[2][0..evals_midpoint].iter().sum();
        let c3 = self.cols[3][0..evals_midpoint].iter().sum();

        let y0 = SecureField::from_m31_array([c0, c1, c2, c3]);
        let y1 = claim - y0;

        Polynomial::interpolate_lagrange(&[x0, x1], &[y0, y1])
    }

    fn fix_first(self, challenge: SecureField) -> Self {
        self.fix_first(challenge)
    }
}
