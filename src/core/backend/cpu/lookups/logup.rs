use num_traits::{One, Zero};

use crate::core::air::evaluation::SecureColumn;
use crate::core::backend::CPUBackend;
use crate::core::fields::m31::BaseField;
use crate::core::fields::qm31::SecureField;
use crate::core::fields::Field;
use crate::core::lookups::logup::{Fraction, LogupOps, LogupOracle, LogupTrace};
use crate::core::lookups::mle::Mle;
use crate::core::lookups::sumcheck::{SumcheckOracle, UnivariateEvals};

impl LogupOps for CPUBackend {
    fn next_layer(layer: &LogupTrace<Self>) -> LogupTrace<Self> {
        let mut next_numerators = SecureColumn::default();
        let mut next_denominators = SecureColumn::default();

        let half_layer_len = layer.len() / 2;
        let one = BaseField::one();

        match layer {
            LogupTrace::Singles { denominators } => {
                for i in 0..half_layer_len {
                    let d0 = denominators.at(i * 2);
                    let d1 = denominators.at(i * 2 + 1);

                    let res = Fraction::new(one, d0) + Fraction::new(one, d1);

                    next_numerators.push(res.numerator);
                    next_denominators.push(res.denominator);
                }
            }
            LogupTrace::Multiplicities {
                numerators,
                denominators,
            } => {
                for i in 0..half_layer_len {
                    let n0 = numerators[i * 2];
                    let d0 = denominators.at(i * 2);

                    let n1 = numerators[i * 2 + 1];
                    let d1 = denominators.at(i * 2 + 1);

                    let res = Fraction::new(n0, d0) + Fraction::new(n1, d1);

                    next_numerators.push(res.numerator);
                    next_denominators.push(res.denominator);
                }
            }
            LogupTrace::Generic {
                numerators,
                denominators,
            } => {
                for i in 0..half_layer_len {
                    let n0 = numerators.at(i * 2);
                    let d0 = denominators.at(i * 2);

                    let n1 = numerators.at(i * 2 + 1);
                    let d1 = denominators.at(i * 2 + 1);

                    let res = Fraction::new(n0, d0) + Fraction::new(n1, d1);

                    next_numerators.push(res.numerator);
                    next_denominators.push(res.denominator);
                }
            }
        }

        LogupTrace::Generic {
            numerators: Mle::new(next_numerators),
            denominators: Mle::new(next_denominators),
        }
    }

    fn univariate_sum_evals(oracle: &LogupOracle<'_, Self>) -> UnivariateEvals {
        let num_terms = 1 << (oracle.num_variables() - 1);
        let lambda = oracle.lambda();
        let eq_evals = oracle.eq_evals();
        let trace = oracle.trace();

        // Obtain the evaluations at `0` and `2`.
        let mut eval_at_0 = SecureField::zero();
        let mut eval_at_2 = SecureField::zero();

        for i in 0..num_terms {
            let (fraction0, fraction2) = match trace {
                LogupTrace::Singles { denominators } => {
                    let d0_lhs = denominators.at(i * 2);
                    let d1_lhs = denominators.at(i * 2 + 1);

                    let fraction0 = Fraction::new(d0_lhs + d1_lhs, d0_lhs * d1_lhs);

                    let d0_rhs = denominators.at((num_terms + i) * 2);
                    let d1_rhs = denominators.at((num_terms + i) * 2 + 1);

                    let fraction2 = {
                        let d0 = d0_rhs.double() - d0_lhs;
                        let d1 = d1_rhs.double() - d1_lhs;
                        Fraction::new(d0 + d1, d0 * d1)
                    };

                    (fraction0, fraction2)
                }
                LogupTrace::Multiplicities {
                    numerators,
                    denominators,
                } => {
                    let n0_lhs = numerators[i * 2];
                    let n1_lhs = numerators[i * 2 + 1];

                    let d0_lhs = denominators.at(i * 2);
                    let d1_lhs = denominators.at(i * 2 + 1);

                    let fraction0 = {
                        let a = Fraction::new(n0_lhs, d0_lhs);
                        let b = Fraction::new(n1_lhs, d1_lhs);
                        a + b
                    };

                    let n0_rhs = numerators[(num_terms + i) * 2];
                    let n1_rhs = numerators[(num_terms + i) * 2 + 1];

                    let d0_rhs = denominators.at((num_terms + i) * 2);
                    let d1_rhs = denominators.at((num_terms + i) * 2 + 1);

                    let fraction2 = {
                        let n0 = n0_rhs.double() - n0_lhs;
                        let n1 = n1_rhs.double() - n1_lhs;

                        let d0 = d0_rhs.double() - d0_lhs;
                        let d1 = d1_rhs.double() - d1_lhs;

                        let a = Fraction::new(n0, d0);
                        let b = Fraction::new(n1, d1);
                        a + b
                    };

                    (fraction0, fraction2)
                }
                LogupTrace::Generic {
                    numerators,
                    denominators,
                } => {
                    let n0_lhs = numerators.at(i * 2);
                    let n1_lhs = numerators.at(i * 2 + 1);

                    let d0_lhs = denominators.at(i * 2);
                    let d1_lhs = denominators.at(i * 2 + 1);

                    let fraction0 = {
                        let a = Fraction::new(n0_lhs, d0_lhs);
                        let b = Fraction::new(n1_lhs, d1_lhs);
                        a + b
                    };

                    let n0_rhs = numerators.at((num_terms + i) * 2);
                    let n1_rhs = numerators.at((num_terms + i) * 2 + 1);

                    let d0_rhs = denominators.at((num_terms + i) * 2);
                    let d1_rhs = denominators.at((num_terms + i) * 2 + 1);

                    let fraction2 = {
                        let n0 = n0_rhs.double() - n0_lhs;
                        let n1 = n1_rhs.double() - n1_lhs;

                        let d0 = d0_rhs.double() - d0_lhs;
                        let d1 = d1_rhs.double() - d1_lhs;

                        let a = Fraction::new(n0, d0);
                        let b = Fraction::new(n1, d1);
                        a + b
                    };

                    (fraction0, fraction2)
                }
            };

            let eq_eval = eq_evals[i];
            eval_at_0 += eq_eval * (fraction0.numerator + lambda * fraction0.denominator);
            eval_at_2 += eq_eval * (fraction2.numerator + lambda * fraction2.denominator);
        }

        UnivariateEvals {
            eval_at_0,
            eval_at_2,
        }
    }
}
