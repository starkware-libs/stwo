use num_traits::{One, Zero};

use crate::core::backend::CpuBackend;
use crate::core::fields::m31::BaseField;
use crate::core::fields::qm31::SecureField;
use crate::core::fields::Field;
use crate::core::lookups::gkr::{correct_sum_as_poly_in_first_variable, GkrBinaryLayer};
use crate::core::lookups::logup::{Fraction, LogupOps, LogupOracle, LogupTrace};
use crate::core::lookups::mle::Mle;
use crate::core::lookups::sumcheck::MultivariatePolyOracle;
use crate::core::lookups::utils::UnivariatePoly;

impl LogupOps for CpuBackend {
    fn next_layer(layer: &LogupTrace<Self>) -> LogupTrace<Self> {
        let half_n = 1 << (layer.n_variables() - 1);
        let mut next_numerators = Vec::with_capacity(half_n);
        let mut next_denominators = Vec::with_capacity(half_n);
        let one = BaseField::one();

        match layer {
            LogupTrace::Singles { denominators } => {
                for i in 0..half_n {
                    let d0 = denominators[i * 2];
                    let d1 = denominators[i * 2 + 1];

                    let res = Fraction::new(one, d0) + Fraction::new(one, d1);

                    next_numerators.push(res.numerator);
                    next_denominators.push(res.denominator);
                }
            }
            LogupTrace::Multiplicities {
                numerators,
                denominators,
            } => {
                for i in 0..half_n {
                    let n0 = numerators[i * 2];
                    let d0 = denominators[i * 2];

                    let n1 = numerators[i * 2 + 1];
                    let d1 = denominators[i * 2 + 1];

                    let res = Fraction::new(n0, d0) + Fraction::new(n1, d1);

                    next_numerators.push(res.numerator);
                    next_denominators.push(res.denominator);
                }
            }
            LogupTrace::Generic {
                numerators,
                denominators,
            } => {
                for i in 0..half_n {
                    let n0 = numerators[i * 2];
                    let d0 = denominators[i * 2];

                    let n1 = numerators[i * 2 + 1];
                    let d1 = denominators[i * 2 + 1];

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

    fn sum_as_poly_in_first_variable(
        h: &LogupOracle<'_, Self>,
        claim: SecureField,
    ) -> UnivariatePoly<SecureField> {
        let k = h.n_variables();
        let n_terms = 1 << (k - 1);
        let lambda = h.lambda();
        let eq_evals = h.eq_evals();
        let y = eq_evals.y();
        let trace = h.trace();

        let mut eval_at_0 = SecureField::zero();
        let mut eval_at_2 = SecureField::zero();

        for i in 0..n_terms {
            let (fraction0, fraction2) = match trace {
                LogupTrace::Singles { denominators } => {
                    let d0_lhs = denominators[i * 2];
                    let d1_lhs = denominators[i * 2 + 1];

                    let fraction0 = Fraction::new(d0_lhs + d1_lhs, d0_lhs * d1_lhs);

                    let d0_rhs = denominators[(n_terms + i) * 2];
                    let d1_rhs = denominators[(n_terms + i) * 2 + 1];

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

                    let d0_lhs = denominators[i * 2];
                    let d1_lhs = denominators[i * 2 + 1];

                    let fraction0 = {
                        let a = Fraction::new(n0_lhs, d0_lhs);
                        let b = Fraction::new(n1_lhs, d1_lhs);
                        a + b
                    };

                    let n0_rhs = numerators[(n_terms + i) * 2];
                    let n1_rhs = numerators[(n_terms + i) * 2 + 1];

                    let d0_rhs = denominators[(n_terms + i) * 2];
                    let d1_rhs = denominators[(n_terms + i) * 2 + 1];

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
                    let n0_lhs = numerators[i * 2];
                    let n1_lhs = numerators[i * 2 + 1];

                    let d0_lhs = denominators[i * 2];
                    let d1_lhs = denominators[i * 2 + 1];

                    let fraction0 = {
                        let a = Fraction::new(n0_lhs, d0_lhs);
                        let b = Fraction::new(n1_lhs, d1_lhs);
                        a + b
                    };

                    let n0_rhs = numerators[(n_terms + i) * 2];
                    let n1_rhs = numerators[(n_terms + i) * 2 + 1];

                    let d0_rhs = denominators[(n_terms + i) * 2];
                    let d1_rhs = denominators[(n_terms + i) * 2 + 1];

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

        eval_at_0 *= h.eq_fixed_var_correction();
        eval_at_2 *= h.eq_fixed_var_correction();

        correct_sum_as_poly_in_first_variable(eval_at_0, eval_at_2, claim, y, k)
    }
}

#[cfg(test)]
mod tests {
    use std::iter::zip;

    use num_traits::One;
    use rand::rngs::SmallRng;
    use rand::{Rng, SeedableRng};

    use crate::core::backend::CpuBackend;
    use crate::core::fields::m31::BaseField;
    use crate::core::fields::qm31::SecureField;
    use crate::core::lookups::gkr::{partially_verify_batch, prove_batch, GkrArtifact, GkrError};
    use crate::core::lookups::logup::{Fraction, LogupGate, LogupTrace};
    use crate::core::lookups::mle::Mle;
    use crate::core::test_utils::test_channel;

    #[test]
    fn logup_with_generic_trace_works() -> Result<(), GkrError> {
        const N: usize = 1 << 5;
        let mut rng = SmallRng::seed_from_u64(0);
        let numerator_values = (0..N).map(|_| rng.gen()).collect::<Vec<SecureField>>();
        let denominator_values = (0..N).map(|_| rng.gen()).collect::<Vec<SecureField>>();
        let sum = zip(&numerator_values, &denominator_values)
            .map(|(&n, &d)| Fraction::new(n, d))
            .sum::<Fraction<SecureField>>();
        let numerators = Mle::<CpuBackend, SecureField>::new(numerator_values);
        let denominators = Mle::<CpuBackend, SecureField>::new(denominator_values);
        let top_layer = LogupTrace::Generic {
            numerators: numerators.clone(),
            denominators: denominators.clone(),
        };
        let (proof, _) = prove_batch(&mut test_channel(), vec![top_layer]);

        let GkrArtifact {
            ood_point,
            claims_to_verify_by_component,
            n_variables_by_component: _,
        } = partially_verify_batch(vec![&LogupGate], &proof, &mut test_channel())?;

        assert_eq!(claims_to_verify_by_component.len(), 1);
        assert_eq!(proof.output_claims_by_component.len(), 1);
        assert_eq!(
            claims_to_verify_by_component[0],
            [
                numerators.eval_at_point(&ood_point),
                denominators.eval_at_point(&ood_point)
            ]
        );
        assert_eq!(
            proof.output_claims_by_component[0],
            [sum.numerator, sum.denominator]
        );
        Ok(())
    }

    #[test]
    fn logup_with_singles_trace_works() -> Result<(), GkrError> {
        const N: usize = 1 << 5;
        let mut rng = SmallRng::seed_from_u64(0);
        let denominator_values = (0..N).map(|_| rng.gen()).collect::<Vec<SecureField>>();
        let sum = denominator_values
            .iter()
            .map(|&d| Fraction::new(SecureField::one(), d))
            .sum::<Fraction<SecureField>>();
        let denominators = Mle::<CpuBackend, SecureField>::new(denominator_values);
        let top_layer = LogupTrace::Singles {
            denominators: denominators.clone(),
        };
        let (proof, _) = prove_batch(&mut test_channel(), vec![top_layer]);

        let GkrArtifact {
            ood_point,
            claims_to_verify_by_component,
            n_variables_by_component: _,
        } = partially_verify_batch(vec![&LogupGate], &proof, &mut test_channel())?;

        assert_eq!(claims_to_verify_by_component.len(), 1);
        assert_eq!(proof.output_claims_by_component.len(), 1);
        assert_eq!(
            claims_to_verify_by_component[0],
            [SecureField::one(), denominators.eval_at_point(&ood_point)]
        );
        assert_eq!(
            proof.output_claims_by_component[0],
            [sum.numerator, sum.denominator]
        );
        Ok(())
    }

    #[test]
    fn logup_with_multiplicities_trace_works() -> Result<(), GkrError> {
        const N: usize = 1 << 5;
        let mut rng = SmallRng::seed_from_u64(0);
        let numerator_values = (0..N).map(|_| rng.gen()).collect::<Vec<BaseField>>();
        let denominator_values = (0..N).map(|_| rng.gen()).collect::<Vec<SecureField>>();
        let sum = zip(&numerator_values, &denominator_values)
            .map(|(&n, &d)| Fraction::new(n.into(), d))
            .sum::<Fraction<SecureField>>();
        let numerators = Mle::<CpuBackend, BaseField>::new(numerator_values);
        let denominators = Mle::<CpuBackend, SecureField>::new(denominator_values);
        let top_layer = LogupTrace::Multiplicities {
            numerators: numerators.clone(),
            denominators: denominators.clone(),
        };
        let (proof, _) = prove_batch(&mut test_channel(), vec![top_layer]);

        let GkrArtifact {
            ood_point,
            claims_to_verify_by_component,
            n_variables_by_component: _,
        } = partially_verify_batch(vec![&LogupGate], &proof, &mut test_channel())?;

        assert_eq!(claims_to_verify_by_component.len(), 1);
        assert_eq!(proof.output_claims_by_component.len(), 1);
        assert_eq!(
            claims_to_verify_by_component[0],
            [
                numerators.eval_at_point(&ood_point),
                denominators.eval_at_point(&ood_point)
            ]
        );
        assert_eq!(
            proof.output_claims_by_component[0],
            [sum.numerator, sum.denominator]
        );
        Ok(())
    }
}
