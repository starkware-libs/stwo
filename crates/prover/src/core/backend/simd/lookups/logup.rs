use num_traits::Zero;

use crate::core::backend::simd::column::SecureFieldVec;
use crate::core::backend::simd::m31::N_LANES;
use crate::core::backend::simd::qm31::PackedSecureField;
use crate::core::backend::simd::SimdBackend;
use crate::core::backend::{Column, CpuBackend};
use crate::core::fields::qm31::SecureField;
use crate::core::lookups::gkr::correct_sum_as_poly_in_first_variable;
use crate::core::lookups::logup::{LogupOps, LogupOracle, LogupTrace};
use crate::core::lookups::mle::Mle;
use crate::core::lookups::sumcheck::MultivariatePolyOracle;
use crate::core::lookups::utils::UnivariatePoly;

impl LogupOps for SimdBackend {
    fn next_layer(layer: &LogupTrace<Self>) -> LogupTrace<Self> {
        let next_layer_len = layer.len() / 2;

        // Offload to CPU backend to prevent dealing with instances smaller than a SIMD vector.
        if next_layer_len < N_LANES {
            return to_simd_trace(&CpuBackend::next_layer(&layer.to_cpu()));
        }

        let next_layer_packed_len = next_layer_len / N_LANES;

        let mut next_numerators = Vec::with_capacity(next_layer_packed_len);
        let mut next_denominators = Vec::with_capacity(next_layer_packed_len);

        match layer {
            LogupTrace::Singles { denominators } => {
                for i in 0..next_layer_packed_len {
                    let (d_even, d_odd) =
                        denominators.data[i * 2].deinterleave(denominators.data[i * 2 + 1]);

                    let numerator = d_even + d_odd;
                    let denominator = d_even * d_odd;

                    next_numerators.push(numerator);
                    next_denominators.push(denominator);
                }
            }
            LogupTrace::Multiplicities {
                numerators,
                denominators,
            } => {
                for i in 0..next_layer_packed_len {
                    let (n_even, n_odd) =
                        numerators.data[i * 2].deinterleave(numerators.data[i * 2 + 1]);
                    let (d_even, d_odd) =
                        denominators.data[i * 2].deinterleave(denominators.data[i * 2 + 1]);

                    let numerator = d_even * n_odd + d_odd * n_even;
                    let denominator = d_even * d_odd;

                    next_numerators.push(numerator);
                    next_denominators.push(denominator);
                }
            }
            LogupTrace::Generic {
                numerators,
                denominators,
            } => {
                for i in 0..next_layer_packed_len {
                    let (n_even, n_odd) =
                        numerators.data[i * 2].deinterleave(numerators.data[i * 2 + 1]);
                    let (d_even, d_odd) =
                        denominators.data[i * 2].deinterleave(denominators.data[i * 2 + 1]);

                    let numerator = d_even * n_odd + d_odd * n_even;
                    let denominator = d_even * d_odd;

                    next_numerators.push(numerator);
                    next_denominators.push(denominator);
                }
            }
        }

        let next_numerators = SecureFieldVec {
            data: next_numerators,
            length: next_layer_len,
        };

        let next_denominators = SecureFieldVec {
            data: next_denominators,
            length: next_layer_len,
        };

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
        let eq_evals = h.eq_evals();
        let y = eq_evals.y();
        let trace = h.trace();

        // Offload to CPU backend to prevent dealing with instances smaller than a SIMD vector.
        if n_terms < 2 * N_LANES {
            return h.to_cpu().sum_as_poly_in_first_variable(claim);
        }

        let n_packed_terms = n_terms / N_LANES;

        let packed_lambda = PackedSecureField::broadcast(h.lambda());

        let mut packed_eval_at_0 = PackedSecureField::zero();
        let mut packed_eval_at_2 = PackedSecureField::zero();

        for i in 0..n_packed_terms {
            let numerator0;
            let denominator0;

            let numerator2;
            let denominator2;

            match trace {
                LogupTrace::Singles { denominators } => {
                    let (d_even_lhs, d_odd_lhs) =
                        denominators.data[i * 2].deinterleave(denominators.data[i * 2 + 1]);

                    numerator0 = d_even_lhs + d_odd_lhs;
                    denominator0 = d_even_lhs * d_odd_lhs;

                    let (d_even_rhs, d_odd_rhs) = denominators.data[(n_terms + i) * 2]
                        .deinterleave(denominators.data[(n_terms + i) * 2 + 1]);

                    let d2_even = d_even_rhs.double() - d_even_lhs;
                    let d2_odd = d_odd_rhs.double() - d_odd_lhs;

                    numerator2 = d2_even + d2_odd;
                    denominator2 = d2_even * d2_odd;
                }
                LogupTrace::Multiplicities {
                    numerators,
                    denominators,
                } => {
                    let (n_even_lhs, n_odd_lhs) =
                        numerators.data[i * 2].deinterleave(numerators.data[i * 2 + 1]);
                    let (d_even_lhs, d_odd_lhs) =
                        denominators.data[i * 2].deinterleave(denominators.data[i * 2 + 1]);

                    numerator0 = d_even_lhs * n_odd_lhs + d_odd_lhs * n_even_lhs;
                    denominator0 = d_even_lhs * d_odd_lhs;

                    let (n_even_rhs, n_odd_rhs) = numerators.data[(n_terms + i) * 2]
                        .deinterleave(numerators.data[(n_terms + i) * 2 + 1]);
                    let (d_even_rhs, d_odd_rhs) = denominators.data[(n_terms + i) * 2]
                        .deinterleave(denominators.data[(n_terms + i) * 2 + 1]);

                    let n2_even = n_even_rhs.double() - n_even_lhs;
                    let n2_odd = n_odd_rhs.double() - n_odd_lhs;
                    let d2_even = d_even_rhs.double() - d_even_lhs;
                    let d2_odd = d_odd_rhs.double() - d_odd_lhs;

                    numerator2 = d2_even * n2_odd + d2_odd * n2_even;
                    denominator2 = d2_even * d2_odd;
                }
                LogupTrace::Generic {
                    numerators,
                    denominators,
                } => {
                    let (n_even_lhs, n_odd_lhs) =
                        numerators.data[i * 2].deinterleave(numerators.data[i * 2 + 1]);
                    let (d_even_lhs, d_odd_lhs) =
                        denominators.data[i * 2].deinterleave(denominators.data[i * 2 + 1]);

                    numerator0 = d_even_lhs * n_odd_lhs + d_odd_lhs * n_even_lhs;
                    denominator0 = d_even_lhs * d_odd_lhs;

                    let (n_even_rhs, n_odd_rhs) = numerators.data[(n_terms + i) * 2]
                        .deinterleave(numerators.data[(n_terms + i) * 2 + 1]);
                    let (d_even_rhs, d_odd_rhs) = denominators.data[(n_terms + i) * 2]
                        .deinterleave(denominators.data[(n_terms + i) * 2 + 1]);

                    let n2_even = n_even_rhs.double() - n_even_lhs;
                    let n2_odd = n_odd_rhs.double() - n_odd_lhs;
                    let d2_even = d_even_rhs.double() - d_even_lhs;
                    let d2_odd = d_odd_rhs.double() - d_odd_lhs;

                    numerator2 = d2_even * n2_odd + d2_odd * n2_even;
                    denominator2 = d2_even * d2_odd;
                }
            };

            let eq_eval = eq_evals.data[i];
            packed_eval_at_0 += eq_eval * (numerator0 + packed_lambda * denominator0);
            packed_eval_at_2 += eq_eval * (numerator2 + packed_lambda * denominator2);
        }

        let eval_at_0 = packed_eval_at_0.pointwise_sum() * h.eq_fixed_var_correction();
        let eval_at_2 = packed_eval_at_2.pointwise_sum() * h.eq_fixed_var_correction();

        correct_sum_as_poly_in_first_variable(eval_at_0, eval_at_2, claim, y, k)
    }
}

fn to_simd_trace(cpu_trace: &LogupTrace<CpuBackend>) -> LogupTrace<SimdBackend> {
    match cpu_trace {
        LogupTrace::Singles { denominators } => LogupTrace::Singles {
            denominators: Mle::new(denominators.to_cpu().into_iter().collect()),
        },
        LogupTrace::Multiplicities {
            numerators,
            denominators,
        } => LogupTrace::Multiplicities {
            numerators: Mle::new(numerators.to_cpu().into_iter().collect()),
            denominators: Mle::new(denominators.to_cpu().into_iter().collect()),
        },
        LogupTrace::Generic {
            numerators,
            denominators,
        } => LogupTrace::Generic {
            numerators: Mle::new(numerators.to_cpu().into_iter().collect()),
            denominators: Mle::new(denominators.to_cpu().into_iter().collect()),
        },
    }
}

#[cfg(test)]
mod tests {
    use std::iter::zip;

    use num_traits::One;
    use rand::rngs::SmallRng;
    use rand::{Rng, SeedableRng};

    use crate::core::backend::simd::SimdBackend;
    use crate::core::fields::m31::BaseField;
    use crate::core::fields::qm31::SecureField;
    use crate::core::lookups::gkr::{partially_verify_batch, prove_batch, GkrArtifact, GkrError};
    use crate::core::lookups::logup::{Fraction, LogupGate, LogupTrace};
    use crate::core::lookups::mle::Mle;
    use crate::core::test_utils::test_channel;

    #[test]
    fn logup_with_generic_trace_works() -> Result<(), GkrError> {
        const N: usize = 1 << 6;
        let mut rng = SmallRng::seed_from_u64(0);
        let numerators = (0..N).map(|_| rng.gen()).collect::<Vec<SecureField>>();
        let denominators = (0..N).map(|_| rng.gen()).collect::<Vec<SecureField>>();
        let sum = zip(&numerators, &denominators)
            .map(|(&n, &d)| Fraction::new(n, d))
            .sum::<Fraction<SecureField>>();
        let numerators = Mle::<SimdBackend, SecureField>::new(numerators.into_iter().collect());
        let denominators = Mle::<SimdBackend, SecureField>::new(denominators.into_iter().collect());
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
        const N: usize = 1 << 6;
        let mut rng = SmallRng::seed_from_u64(0);
        let denominators = (0..N).map(|_| rng.gen()).collect::<Vec<SecureField>>();
        let sum = denominators
            .iter()
            .map(|&d| Fraction::new(SecureField::one(), d))
            .sum::<Fraction<SecureField>>();
        let denominators = Mle::<SimdBackend, SecureField>::new(denominators.into_iter().collect());
        let top_layer = LogupTrace::Singles {
            denominators: denominators.clone(),
        };
        let (proof, _) = prove_batch(&mut test_channel(), vec![top_layer.clone()]);

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
        const N: usize = 1 << 6;
        let mut rng = SmallRng::seed_from_u64(0);
        let numerators = (0..N).map(|_| rng.gen()).collect::<Vec<BaseField>>();
        let denominators = (0..N).map(|_| rng.gen()).collect::<Vec<SecureField>>();
        let sum = zip(&numerators, &denominators)
            .map(|(&n, &d)| Fraction::new(n.into(), d))
            .sum::<Fraction<SecureField>>();
        let numerators = Mle::<SimdBackend, BaseField>::new(numerators.into_iter().collect());
        let denominators = Mle::<SimdBackend, SecureField>::new(denominators.into_iter().collect());
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
