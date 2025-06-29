use crate::core::backend::simd::SimdBackend;
use crate::core::backend::web::WebBackend;
use crate::core::fields::qm31::SecureField;
use crate::core::lookups::gkr_prover::{GkrMultivariatePolyOracle, GkrOps, Layer};
use crate::core::lookups::mle::Mle;
use crate::core::lookups::utils::UnivariatePoly;

impl GkrOps for WebBackend {
    #[allow(clippy::uninit_vec)]
    fn gen_eq_evals(y: &[SecureField], v: SecureField) -> Mle<Self, SecureField> {
        SimdBackend::gen_eq_evals(y, v).into()
    }

    fn next_layer(layer: &Layer<Self>) -> Layer<Self> {
        SimdBackend::next_layer(layer.as_ref()).into()
    }

    fn sum_as_poly_in_first_variable(
        h: &GkrMultivariatePolyOracle<'_, Self>,
        claim: SecureField,
    ) -> UnivariatePoly<SecureField> {
        SimdBackend::sum_as_poly_in_first_variable(h.as_ref(), claim)
    }
}

#[cfg(test)]
mod tests {
    use std::iter::zip;

    use num_traits::One;
    use rand::rngs::SmallRng;
    use rand::{Rng, SeedableRng};

    use super::WebBackend;
    // use crate::core::backend::simd::SimdBackend;
    use crate::core::backend::{Column, CpuBackend};
    use crate::core::channel::Channel;
    use crate::core::fields::m31::BaseField;
    use crate::core::fields::qm31::SecureField;
    use crate::core::lookups::gkr_prover::{prove_batch, GkrOps, Layer};
    use crate::core::lookups::gkr_verifier::{partially_verify_batch, Gate, GkrArtifact, GkrError};
    use crate::core::lookups::mle::Mle;
    use crate::core::lookups::utils::Fraction;
    use crate::core::test_utils::test_channel;

    #[test]
    fn gen_eq_evals_matches_cpu() {
        let two = BaseField::from(2).into();
        let y = [7, 3, 5, 6, 1, 1, 9].map(|v| BaseField::from(v).into());
        let eq_evals_cpu = CpuBackend::gen_eq_evals(&y, two);

        let eq_evals_simd = WebBackend::gen_eq_evals(&y, two);

        assert_eq!(eq_evals_simd.to_cpu(), *eq_evals_cpu);
    }

    #[test]
    fn gen_eq_evals_with_small_assignment_matches_cpu() {
        let two = BaseField::from(2).into();
        let y = [7, 3, 5].map(|v| BaseField::from(v).into());
        let eq_evals_cpu = CpuBackend::gen_eq_evals(&y, two);

        let eq_evals_simd = WebBackend::gen_eq_evals(&y, two);

        assert_eq!(eq_evals_simd.to_cpu(), *eq_evals_cpu);
    }

    #[test]
    fn grand_product_works() -> Result<(), GkrError> {
        const N: usize = 1 << 8;
        let values = test_channel().draw_secure_felts(N);
        let product = values.iter().product();
        let col = Mle::<WebBackend, SecureField>::new(values.into_iter().collect());
        let input_layer = Layer::GrandProduct(col.clone());
        let (proof, _) = prove_batch(&mut test_channel(), vec![input_layer]);

        let GkrArtifact {
            ood_point,
            claims_to_verify_by_instance,
            n_variables_by_instance: _,
        } = partially_verify_batch(vec![Gate::GrandProduct], &proof, &mut test_channel())?;

        assert_eq!(proof.output_claims_by_instance, [vec![product]]);
        assert_eq!(
            claims_to_verify_by_instance,
            [vec![col.eval_at_point(&ood_point)]]
        );
        Ok(())
    }

    #[test]
    fn logup_with_generic_trace_works() -> Result<(), GkrError> {
        const N: usize = 1 << 8;
        let mut rng = SmallRng::seed_from_u64(0);
        let numerators = (0..N).map(|_| rng.gen()).collect::<Vec<SecureField>>();
        let denominators = (0..N).map(|_| rng.gen()).collect::<Vec<SecureField>>();
        let sum = zip(&numerators, &denominators)
            .map(|(&n, &d)| Fraction::new(n, d))
            .sum::<Fraction<SecureField, SecureField>>();
        let numerators = Mle::<WebBackend, SecureField>::new(numerators.into_iter().collect());
        let denominators = Mle::<WebBackend, SecureField>::new(denominators.into_iter().collect());
        let input_layer = Layer::LogUpGeneric {
            numerators: numerators.clone(),
            denominators: denominators.clone(),
        };
        let (proof, _) = prove_batch(&mut test_channel(), vec![input_layer]);

        let GkrArtifact {
            ood_point,
            claims_to_verify_by_instance,
            n_variables_by_instance: _,
        } = partially_verify_batch(vec![Gate::LogUp], &proof, &mut test_channel())?;

        assert_eq!(claims_to_verify_by_instance.len(), 1);
        assert_eq!(proof.output_claims_by_instance.len(), 1);
        assert_eq!(
            claims_to_verify_by_instance[0],
            [
                numerators.eval_at_point(&ood_point),
                denominators.eval_at_point(&ood_point)
            ]
        );
        assert_eq!(
            proof.output_claims_by_instance[0],
            [sum.numerator, sum.denominator]
        );
        Ok(())
    }

    #[test]
    fn logup_with_multiplicities_trace_works() -> Result<(), GkrError> {
        const N: usize = 1 << 8;
        let mut rng = SmallRng::seed_from_u64(0);
        let numerators = (0..N).map(|_| rng.gen()).collect::<Vec<BaseField>>();
        let denominators = (0..N).map(|_| rng.gen()).collect::<Vec<SecureField>>();
        let sum = zip(&numerators, &denominators)
            .map(|(&n, &d)| Fraction::new(n.into(), d))
            .sum::<Fraction<SecureField, SecureField>>();
        let numerators = Mle::<WebBackend, BaseField>::new(numerators.into_iter().collect());
        let denominators = Mle::<WebBackend, SecureField>::new(denominators.into_iter().collect());
        let input_layer = Layer::LogUpMultiplicities {
            numerators: numerators.clone(),
            denominators: denominators.clone(),
        };
        let (proof, _) = prove_batch(&mut test_channel(), vec![input_layer]);

        let GkrArtifact {
            ood_point,
            claims_to_verify_by_instance,
            n_variables_by_instance: _,
        } = partially_verify_batch(vec![Gate::LogUp], &proof, &mut test_channel())?;

        assert_eq!(claims_to_verify_by_instance.len(), 1);
        assert_eq!(proof.output_claims_by_instance.len(), 1);
        assert_eq!(
            claims_to_verify_by_instance[0],
            [
                numerators.eval_at_point(&ood_point),
                denominators.eval_at_point(&ood_point)
            ]
        );
        assert_eq!(
            proof.output_claims_by_instance[0],
            [sum.numerator, sum.denominator]
        );
        Ok(())
    }

    #[test]
    fn logup_with_singles_trace_works() -> Result<(), GkrError> {
        const N: usize = 1 << 8;
        let mut rng = SmallRng::seed_from_u64(0);
        let denominators = (0..N).map(|_| rng.gen()).collect::<Vec<SecureField>>();
        let sum = denominators
            .iter()
            .map(|&d| Fraction::new(SecureField::one(), d))
            .sum::<Fraction<SecureField, SecureField>>();
        let denominators = Mle::<WebBackend, SecureField>::new(denominators.into_iter().collect());
        let input_layer = Layer::LogUpSingles {
            denominators: denominators.clone(),
        };
        let (proof, _) = prove_batch(&mut test_channel(), vec![input_layer]);

        let GkrArtifact {
            ood_point,
            claims_to_verify_by_instance,
            n_variables_by_instance: _,
        } = partially_verify_batch(vec![Gate::LogUp], &proof, &mut test_channel())?;

        assert_eq!(claims_to_verify_by_instance.len(), 1);
        assert_eq!(proof.output_claims_by_instance.len(), 1);
        assert_eq!(
            claims_to_verify_by_instance[0],
            [SecureField::one(), denominators.eval_at_point(&ood_point)]
        );
        assert_eq!(
            proof.output_claims_by_instance[0],
            [sum.numerator, sum.denominator]
        );
        Ok(())
    }
}
