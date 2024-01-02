use num_traits::One;

use crate::commitment_scheme::hasher::Hasher;
use crate::commitment_scheme::merkle_tree::MerkleTree;
use crate::core::air::{Mask, MaskItem};
use crate::core::channel::{Blake2sChannel, Channel as ChannelTrait};
use crate::core::circle::{CirclePoint, Coset};
use crate::core::constraints::{
    coset_vanishing, point_excluder, point_vanishing, EvalByEvaluation, PolyOracle,
};
use crate::core::fields::m31::BaseField;
use crate::core::fields::qm31::QM31;
use crate::core::fields::{ExtensionOf, Field, IntoSlice};
use crate::core::oods::{get_oods_quotient, get_oods_values};
use crate::core::poly::circle::{CanonicCoset, CircleDomain, CircleEvaluation, PointMapping};

type Channel = Blake2sChannel;
type MerkleHasher = <Channel as ChannelTrait>::ChannelHasher;

const BLOW_UP_FACTOR_BITS: usize = 1;

pub struct Fibonacci {
    pub trace_domain: CanonicCoset,
    pub trace_eval_domain: CircleDomain,
    pub trace_commitment_domain: CanonicCoset,
    pub constraint_domain: Coset,
    pub constraint_eval_domain: CircleDomain,
    pub quotient_commitment_domain: CanonicCoset,
    pub claim: BaseField,
}

pub struct AdditionalProofData {
    pub quotient_oods_value: QM31,
    pub quotient_random_coeff: QM31,
    pub oods_point: CirclePoint<QM31>,
}

pub struct FibonacciProof {
    pub public_input: BaseField,
    pub trace_commitment: <MerkleHasher as Hasher>::Hash,
    pub quotient_commitment: <MerkleHasher as Hasher>::Hash,
    // TODO(AlonH): Consider including only the values.
    pub trace_oods_evaluation: PointMapping<QM31>,
    pub additional_proof_data: AdditionalProofData,
}

impl Fibonacci {
    pub fn new(n_bits: usize, claim: BaseField) -> Self {
        let trace_domain = CanonicCoset::new(n_bits);
        let trace_eval_domain = trace_domain.evaluation_domain(n_bits + 1);
        let trace_commitment_domain = CanonicCoset::new(n_bits + BLOW_UP_FACTOR_BITS);
        let constraint_domain = Coset::subgroup(n_bits);
        let constraint_eval_domain = CircleDomain::constraint_evaluation_domain(n_bits + 1);
        let quotient_commitment_domain = CanonicCoset::new(n_bits + 1 + BLOW_UP_FACTOR_BITS);
        Self {
            trace_domain,
            trace_eval_domain,
            trace_commitment_domain,
            constraint_domain,
            constraint_eval_domain,
            quotient_commitment_domain,
            claim,
        }
    }

    pub fn get_trace(&self) -> CircleEvaluation<BaseField> {
        // Trace.
        let mut trace = Vec::with_capacity(self.trace_domain.size());

        // Fill trace with fibonacci squared.
        let mut a = BaseField::one();
        let mut b = BaseField::one();
        for _ in 0..self.trace_domain.size() {
            trace.push(a);
            let tmp = a.square() + b.square();
            a = b;
            b = tmp;
        }

        // Returns as a CircleEvaluation.
        CircleEvaluation::new_canonical_ordered(self.trace_domain, trace)
    }

    pub fn eval_step_constraint<F: ExtensionOf<BaseField>>(&self, trace: impl PolyOracle<F>) -> F {
        trace.get_at(self.trace_domain.index_at(0)).square()
            + trace.get_at(self.trace_domain.index_at(1)).square()
            - trace.get_at(self.trace_domain.index_at(2))
    }

    pub fn eval_step_quotient<F: ExtensionOf<BaseField>>(&self, trace: impl PolyOracle<F>) -> F {
        let excluded0 = self.constraint_domain.at(self.constraint_domain.size() - 2);
        let excluded1 = self.constraint_domain.at(self.constraint_domain.size() - 1);
        let num = self.eval_step_constraint(trace)
            * point_excluder(excluded0, trace.point())
            * point_excluder(excluded1, trace.point());
        let denom = coset_vanishing(self.constraint_domain, trace.point());
        num / denom
    }

    pub fn eval_boundary_constraint<F: ExtensionOf<BaseField>>(
        &self,
        trace: impl PolyOracle<F>,
        value: BaseField,
    ) -> F {
        trace.get_at(self.trace_domain.index_at(0)) - value
    }

    pub fn eval_boundary_quotient<F: ExtensionOf<BaseField>>(
        &self,
        trace: impl PolyOracle<F>,
        point_index: usize,
        value: BaseField,
    ) -> F {
        let num = self.eval_boundary_constraint(trace, value);
        let denom = point_vanishing(self.constraint_domain.at(point_index), trace.point());
        num / denom
    }

    pub fn eval_quotient<F: ExtensionOf<BaseField>, EF: ExtensionOf<F>>(
        &self,
        random_coeff: EF,
        trace: impl PolyOracle<F>,
    ) -> EF {
        let mut quotient = random_coeff.pow(0) * self.eval_step_quotient(trace);
        quotient += random_coeff.pow(1) * self.eval_boundary_quotient(trace, 0, BaseField::one());
        quotient += random_coeff.pow(2)
            * self.eval_boundary_quotient(trace, self.constraint_domain.size() - 1, self.claim);
        quotient
    }

    pub fn get_mask(&self) -> Mask {
        Mask::new(
            (0..3)
                .map(|offset| MaskItem {
                    column_index: 0,
                    offset,
                })
                .collect(),
        )
    }

    /// Returns the quotient values using the trace and a random coefficient.
    pub fn compute_quotient(
        &self,
        random_coeff: QM31,
        trace_evaluation: &CircleEvaluation<BaseField>,
    ) -> CircleEvaluation<QM31> {
        let mut quotient_values = Vec::with_capacity(self.constraint_eval_domain.size());
        for p_ind in self.constraint_eval_domain.iter_indices() {
            quotient_values.push(
                self.eval_quotient(random_coeff, EvalByEvaluation::new(p_ind, trace_evaluation)),
            );
        }
        CircleEvaluation::new(self.constraint_eval_domain, quotient_values)
    }

    pub fn prove(&self) -> FibonacciProof {
        let channel = &mut Channel::new(<Channel as ChannelTrait>::ChannelHasher::hash(
            BaseField::into_slice(&[self.claim]),
        ));
        let trace = self.get_trace();
        let trace_poly = trace.interpolate();
        let trace_evaluation = trace_poly.evaluate(self.trace_eval_domain);
        let trace_commitment_evaluation =
            trace_poly.evaluate(self.trace_commitment_domain.circle_domain());
        let trace_merkle =
            MerkleTree::<BaseField, MerkleHasher>::commit(vec![trace_commitment_evaluation
                .values
                .clone()]);
        channel.mix_with_seed(trace_merkle.root());

        let verifier_randomness = channel.draw_random_felts();
        let random_coeff = QM31::from_m31_array(verifier_randomness[..4].try_into().unwrap());
        let quotient = self.compute_quotient(random_coeff, &trace_evaluation);
        let quotient_poly = quotient.interpolate();
        let quotient_commitment_evaluation =
            quotient_poly.evaluate(self.quotient_commitment_domain.circle_domain());
        // TODO(AlonH): Remove the clone.
        let quotient_merkle =
            MerkleTree::<QM31, MerkleHasher>::commit(vec![quotient_commitment_evaluation
                .values
                .clone()]);
        channel.mix_with_seed(quotient_merkle.root());

        let oods_point = CirclePoint::<QM31>::get_random_point(channel);
        let mask = self.get_mask();
        let trace_oods_evaluation =
            get_oods_values(&mask, oods_point, &[self.trace_domain], &[trace_poly]);
        let quotient_oods_value = quotient_poly.eval_at_point(oods_point);

        // A quotient for each mask item and for the CP, in the OODS point and its conjugate.
        let mut oods_quotients = Vec::with_capacity((mask.len() + 1) * 2);
        for (point, value) in trace_oods_evaluation.iter() {
            oods_quotients.push(get_oods_quotient(
                *point,
                *value,
                &trace_commitment_evaluation,
            ));
        }
        oods_quotients.push(get_oods_quotient(
            oods_point,
            quotient_oods_value,
            &quotient_commitment_evaluation,
        ));

        // TODO(AlonH): Complete the proof and add the relevant fields.
        FibonacciProof {
            public_input: self.claim,
            trace_commitment: trace_merkle.root(),
            quotient_commitment: quotient_merkle.root(),
            trace_oods_evaluation,
            additional_proof_data: AdditionalProofData {
                quotient_oods_value,
                quotient_random_coeff: random_coeff,
                oods_point,
            },
        }
    }
}

#[cfg(test)]
mod tests {
    use num_traits::One;

    use super::Fibonacci;
    use crate::core::circle::CirclePoint;
    use crate::core::constraints::{EvalByEvaluation, EvalByPointMapping, EvalByPoly};
    use crate::core::fields::m31::{BaseField, M31};
    use crate::core::fields::qm31::QM31;
    use crate::core::poly::circle::CircleEvaluation;
    use crate::{m31, qm31};

    #[test]
    fn test_constraint_on_trace() {
        use num_traits::Zero;

        let fib = Fibonacci::new(3, m31!(1056169651));
        let trace = fib.get_trace();

        // Assert that the step constraint is satisfied on the trace.
        for p_ind in fib
            .constraint_domain
            .iter_indices()
            .take(fib.constraint_domain.size() - 2)
        {
            let res = fib.eval_step_constraint(EvalByEvaluation::new(p_ind, &trace));
            assert_eq!(res, BaseField::zero());
        }

        // Assert that the first trace value is 1.
        assert_eq!(
            fib.eval_boundary_constraint(
                EvalByEvaluation::new(fib.constraint_domain.index_at(0), &trace,),
                BaseField::one()
            ),
            BaseField::zero()
        );

        // Assert that the last trace value is the fibonacci claim.
        assert_eq!(
            fib.eval_boundary_constraint(
                EvalByEvaluation::new(
                    fib.constraint_domain
                        .index_at(fib.constraint_domain.size() - 1),
                    &trace,
                ),
                fib.claim
            ),
            BaseField::zero()
        );
    }

    #[test]
    fn test_quotient_is_low_degree() {
        let fib = Fibonacci::new(5, m31!(443693538));
        let trace = fib.get_trace();
        let trace_poly = trace.interpolate();

        let extended_evaluation = trace_poly.evaluate(fib.trace_eval_domain);

        // TODO(ShaharS), Change to a channel implementation to retrieve the random
        // coefficients from extension field.
        let random_coeff = qm31!(2213980, 2213981, 2213982, 2213983);

        // Compute quotient on the evaluation domain.
        let mut quotient_values = Vec::with_capacity(fib.constraint_eval_domain.size());
        for p_ind in fib.constraint_eval_domain.iter_indices() {
            quotient_values.push(fib.eval_quotient(
                random_coeff,
                EvalByEvaluation::new(p_ind, &extended_evaluation),
            ));
        }
        let quotient_eval = CircleEvaluation::new(fib.constraint_eval_domain, quotient_values);
        // Interpolate the poly. The poly is indeed of degree lower than the size of
        // trace_eval_domain, then it should interpolate correctly.
        let interpolated_quotient_poly = quotient_eval.interpolate();

        // Evaluate this polynomial at another point, out of trace_eval_domain and compare to what
        // we expect.
        let oods_point = CirclePoint::<QM31>::get_point(98989892);
        let trace_evaluator = EvalByPoly {
            point: oods_point,
            poly: &trace_poly,
        };

        assert_eq!(
            interpolated_quotient_poly.eval_at_point(oods_point),
            fib.eval_quotient(random_coeff, trace_evaluator)
        );
    }

    #[test]
    fn test_prove() {
        let fib = Fibonacci::new(5, m31!(443693538));

        let proof = fib.prove();

        let oods_point = proof.additional_proof_data.oods_point;
        let hz = fib.eval_quotient(
            proof.additional_proof_data.quotient_random_coeff,
            EvalByPointMapping {
                point: oods_point,
                point_mapping: &proof.trace_oods_evaluation,
            },
        );
        assert_eq!(proof.additional_proof_data.quotient_oods_value, hz);
    }
}
