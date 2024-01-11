use std::collections::BTreeSet;

use num_traits::One;

use crate::commitment_scheme::hasher::Hasher;
use crate::commitment_scheme::merkle_decommitment::MerkleDecommitment;
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
use crate::core::oods::{get_oods_quotient_combination, get_oods_values};
use crate::core::poly::circle::{CanonicCoset, CircleDomain, CircleEvaluation, PointMapping};
use crate::core::queries::{generate_queries, get_folded_queries};

type Channel = Blake2sChannel;
type MerkleHasher = <Channel as ChannelTrait>::ChannelHasher;

const LOG_BLOWUP_FACTOR: u32 = 1;
const N_QUERIES: usize = 3;

pub struct Fibonacci {
    pub trace_domain: CanonicCoset,
    pub trace_eval_domain: CircleDomain,
    pub trace_commitment_domain: CanonicCoset,
    pub constraint_zero_domain: Coset,
    pub composition_polynomial_eval_domain: CircleDomain,
    pub composition_polynomial_commitment_domain: CanonicCoset,
    pub claim: BaseField,
}

pub struct AdditionalProofData {
    pub composition_polynomial_oods_value: QM31,
    pub composition_polynomial_random_coeff: QM31,
    pub oods_point: CirclePoint<QM31>,
    pub composition_polynomial_queries: BTreeSet<usize>,
    pub trace_queries: BTreeSet<usize>,
}

pub struct CommitmentProof<F: ExtensionOf<BaseField>, H: Hasher> {
    pub decommitment: MerkleDecommitment<F, H>,
    pub commitment: H::Hash,
}

impl<F: ExtensionOf<BaseField> + IntoSlice<H::NativeType>, H: Hasher> CommitmentProof<F, H> {
    pub fn verify(&self, queries: BTreeSet<usize>) -> bool {
        self.decommitment.verify(self.commitment, queries)
    }
}

pub struct FibonacciProof {
    pub public_input: BaseField,
    pub trace_commitment: CommitmentProof<BaseField, MerkleHasher>,
    pub composition_polynomial_commitment: CommitmentProof<QM31, MerkleHasher>,
    // TODO(AlonH): Consider including only the values.
    pub trace_oods_evaluation: PointMapping<QM31>,
    pub additional_proof_data: AdditionalProofData,
}

impl Fibonacci {
    pub fn new(log_size: u32, claim: BaseField) -> Self {
        let trace_domain = CanonicCoset::new(log_size);
        let trace_eval_domain = trace_domain.evaluation_domain(log_size + 1);
        let trace_commitment_domain = CanonicCoset::new(log_size + LOG_BLOWUP_FACTOR);
        let constraint_zero_domain = Coset::subgroup(log_size);
        let composition_polynomial_eval_domain =
            CircleDomain::constraint_evaluation_domain(log_size + 1);
        let composition_polynomial_commitment_domain =
            CanonicCoset::new(log_size + 1 + LOG_BLOWUP_FACTOR);
        Self {
            trace_domain,
            trace_eval_domain,
            trace_commitment_domain,
            constraint_zero_domain,
            composition_polynomial_eval_domain,
            composition_polynomial_commitment_domain,
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
        let excluded0 = self
            .constraint_zero_domain
            .at(self.constraint_zero_domain.size() - 2);
        let excluded1 = self
            .constraint_zero_domain
            .at(self.constraint_zero_domain.size() - 1);
        let num = self.eval_step_constraint(trace)
            * point_excluder(excluded0, trace.point())
            * point_excluder(excluded1, trace.point());
        let denom = coset_vanishing(self.constraint_zero_domain, trace.point());
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
        let denom = point_vanishing(self.constraint_zero_domain.at(point_index), trace.point());
        num / denom
    }

    pub fn eval_composition_polynomial<F: ExtensionOf<BaseField>, EF: ExtensionOf<F>>(
        &self,
        random_coeff: EF,
        trace: impl PolyOracle<F>,
    ) -> EF {
        let mut value = random_coeff.pow(0) * self.eval_step_quotient(trace);
        value += random_coeff.pow(1) * self.eval_boundary_quotient(trace, 0, BaseField::one());
        value += random_coeff.pow(2)
            * self.eval_boundary_quotient(
                trace,
                self.constraint_zero_domain.size() - 1,
                self.claim,
            );
        value
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

    /// Returns the composition polynomial evaluations using the trace and a random coefficient.
    pub fn compute_composition_polynomial(
        &self,
        random_coeff: QM31,
        trace_evaluation: &CircleEvaluation<BaseField>,
    ) -> CircleEvaluation<QM31> {
        let mut composition_polynomial_values =
            Vec::with_capacity(self.composition_polynomial_eval_domain.size());
        for p_ind in self.composition_polynomial_eval_domain.iter_indices() {
            composition_polynomial_values.push(self.eval_composition_polynomial(
                random_coeff,
                EvalByEvaluation::new(p_ind, trace_evaluation),
            ));
        }
        CircleEvaluation::new(
            self.composition_polynomial_eval_domain,
            composition_polynomial_values,
        )
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

        let composition_polynomial_random_coeff = channel.draw_random_extension_felts()[0];
        let composition_polynomial = self
            .compute_composition_polynomial(composition_polynomial_random_coeff, &trace_evaluation);
        let composition_polynomial_poly = composition_polynomial.interpolate();
        let composition_polynomial_commitment_evaluation = composition_polynomial_poly.evaluate(
            self.composition_polynomial_commitment_domain
                .circle_domain(),
        );
        // TODO(AlonH): Remove the clone.
        let composition_polynomial_merkle = MerkleTree::<QM31, MerkleHasher>::commit(vec![
            composition_polynomial_commitment_evaluation.values.clone(),
        ]);
        channel.mix_with_seed(composition_polynomial_merkle.root());

        let oods_point = CirclePoint::<QM31>::get_random_point(channel);
        let mask = self.get_mask();
        let trace_oods_evaluation =
            get_oods_values(&mask, oods_point, &[self.trace_domain], &[trace_poly]);
        let composition_polynomial_oods_value =
            composition_polynomial_poly.eval_at_point(oods_point);

        let oods_random_coeff = channel.draw_random_extension_felts()[0];
        let combined_trace_oods_quotients = get_oods_quotient_combination(
            oods_random_coeff,
            &trace_oods_evaluation,
            &trace_commitment_evaluation,
        );
        let trace_oods_quotients_merkle =
            MerkleTree::<QM31, MerkleHasher>::commit(combined_trace_oods_quotients);
        channel.mix_with_seed(trace_oods_quotients_merkle.root());
        let composition_polynomial_oods_quotient = get_oods_quotient_combination(
            oods_random_coeff,
            &PointMapping::new([(oods_point, composition_polynomial_oods_value)].into()),
            &composition_polynomial_commitment_evaluation,
        );
        let composition_polynomial_oods_quotient_merkle =
            MerkleTree::<QM31, MerkleHasher>::commit(composition_polynomial_oods_quotient);
        channel.mix_with_seed(composition_polynomial_oods_quotient_merkle.root());

        let composition_polynomial_queries = generate_queries(
            channel,
            self.composition_polynomial_commitment_domain.log_size,
            N_QUERIES,
        );
        let trace_queries = get_folded_queries(
            &composition_polynomial_queries,
            self.composition_polynomial_commitment_domain.log_size
                - self.trace_commitment_domain.log_size,
        );
        let composition_polynomial_decommitment = composition_polynomial_merkle
            .generate_decommitment(composition_polynomial_queries.clone());
        let trace_decommitment = trace_merkle.generate_decommitment(trace_queries.clone());

        // TODO(AlonH): Complete the proof and add the relevant fields.
        FibonacciProof {
            public_input: self.claim,
            trace_commitment: CommitmentProof {
                decommitment: trace_decommitment,
                commitment: trace_merkle.root(),
            },
            composition_polynomial_commitment: CommitmentProof {
                decommitment: composition_polynomial_decommitment,
                commitment: composition_polynomial_merkle.root(),
            },
            trace_oods_evaluation,
            additional_proof_data: AdditionalProofData {
                composition_polynomial_oods_value,
                composition_polynomial_random_coeff,
                oods_point,
                composition_polynomial_queries,
                trace_queries,
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
            .constraint_zero_domain
            .iter_indices()
            .take(fib.constraint_zero_domain.size() - 2)
        {
            let res = fib.eval_step_constraint(EvalByEvaluation::new(p_ind, &trace));
            assert_eq!(res, BaseField::zero());
        }

        // Assert that the first trace value is 1.
        assert_eq!(
            fib.eval_boundary_constraint(
                EvalByEvaluation::new(fib.constraint_zero_domain.index_at(0), &trace,),
                BaseField::one()
            ),
            BaseField::zero()
        );

        // Assert that the last trace value is the fibonacci claim.
        assert_eq!(
            fib.eval_boundary_constraint(
                EvalByEvaluation::new(
                    fib.constraint_zero_domain
                        .index_at(fib.constraint_zero_domain.size() - 1),
                    &trace,
                ),
                fib.claim
            ),
            BaseField::zero()
        );
    }

    #[test]
    fn test_composition_polynomial_is_low_degree() {
        let fib = Fibonacci::new(5, m31!(443693538));
        let trace = fib.get_trace();
        let trace_poly = trace.interpolate();

        let extended_evaluation = trace_poly.evaluate(fib.trace_eval_domain);

        // TODO(ShaharS), Change to a channel implementation to retrieve the random
        // coefficients from extension field.
        let random_coeff = qm31!(2213980, 2213981, 2213982, 2213983);

        // Compute composition_polynomial on the evaluation domain.
        let mut composition_polynomial_values =
            Vec::with_capacity(fib.composition_polynomial_eval_domain.size());
        for p_ind in fib.composition_polynomial_eval_domain.iter_indices() {
            composition_polynomial_values.push(fib.eval_composition_polynomial(
                random_coeff,
                EvalByEvaluation::new(p_ind, &extended_evaluation),
            ));
        }
        let composition_polynomial_eval = CircleEvaluation::new(
            fib.composition_polynomial_eval_domain,
            composition_polynomial_values,
        );
        // Interpolate the poly. The poly is indeed of degree lower than the size of
        // trace_eval_domain, then it should interpolate correctly.
        let interpolated_composition_polynomial_poly = composition_polynomial_eval.interpolate();

        // Evaluate this polynomial at another point, out of trace_eval_domain and compare to what
        // we expect.
        let oods_point = CirclePoint::<QM31>::get_point(98989892);
        let trace_evaluator = EvalByPoly {
            point: oods_point,
            poly: &trace_poly,
        };

        assert_eq!(
            interpolated_composition_polynomial_poly.eval_at_point(oods_point),
            fib.eval_composition_polynomial(random_coeff, trace_evaluator)
        );
    }

    #[test]
    fn test_prove() {
        let fib = Fibonacci::new(5, m31!(443693538));

        let proof = fib.prove();
        let oods_point = proof.additional_proof_data.oods_point;
        let hz = fib.eval_composition_polynomial(
            proof
                .additional_proof_data
                .composition_polynomial_random_coeff,
            EvalByPointMapping {
                point: oods_point,
                point_mapping: &proof.trace_oods_evaluation,
            },
        );

        assert_eq!(
            proof
                .additional_proof_data
                .composition_polynomial_oods_value,
            hz
        );
        assert!(proof
            .composition_polynomial_commitment
            .verify(proof.additional_proof_data.composition_polynomial_queries));
        assert!(proof
            .trace_commitment
            .verify(proof.additional_proof_data.trace_queries));
    }
}
