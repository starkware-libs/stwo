use num_traits::One;

use crate::commitment_scheme::hasher::Hasher;
use crate::commitment_scheme::merkle_tree::MerkleTree;
use crate::core::air::{Mask, MaskItem};
use crate::core::channel::{Blake2sChannel, Channel as ChannelTrait};
use crate::core::circle::{CirclePoint, Coset};
use crate::core::constraints::{
    coset_vanishing, point_excluder, point_vanishing, EvalByEvaluation, EvalByPoly, PolyOracle,
};
use crate::core::fields::m31::BaseField;
use crate::core::fields::qm31::QM31;
use crate::core::fields::{ExtensionOf, Field, IntoSlice};
use crate::core::poly::circle::{
    CanonicCoset, CircleDomain, CircleEvaluation, CirclePoly, PointSetEvaluation,
};

type Channel = Blake2sChannel;
type MerkleHasher = <Channel as ChannelTrait>::ChannelHasher;

pub struct Fibonacci {
    pub trace_domain: CanonicCoset,
    pub trace_eval_domain: CircleDomain,
    pub constraint_domain: Coset,
    pub constraint_eval_domain: CircleDomain,
    pub claim: BaseField,
}

pub struct FibonacciProof {
    pub public_input: BaseField,
    pub trace_commitment: <MerkleHasher as Hasher>::Hash,
    pub quotient_commitment: <MerkleHasher as Hasher>::Hash,
    pub oods_eval: PointSetEvaluation<QM31>,
}

impl Fibonacci {
    pub fn new(n_bits: usize, claim: BaseField) -> Self {
        let trace_domain = CanonicCoset::new(n_bits);
        let trace_eval_domain = trace_domain.evaluation_domain(n_bits + 1);
        let constraint_domain = Coset::subgroup(n_bits);
        let constraint_eval_domain = CircleDomain::constraint_evaluation_domain(n_bits + 1);
        Self {
            trace_domain,
            trace_eval_domain,
            constraint_domain,
            constraint_eval_domain,
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

    /// Returns the quotient value for the OODS point.
    pub fn eval_mask_quotient(
        &self,
        trace: impl PolyOracle<BaseField>,
        oods_point: CirclePoint<QM31>,
        oods_value: QM31,
    ) -> QM31 {
        let num = trace.get_at(self.trace_domain.index_at(0)) - oods_value;
        let denom: QM31 = point_vanishing(oods_point, trace.point().into_ef());
        num / denom
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
        channel: &mut Channel,
        trace_eval: &CircleEvaluation<BaseField>,
    ) -> CircleEvaluation<QM31> {
        let verifier_randomness = channel.draw_random_felts();
        let random_coeff = QM31::from_m31_array(verifier_randomness[..4].try_into().unwrap());
        let mut quotient_values = Vec::with_capacity(self.constraint_eval_domain.size());
        for p_ind in self.constraint_eval_domain.iter_indices() {
            quotient_values.push(self.eval_quotient(
                random_coeff,
                EvalByEvaluation {
                    offset: p_ind,
                    eval: trace_eval,
                },
            ));
        }
        CircleEvaluation::new(self.constraint_eval_domain, quotient_values)
    }

    /// Returns the mask values for the OODS point.
    pub fn get_oods_values(
        &self,
        channel: &mut Channel,
        trace_poly: &CirclePoly<BaseField>,
    ) -> PointSetEvaluation<QM31> {
        let oods_point = CirclePoint::<QM31>::get_random_point(channel);
        let oods_eval = EvalByPoly {
            point: oods_point,
            poly: trace_poly,
        };
        let oods_conjugate_eval = EvalByPoly {
            point: -oods_point,
            poly: trace_poly,
        };
        self.get_mask()
            .get_evaluation(&[self.trace_domain], &[oods_eval], &[oods_conjugate_eval])
    }

    pub fn get_mask_quotient(
        &self,
        point: CirclePoint<QM31>,
        value: QM31,
        eval: &CircleEvaluation<BaseField>,
    ) -> Vec<QM31> {
        let domain = CanonicCoset::new(self.trace_domain.n_bits + 1);
        let mut values = Vec::with_capacity(domain.size());
        for p_ind in domain.iter_indices() {
            values.push(self.eval_mask_quotient(
                EvalByEvaluation {
                    offset: p_ind,
                    eval,
                },
                point,
                value,
            ));
        }
        values
    }

    pub fn commit_on_poly<F: ExtensionOf<BaseField>>(
        &self,
        poly: &CirclePoly<F>,
        n_bits: usize,
    ) -> MerkleTree<F, MerkleHasher> {
        let commitment_domain = CanonicCoset::new(n_bits);
        let evaluation = poly.evaluate(commitment_domain.circle_domain());
        MerkleTree::<F, MerkleHasher>::commit(vec![evaluation.values])
    }

    pub fn prove(self) -> FibonacciProof {
        let mut channel = Channel::new(<Channel as ChannelTrait>::ChannelHasher::hash(
            BaseField::into_slice(&[self.claim]),
        ));
        let trace = self.get_trace();
        let trace_poly = trace.interpolate();
        let trace_evaluation = trace_poly.evaluate(self.trace_eval_domain);
        let trace_merkle = self.commit_on_poly(&trace_poly, self.trace_domain.n_bits + 1);
        channel.mix_with_seed(trace_merkle.root());

        let quotient = self.compute_quotient(&mut channel, &trace_evaluation);
        let quotient_poly = quotient.interpolate();
        let quotient_merkle =
            self.commit_on_poly(&quotient_poly, self.constraint_eval_domain.n_bits() + 1);
        channel.mix_with_seed(quotient_merkle.root());

        let oods_eval = self.get_oods_values(&mut channel, &trace_poly);
        let mut mask_quotients = Vec::with_capacity(oods_eval.len());
        for (point, value) in oods_eval.iter() {
            mask_quotients.push(self.get_mask_quotient(*point, *value, &trace_evaluation));
        }

        // TODO(AlonH): Complete the proof and add the relevant fields.
        FibonacciProof {
            public_input: self.claim,
            trace_commitment: trace_merkle.root(),
            quotient_commitment: quotient_merkle.root(),
            oods_eval,
        }
    }
}

#[cfg(test)]
mod tests {
    use num_traits::One;

    use super::Fibonacci;
    use crate::core::circle::CirclePoint;
    use crate::core::constraints::{EvalByEvaluation, EvalByPoly};
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
            let res = fib.eval_step_constraint(EvalByEvaluation {
                offset: p_ind,
                eval: &trace,
            });
            assert_eq!(res, BaseField::zero());
        }

        // Assert that the first trace value is 1.
        assert_eq!(
            fib.eval_boundary_constraint(
                EvalByEvaluation {
                    offset: fib.constraint_domain.index_at(0),
                    eval: &trace,
                },
                BaseField::one()
            ),
            BaseField::zero()
        );

        // Assert that the last trace value is the fibonacci claim.
        assert_eq!(
            fib.eval_boundary_constraint(
                EvalByEvaluation {
                    offset: fib
                        .constraint_domain
                        .index_at(fib.constraint_domain.size() - 1),
                    eval: &trace,
                },
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
                EvalByEvaluation {
                    offset: p_ind,
                    eval: &extended_evaluation,
                },
            ));
        }
        let quotient_eval = CircleEvaluation::new(fib.constraint_eval_domain, quotient_values);
        // Interpolate the poly. The poly is indeed of degree lower than the size of
        // eval_domain, then it should interpolate correctly.
        let interpolated_quotient_poly = quotient_eval.interpolate();

        // Evaluate this polynomial at another point, out of eval_domain and compare to what we
        // expect.
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
}