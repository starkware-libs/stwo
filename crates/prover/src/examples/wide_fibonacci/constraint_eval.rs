use num_traits::Zero;
use stwo_verifier::core::fields::m31::BaseField;
use stwo_verifier::core::fields::qm31::SecureField;
use stwo_verifier::core::fields::MulGroup;

use super::component::{WideFibAir, WideFibComponent};
use crate::core::air::accumulation::DomainEvaluationAccumulator;
use crate::core::air::{AirProver, Component, ComponentProver, ComponentTrace};
use crate::core::backend::{CPUBackend, Column};
use crate::core::constraints::coset_vanishing;
use crate::core::poly::circle::CanonicCoset;
use crate::core::utils::bit_reverse;

impl AirProver<CPUBackend> for WideFibAir {
    fn prover_components(&self) -> Vec<&dyn ComponentProver<CPUBackend>> {
        vec![&self.component]
    }
}

impl ComponentProver<CPUBackend> for WideFibComponent {
    fn evaluate_constraint_quotients_on_domain(
        &self,
        trace: &ComponentTrace<'_, CPUBackend>,
        evaluation_accumulator: &mut DomainEvaluationAccumulator<CPUBackend>,
    ) {
        let max_constraint_degree = self.max_constraint_log_degree_bound();
        let trace_eval_domain = CanonicCoset::new(max_constraint_degree).circle_domain();
        let trace_evals = &trace.evals;
        let zero_domain = CanonicCoset::new(self.log_size).coset;
        let mut denoms = vec![];
        for point in trace_eval_domain.iter() {
            denoms.push(coset_vanishing(zero_domain, point));
        }
        bit_reverse(&mut denoms);
        let mut denom_inverses = vec![BaseField::zero(); 1 << (max_constraint_degree)];
        BaseField::batch_inverse(&denoms, &mut denom_inverses);
        let mut numerators = vec![SecureField::zero(); 1 << (max_constraint_degree)];
        let [mut accum] =
            evaluation_accumulator.columns([(max_constraint_degree, self.n_constraints())]);

        #[allow(clippy::needless_range_loop)]
        for i in 0..trace_eval_domain.size() {
            // Boundary constraint.
            numerators[i] += accum.random_coeff_powers[254]
                * (trace_evals[0].values.at(i) - BaseField::from_u32_unchecked(1));

            // Step constraints.
            for j in 0..254 {
                numerators[i] += accum.random_coeff_powers[253 - j]
                    * (trace_evals[j].values.at(i).square()
                        + trace_evals[j + 1].values.at(i).square()
                        - trace_evals[j + 2].values.at(i));
            }
        }
        for (i, (num, denom)) in numerators.iter().zip(denom_inverses.iter()).enumerate() {
            accum.accumulate(i, *num * *denom);
        }
    }
}
