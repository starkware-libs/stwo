use std::array;
use std::collections::BTreeSet;

use itertools::Itertools;

use super::gkr_lookup_component::prover::{GkrLookupComponentProver, MleAccumulator};
use super::gkr_lookup_component::verifier::{
    GkrLookupComponent, MleClaimAccumulator, UnivariateClaimAccumulator,
};
use crate::core::air::accumulation::{DomainEvaluationAccumulator, PointEvaluationAccumulator};
use crate::core::air::{Component, ComponentProver, ComponentTrace, ComponentTraceWriter};
use crate::core::backend::CpuBackend;
use crate::core::circle::CirclePoint;
use crate::core::fields::m31::BaseField;
use crate::core::fields::qm31::SecureField;
use crate::core::fields::{Field, FieldExpOps};
use crate::core::lookups::gkr_prover::Layer;
use crate::core::lookups::gkr_verifier::Gate;
use crate::core::lookups::mle::Mle;
use crate::core::lookups::utils::horner_eval;
use crate::core::pcs::TreeVec;
use crate::core::poly::circle::CircleEvaluation;
use crate::core::poly::BitReversedOrder;
use crate::core::{ColumnVec, InteractionElements, LookupValues};
use crate::examples::xor::gkr_lookup_component::verifier::LookupInstanceConfig;

pub const XOR_LOOKUP_TABLE_ID: &str = "XOR_8_BIT";

pub const XOR_Z_ID: &str = "XOR_8_BIT_Z";

pub const XOR_ALPHA_ID: &str = "XOR_8_BIT_ALPHA";

/// 8-bit XOR lookup table.
pub struct XorTableComponent;

impl Component for XorTableComponent {
    fn n_constraints(&self) -> usize {
        0
    }

    fn max_constraint_log_degree_bound(&self) -> u32 {
        u8::BITS + u8::BITS
    }

    fn trace_log_degree_bounds(&self) -> TreeVec<ColumnVec<u32>> {
        TreeVec::new(vec![vec![u8::BITS + u8::BITS], vec![]])
    }

    fn mask_points(
        &self,
        point: CirclePoint<SecureField>,
    ) -> TreeVec<ColumnVec<Vec<CirclePoint<SecureField>>>> {
        TreeVec::new(vec![vec![vec![point]], vec![]])
    }

    fn interaction_element_ids(&self) -> Vec<String> {
        vec![XOR_ALPHA_ID.to_string(), XOR_Z_ID.to_string()]
    }

    fn evaluate_constraint_quotients_at_point(
        &self,
        _point: CirclePoint<SecureField>,
        _mask: &ColumnVec<Vec<SecureField>>,
        _evaluation_accumulator: &mut PointEvaluationAccumulator,
        _interaction_elements: &InteractionElements,
        _lookup_values: &LookupValues,
    ) {
    }

    fn n_interaction_phases(&self) -> u32 {
        0
    }
}

impl GkrLookupComponent for XorTableComponent {
    fn lookup_config(&self) -> Vec<LookupInstanceConfig> {
        vec![LookupInstanceConfig {
            variant: Gate::LogUp,
            is_table: true,
            table_id: XOR_LOOKUP_TABLE_ID.to_string(),
        }]
    }

    // fn n_lookup_columns_for_univariate_iop(&self) -> usize {
    //     1
    // }

    // fn max_lookup_column_for_univariate_iop_log_size(&self) -> u32 {
    //     u8::BITS + u8::BITS
    // }

    // fn validate_succinct_multilinear_lookup_column_claims(
    //     &self,
    //     ood_point: &[SecureField],
    //     multilinear_claims_by_instance: &[Vec<SecureField>],
    //     interaction_elements: &InteractionElements,
    // ) -> bool {
    //     let denominator_polynomial = XorDenominatorMle {
    //         alpha: interaction_elements[XOR_ALPHA_ID],
    //         z: interaction_elements[XOR_Z_ID],
    //     };

    //     let denominator_claim = multilinear_claims_by_instance[0][1];

    //     let (&assignment, _) = ood_point.split_last_chunk::<16>().unwrap();

    //     denominator_claim == denominator_polynomial.eval(assignment)
    // }

    // fn accumulate_lookup_column_claims_for_univariate_iop(
    //     &self,
    //     multilinear_claims_by_instance: &[Vec<SecureField>],
    //     accumulator: &mut SizedClaimAccumulator,
    // ) {
    //     let numberator_column_n_variables = 16;
    //     let numerator_column_claim = multilinear_claims_by_instance[0][0];
    //     accumulator.accumulate(numberator_column_n_variables, numerator_column_claim)
    // }

    fn mle_n_variables_for_univariate_iop(&self) -> BTreeSet<u32> {
        BTreeSet::from_iter([u8::BITS + u8::BITS])
    }

    fn validate_succinct_mle_claims(
        &self,
        ood_point: &[SecureField],
        multilinear_claims_by_instance: &[Vec<SecureField>],
        interaction_elements: &InteractionElements,
    ) -> bool {
        let denominator_polynomial = XorDenominatorMle {
            alpha: interaction_elements[XOR_ALPHA_ID],
            z: interaction_elements[XOR_Z_ID],
        };

        let denominator_claim = multilinear_claims_by_instance[0][1];

        let (&assignment, _) = ood_point.split_last_chunk::<16>().unwrap();

        denominator_claim == denominator_polynomial.eval(assignment)
    }

    fn accumulate_mle_claims_for_univariate_iop(
        &self,
        mle_claims_by_instance: &[Vec<SecureField>],
        accumulator: &mut MleClaimAccumulator,
    ) {
        let numerators_n_variables = 16;
        let numerators_claim = mle_claims_by_instance[0][0];
        accumulator.accumulate(numerators_n_variables, numerators_claim)
    }

    fn evaluate_lookup_columns_for_univariate_iop_at_point(
        &self,
        _point: CirclePoint<SecureField>,
        mask: &ColumnVec<Vec<SecureField>>,
        accumulator: &mut UnivariateClaimAccumulator,
        _interaction_elements: &InteractionElements,
    ) {
        let numerators_log_size = 16;
        let numerators_eval = mask[0][0];
        accumulator.accumulate(numerators_log_size, numerators_eval);
    }
}

impl ComponentTraceWriter<CpuBackend> for XorTableComponent {
    fn write_interaction_trace(
        &self,
        _trace: &ColumnVec<&CircleEvaluation<CpuBackend, BaseField, BitReversedOrder>>,
        _elements: &InteractionElements,
    ) -> ColumnVec<CircleEvaluation<CpuBackend, BaseField, BitReversedOrder>> {
        vec![]
    }
}

impl ComponentProver<CpuBackend> for XorTableComponent {
    fn evaluate_constraint_quotients_on_domain(
        &self,
        _trace: &ComponentTrace<'_, CpuBackend>,
        _evaluation_accumulator: &mut DomainEvaluationAccumulator<CpuBackend>,
        _interaction_elements: &InteractionElements,
        _lookup_values: &LookupValues,
    ) {
    }
}

impl GkrLookupComponentProver<CpuBackend> for XorTableComponent {
    fn accumulate_mle_for_univariate_iop(
        &self,
        lookup_instances: Vec<Layer<CpuBackend>>,
        accumulator: &mut MleAccumulator<CpuBackend>,
    ) {
        let max_column_log_degree = 16;
        let acc_coeff = accumulator.accumulation_coeff();
        let acc_mle = accumulator.column(max_column_log_degree);

        let numerators = match &lookup_instances[0] {
            Layer::LogUpMultiplicities { numerators, .. } => numerators,
            _ => panic!(),
        };

        for (i, &eval) in numerators.iter().enumerate() {
            acc_mle[i] = acc_mle[i] * acc_coeff + eval;
        }
    }

    fn write_lookup_instances(
        &self,
        trace: ColumnVec<&CircleEvaluation<CpuBackend, BaseField, BitReversedOrder>>,
        interaction_elements: &InteractionElements,
    ) -> Vec<Layer<CpuBackend>> {
        let z = interaction_elements[XOR_Z_ID];
        let alpha = interaction_elements[XOR_ALPHA_ID];

        let xor_multiplicities = &trace[0];

        let numerators = xor_multiplicities.to_vec();
        let denominators = (0..1usize << 16)
            .map(|i| {
                // We want values in bit-reversed order.
                let i_bit_reverse = (i as u16).reverse_bits() as usize;
                let lhs = i_bit_reverse & 0xFF;
                let rhs = i_bit_reverse >> 8;
                let res = lhs ^ rhs;
                z - BaseField::from(lhs)
                    - alpha * BaseField::from(rhs)
                    - alpha.square() * BaseField::from(res)
            })
            .collect_vec();

        vec![Layer::LogUpMultiplicities {
            numerators: Mle::new(numerators),
            denominators: Mle::new(denominators),
        }]
    }
}

/// Succinct multilinear polynomial representing the LogUp denominator terms for an 8-bit xor table.
struct XorDenominatorMle {
    alpha: SecureField,
    z: SecureField,
}

impl XorDenominatorMle {
    /// Evaluates the multilinear polynomial at the assignment.
    ///
    /// Parameters:
    /// - `li`: left operand bit `i`,
    /// - `ri`: right operand bit `i`
    fn eval(
        &self,
        [l0, l1, l2, l3, l4, l5, l6, l7, r0, r1, r2, r3, r4, r5, r6, r7]: [SecureField; 16],
    ) -> SecureField {
        let lhs_assignment = [l0, l1, l2, l3, l4, l5, l6, l7];
        let rhs_assignment = [r0, r1, r2, r3, r4, r5, r6, r7];

        let xor_assignment: [SecureField; 8] = array::from_fn(|i| {
            let li = lhs_assignment[i];
            let ri = rhs_assignment[i];

            // Note `a ^ b = a + b - 2 * a * b` for all `a, b` in `{0, 1}`.
            li + ri - (li * ri).double()
        });

        let two = BaseField::from(2).into();
        let lhs = horner_eval(&lhs_assignment, two);
        let rhs = horner_eval(&rhs_assignment, two);
        let xor = horner_eval(&xor_assignment, two);

        self.z - lhs - rhs * self.alpha - xor * self.alpha.square()
    }
}
