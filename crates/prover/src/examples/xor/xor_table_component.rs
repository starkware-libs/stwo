use std::collections::BTreeMap;

use itertools::Itertools;

use crate::core::air::accumulation::{DomainEvaluationAccumulator, PointEvaluationAccumulator};
use crate::core::air::{Component, ComponentProver, ComponentTrace, LookupInstanceConfig};
use crate::core::backend::{ColumnOps, CpuBackend};
use crate::core::circle::CirclePoint;
use crate::core::fields::m31::BaseField;
use crate::core::fields::qm31::SecureField;
use crate::core::fields::FieldExpOps;
use crate::core::lookups::gkr_prover::Layer;
use crate::core::lookups::gkr_verifier::Gate;
use crate::core::lookups::mle::Mle;
use crate::core::pcs::TreeVec;
use crate::core::poly::circle::CircleEvaluation;
use crate::core::poly::BitReversedOrder;
use crate::core::{ColumnVec, InteractionElements, LookupValues};

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
        _point: CirclePoint<SecureField>,
    ) -> TreeVec<ColumnVec<Vec<CirclePoint<SecureField>>>> {
        TreeVec::new(vec![vec![vec![]], vec![]])
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

    fn gkr_lookup_instance_configs(&self) -> Vec<LookupInstanceConfig> {
        vec![LookupInstanceConfig {
            variant: Gate::LogUp,
            is_table: true,
            table_id: XOR_LOOKUP_TABLE_ID.to_string(),
        }]
    }

    fn eval_at_point_iop_claims_by_n_variables(
        &self,
        _multilinear_eval_claims_by_instance: &[Vec<SecureField>],
    ) -> BTreeMap<u32, Vec<SecureField>> {
        let mut claims_by_n_variables = BTreeMap::new();
        let table_n_variables = u8::BITS + u8::BITS;
        let numerator_claim = _multilinear_eval_claims_by_instance[0][0];
        claims_by_n_variables.insert(table_n_variables, vec![numerator_claim]);
        claims_by_n_variables
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

    fn build_lookup_instances(
        &self,
        trace: ColumnVec<&CircleEvaluation<CpuBackend, BaseField, BitReversedOrder>>,
        interaction_elements: &InteractionElements,
    ) -> Vec<Layer<CpuBackend>> {
        let z = interaction_elements[XOR_Z_ID];
        let alpha = interaction_elements[XOR_ALPHA_ID];

        let xor_multiplicities = &trace[0];

        let numerators = xor_multiplicities.to_vec();
        let mut denominators = (0..256 * 256)
            .map(|i| {
                let lhs = i & 0xFF;
                let rhs = i >> 8;
                let res = lhs ^ rhs;
                z - BaseField::from(lhs)
                    - alpha * BaseField::from(rhs)
                    - alpha.square() * BaseField::from(res)
            })
            .collect_vec();
        CpuBackend::bit_reverse_column(&mut denominators);

        vec![Layer::LogUpMultiplicities {
            numerators: Mle::new(numerators),
            denominators: Mle::new(denominators),
        }]
    }

    fn lookup_multilinears_for_eval_at_point_iop(
        &self,
        lookup_layers: Vec<Layer<CpuBackend>>,
    ) -> Vec<Mle<CpuBackend, SecureField>> {
        match lookup_layers.into_iter().next().unwrap() {
            Layer::LogUpMultiplicities {
                numerators,
                denominators: _,
            } => {
                let numerator_evals = numerators.into_evals();
                vec![Mle::new(
                    numerator_evals.into_iter().map(|e| e.into()).collect(),
                )]
            }
            _ => panic!(),
        }
    }
}

// struct XorTableNumeratorsEvaluator;

// impl TraceExprPolynomial for XorTableNumeratorsEvaluator {
//     fn eval(
//         &self,
//         _interaction_elements: InteractionElements,
//         mask: &ColumnVec<Vec<SecureField>>,
//     ) -> SecureField {
//         mask[0][0]
//     }
// }

// struct XorTableDenominatorsEvaluator;

// impl MultilinearPolynomial for XorTableDenominatorsEvaluator {
//     fn eval(
//         &self,
//         interaction_elements: InteractionElements,
//         point: &[SecureField],
//     ) -> SecureField {
//         let alpha = interaction_elements[XOR_ALPHA_ID];
//         let z = interaction_elements[XOR_Z_ID];

//         match point {
//             // `li`: left operand bit `i`, `ri`: right operand bit `i`
//             &[ref _unused @ .., l0, l1, l2, l3, l4, l5, l6, l7, r0, r1, r2, r3, r4, r5, r6, r7]
// => {                 let lhs_assignment = [l0, l1, l2, l3, l4, l5, l6, l7];
//                 let rhs_assignment = [r0, r1, r2, r3, r4, r5, r6, r7];

//                 let xor_assignment: [SecureField; 8] = array::from_fn(|i| {
//                     let a = lhs_assignment[i];
//                     let b = rhs_assignment[i];

//                     // Note `a ^ b = 1 - a * b - (1 - a)(1 - b)` for all `a, b` in `{0, 1}`.
//                     SecureField::one() - a * b - (SecureField::one() - a) * (SecureField::one() -
// b)                 });

//                 let two = BaseField::from(2).into();
//                 let lhs = horner_eval(&lhs_assignment, two);
//                 let rhs = horner_eval(&rhs_assignment, two);
//                 let xor = horner_eval(&xor_assignment, two);

//                 z - lhs - rhs * alpha - xor * alpha.square()
//             }
//             // TODO: Add error handing to polynomial types.
//             _ => panic!(),
//         }
//     }
// }
