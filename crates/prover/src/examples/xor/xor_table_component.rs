use std::array;

use num_traits::One;

use crate::core::air::accumulation::PointEvaluationAccumulator;
use crate::core::air::{
    ColumnEvaluator, Component, LookupConfig, LookupEvaluator, LookupVariant,
    MultilinearPolynomial, TraceExprPolynomial,
};
use crate::core::circle::CirclePoint;
use crate::core::fields::m31::BaseField;
use crate::core::fields::qm31::SecureField;
use crate::core::fields::FieldExpOps;
use crate::core::lookups::utils::horner_eval;
use crate::core::pcs::TreeVec;
use crate::core::{ColumnVec, InteractionElements, LookupValues};

pub const XOR_Z_ID: &str = "xor_z";

pub const XOR_ALPHA_ID: &str = "xor_alpha";

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
        TreeVec::new(vec![vec![u8::BITS + u8::BITS]])
    }

    fn mask_points(
        &self,
        _point: CirclePoint<SecureField>,
    ) -> TreeVec<ColumnVec<Vec<CirclePoint<SecureField>>>> {
        TreeVec::new(vec![])
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
        todo!()
    }

    fn lookup_config(&self) -> Vec<LookupConfig> {
        vec![LookupConfig::new(
            LookupVariant::Table,
            LookupEvaluator::LogUp {
                numerator: ColumnEvaluator::Univariate(Box::new(XorTableNumeratorsEvaluator)),
                denominator: ColumnEvaluator::Multilinear(Box::new(XorTableDenominatorsEvaluator)),
            },
        )]
    }

    fn n_interaction_phases(&self) -> u32 {
        1
    }
}

struct XorTableNumeratorsEvaluator;

impl TraceExprPolynomial for XorTableNumeratorsEvaluator {
    fn eval(
        &self,
        _interaction_elements: InteractionElements,
        mask: &ColumnVec<Vec<SecureField>>,
    ) -> SecureField {
        mask[0][0]
    }
}

struct XorTableDenominatorsEvaluator;

impl MultilinearPolynomial for XorTableDenominatorsEvaluator {
    fn eval(
        &self,
        interaction_elements: InteractionElements,
        point: &[SecureField],
    ) -> SecureField {
        let alpha = interaction_elements[XOR_ALPHA_ID];
        let z = interaction_elements[XOR_Z_ID];

        match point {
            // `li`: left operand bit `i`, `ri`: right operand bit `i`
            &[ref _unused @ .., l0, l1, l2, l3, l4, l5, l6, l7, r0, r1, r2, r3, r4, r5, r6, r7] => {
                let lhs_assignment = [l0, l1, l2, l3, l4, l5, l6, l7];
                let rhs_assignment = [r0, r1, r2, r3, r4, r5, r6, r7];

                let xor_assignment: [SecureField; 8] = array::from_fn(|i| {
                    let a = lhs_assignment[i];
                    let b = rhs_assignment[i];

                    // Note `a ^ b = 1 - a * b - (1 - a)(1 - b)` for all `a, b` in `{0, 1}`.
                    SecureField::one() - a * b - (SecureField::one() - a) * (SecureField::one() - b)
                });

                let two = BaseField::from(2).into();
                let lhs = horner_eval(&lhs_assignment, two);
                let rhs = horner_eval(&rhs_assignment, two);
                let xor = horner_eval(&xor_assignment, two);

                z - lhs - rhs * alpha - xor * alpha.square()
            }
            // TODO: Add error handing to polynomial types.
            _ => panic!(),
        }
    }
}
