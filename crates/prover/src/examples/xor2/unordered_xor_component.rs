use crate::core::air::accumulation::PointEvaluationAccumulator;
use crate::core::air::{
    ColumnEvaluator, Component, ComponentProver, ConstantPolynomial, LookupConfig, LookupEvaluator,
    LookupVariant, TraceExprPolynomial,
};
use crate::core::backend::CpuBackend;
use crate::core::circle::CirclePoint;
use crate::core::fields::qm31::SecureField;
use crate::core::fields::FieldExpOps;
use crate::core::pcs::TreeVec;
use crate::core::{ColumnVec, InteractionElements, LookupValues};
use crate::examples::xor2::xor_table_component::{XOR_ALPHA_ID, XOR_Z_ID};

/// Component full of random 8-bit XOR operations.
pub struct UnorderedXorComponent;

impl Component for UnorderedXorComponent {
    fn n_constraints(&self) -> usize {
        0
    }

    fn max_constraint_log_degree_bound(&self) -> u32 {
        u8::BITS + u8::BITS
    }

    /// Returns the degree bounds of each trace column.
    fn trace_log_degree_bounds(&self) -> TreeVec<ColumnVec<u32>> {
        TreeVec::new(vec![vec![u8::BITS + u8::BITS; 3]])
    }

    fn mask_points(
        &self,
        point: CirclePoint<SecureField>,
    ) -> TreeVec<ColumnVec<Vec<CirclePoint<SecureField>>>> {
        TreeVec::new(vec![vec![vec![point], vec![point], vec![point]]])
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

    fn lookup_config(&self) -> Vec<LookupConfig> {
        vec![LookupConfig::new(
            LookupVariant::Reference,
            LookupEvaluator::LogUp {
                numerator: ColumnEvaluator::Multilinear(Box::new(ConstantPolynomial::one())),
                denominator: ColumnEvaluator::Univariate(Box::new(
                    UnorderedXorDenominatorEvaluator,
                )),
            },
        )]
    }

    // fn evaluate_lookup_instances_at_point

    fn n_interaction_phases(&self) -> u32 {
        1
    }
}

impl ComponentProver<CpuBackend> for UnorderedXorComponent {
    fn evaluate_constraint_quotients_on_domain(
        &self,
        trace: &crate::core::air::ComponentTrace<'_, CpuBackend>,
        evaluation_accumulator: &mut crate::core::air::accumulation::DomainEvaluationAccumulator<
            CpuBackend,
        >,
        interaction_elements: &InteractionElements,
        lookup_values: &LookupValues,
    ) {
        todo!()
    }

    fn lookup_values(&self, trace: &ComponentTrace<'_, CpuBackend>) -> LookupValues {
        let domain = CanonicCoset::new(self.log_column_size());
        let trace_poly = &trace.polys[BASE_TRACE];
        let values = BTreeMap::from_iter([
            (
                LOOKUP_VALUE_0_ID.to_string(),
                trace_poly[0]
                    .eval_at_point(domain.at(0).into_ef())
                    .try_into()
                    .unwrap(),
            ),
            (
                LOOKUP_VALUE_1_ID.to_string(),
                trace_poly[1]
                    .eval_at_point(domain.at(0).into_ef())
                    .try_into()
                    .unwrap(),
            ),
            (
                LOOKUP_VALUE_N_MINUS_2_ID.to_string(),
                trace_poly[self.n_columns() - 2]
                    .eval_at_point(domain.at(domain.size()).into_ef())
                    .try_into()
                    .unwrap(),
            ),
            (
                LOOKUP_VALUE_N_MINUS_1_ID.to_string(),
                trace_poly[self.n_columns() - 1]
                    .eval_at_point(domain.at(domain.size()).into_ef())
                    .try_into()
                    .unwrap(),
            ),
        ]);
        LookupValues::new(values)
    }

    // fn build_lookup_instances(
    //     &self,
    //     trace: &ColumnVec<CircleEvaluation<B, BaseField, BitReversedOrder>>,
    //     interaction_elements: &InteractionElements,
    // ) Vec<Lookup<CpuBackend>> {
    //     todo!()
    // }
}

struct UnorderedXorDenominatorEvaluator;

impl TraceExprPolynomial for UnorderedXorDenominatorEvaluator {
    fn eval(
        &self,
        interaction_elements: InteractionElements,
        mask: &ColumnVec<Vec<SecureField>>,
    ) -> SecureField {
        let alpha = interaction_elements[XOR_ALPHA_ID];
        let z = interaction_elements[XOR_Z_ID];

        let lhs = mask[0][0];
        let rhs = mask[1][0];
        let res = mask[2][0];

        z - lhs - alpha * rhs - alpha.square() * res
    }
}
