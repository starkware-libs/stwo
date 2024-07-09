use std::collections::HashMap;

use super::ComponentGen;

#[derive(Default)]
pub struct ComponentGenerationRegistry {
    components: HashMap<String, Box<dyn ComponentGen>>,
}

impl ComponentGenerationRegistry {
    pub fn register(&mut self, component_id: &str, component_generator: impl ComponentGen) {
        self.components
            .insert(component_id.to_string(), Box::new(component_generator));
    }

    pub fn get_generator<T: ComponentGen>(&self, component_id: &str) -> &T {
        self.components
            .get(component_id)
            .unwrap_or_else(|| panic!("Component ID: {} not found.", component_id))
            .downcast_ref()
            .unwrap()
    }

    pub fn get_generator_mut<T: ComponentGen>(&mut self, component_id: &str) -> &mut T {
        self.components
            .get_mut(component_id)
            .unwrap_or_else(|| panic!("Component ID: {} not found.", component_id))
            .downcast_mut()
            .unwrap()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::air::accumulation::PointEvaluationAccumulator;
    use crate::core::air::Component;
    use crate::core::backend::simd::m31::{PackedM31, N_LANES};
    use crate::core::backend::simd::SimdBackend;
    use crate::core::backend::CpuBackend;
    use crate::core::circle::CirclePoint;
    use crate::core::fields::m31::{BaseField, M31};
    use crate::core::fields::qm31::SecureField;
    use crate::core::pcs::TreeVec;
    use crate::core::poly::circle::CircleEvaluation;
    use crate::core::poly::BitReversedOrder;
    use crate::core::{ColumnVec, InteractionElements, LookupValues};
    use crate::m31;
    use crate::trace_generation::ComponentTraceGenerator;
    pub struct ComponentA {
        pub n_instances: usize,
    }

    impl Component for ComponentA {
        fn n_constraints(&self) -> usize {
            todo!()
        }

        fn max_constraint_log_degree_bound(&self) -> u32 {
            todo!()
        }

        fn n_interaction_phases(&self) -> u32 {
            todo!()
        }

        fn trace_log_degree_bounds(&self) -> TreeVec<ColumnVec<u32>> {
            todo!()
        }

        fn mask_points(
            &self,
            _point: CirclePoint<SecureField>,
        ) -> TreeVec<ColumnVec<Vec<CirclePoint<SecureField>>>> {
            todo!()
        }

        fn evaluate_constraint_quotients_at_point(
            &self,
            _point: CirclePoint<SecureField>,
            _mask: &TreeVec<Vec<Vec<SecureField>>>,
            _evaluation_accumulator: &mut PointEvaluationAccumulator,
            _interaction_elements: &InteractionElements,
            _lookup_values: &LookupValues,
        ) {
            todo!()
        }
    }

    type ComponentACpuInputs = Vec<(M31, M31)>;
    struct ComponentACpuTraceGenerator {
        inputs: ComponentACpuInputs,
    }
    impl ComponentGen for ComponentACpuTraceGenerator {}

    impl ComponentTraceGenerator<CpuBackend> for ComponentACpuTraceGenerator {
        type Component = ComponentA;
        type Inputs = ComponentACpuInputs;

        fn write_trace(
            _component_id: &str,
            _registry: &mut ComponentGenerationRegistry,
        ) -> ColumnVec<CircleEvaluation<CpuBackend, BaseField, BitReversedOrder>> {
            unimplemented!("TestTraceGenerator::write_trace")
        }

        fn add_inputs(&mut self, inputs: &ComponentACpuInputs) {
            self.inputs.extend(inputs)
        }

        fn component(&self) -> ComponentA {
            ComponentA {
                n_instances: self.inputs.len(),
            }
        }

        fn write_interaction_trace(
            &self,
            _trace: &ColumnVec<&CircleEvaluation<CpuBackend, BaseField, BitReversedOrder>>,
            _elements: &InteractionElements,
        ) -> ColumnVec<CircleEvaluation<CpuBackend, BaseField, BitReversedOrder>> {
            unimplemented!("TestTraceGenerator::write_interaction_trace")
        }
    }

    type ComponentASimdInputs = Vec<(PackedM31, PackedM31)>;
    struct ComponentASimdTraceGenerator {
        inputs: ComponentASimdInputs,
    }
    impl ComponentGen for ComponentASimdTraceGenerator {}

    impl ComponentTraceGenerator<SimdBackend> for ComponentASimdTraceGenerator {
        type Component = ComponentA;
        type Inputs = ComponentASimdInputs;

        fn write_trace(
            _component_id: &str,
            _registry: &mut ComponentGenerationRegistry,
        ) -> ColumnVec<CircleEvaluation<SimdBackend, BaseField, BitReversedOrder>> {
            unimplemented!("TestTraceGenerator::write_trace")
        }

        fn add_inputs(&mut self, inputs: &ComponentASimdInputs) {
            self.inputs.extend(inputs)
        }

        fn component(&self) -> ComponentA {
            ComponentA {
                n_instances: self.inputs.len() * N_LANES,
            }
        }

        fn write_interaction_trace(
            &self,
            _trace: &ColumnVec<&CircleEvaluation<SimdBackend, BaseField, BitReversedOrder>>,
            _elements: &InteractionElements,
        ) -> ColumnVec<CircleEvaluation<SimdBackend, BaseField, BitReversedOrder>> {
            unimplemented!("TestTraceGenerator::write_interaction_trace")
        }
    }

    #[test]
    fn test_component_registry() {
        let mut registry = ComponentGenerationRegistry::default();
        let component_id = "componentA::0";

        let component_a_cpu_trace_generator = ComponentACpuTraceGenerator { inputs: vec![] };
        registry.register(component_id, component_a_cpu_trace_generator);
        let cpu_inputs = vec![(m31!(1), m31!(1)), (m31!(2), m31!(2))];

        registry
            .get_generator_mut::<ComponentACpuTraceGenerator>(component_id)
            .add_inputs(&cpu_inputs);

        assert_eq!(
            registry
                .get_generator_mut::<ComponentACpuTraceGenerator>(component_id)
                .inputs,
            cpu_inputs
        );
    }
}
