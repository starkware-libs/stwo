use std::collections::HashMap;

use super::{ComponentGen, TraceGenerator};
use crate::core::backend::Backend;

#[derive(Default)]
pub struct ComponentRegistry {
    components: HashMap<String, Box<dyn ComponentGen>>,
}

impl ComponentRegistry {
    pub fn register_component(&mut self, component_id: &str, component: impl ComponentGen) {
        self.components
            .insert(component_id.to_string(), Box::new(component));
    }

    pub fn add_inputs<B: Backend, T: ComponentGen + TraceGenerator<B>>(
        &mut self,
        component_id: &str,
        inputs: &T::ComponentInputs,
    ) {
        self.get_component_mut::<T>(component_id).add_inputs(inputs);
    }

    pub fn get_component<T: ComponentGen>(&self, component_id: &str) -> &T {
        self.components
            .get(component_id)
            .unwrap_or_else(|| panic!("Component name {} not found.", component_id))
            .downcast_ref()
            .unwrap()
    }

    fn get_component_mut<T: ComponentGen>(&mut self, component_id: &str) -> &mut T {
        self.components
            .get_mut(component_id)
            .unwrap_or_else(|| panic!("Component name {} not found.", component_id))
            .downcast_mut()
            .unwrap()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::air::accumulation::PointEvaluationAccumulator;
    use crate::core::air::Component;
    use crate::core::backend::simd::m31::PackedM31;
    use crate::core::backend::simd::SimdBackend;
    use crate::core::backend::CpuBackend;
    use crate::core::circle::CirclePoint;
    use crate::core::fields::m31::{BaseField, M31};
    use crate::core::fields::qm31::SecureField;
    use crate::core::pcs::TreeVec;
    use crate::core::poly::circle::CircleEvaluation;
    use crate::core::poly::BitReversedOrder;
    use crate::core::{ColumnVec, InteractionElements};
    use crate::m31;
    use crate::trace_generation::TraceGenerator;
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

        fn interaction_element_ids(&self) -> Vec<String> {
            todo!()
        }

        fn evaluate_constraint_quotients_at_point(
            &self,
            _point: CirclePoint<SecureField>,
            _mask: &ColumnVec<Vec<SecureField>>,
            _evaluation_accumulator: &mut PointEvaluationAccumulator,
            _interaction_elements: &InteractionElements,
        ) {
            todo!()
        }
    }

    type ComponentACpuInput = Vec<(M31, M31)>;
    struct ComponentACpuTraceGenerator {
        inputs: ComponentACpuInput,
    }
    impl ComponentGen for ComponentACpuTraceGenerator {}

    impl TraceGenerator<CpuBackend> for ComponentACpuTraceGenerator {
        type Component = ComponentA;
        type ComponentInputs = ComponentACpuInput;

        fn write_trace(
            &mut self,
            _registry: &mut ComponentRegistry,
        ) -> ColumnVec<CircleEvaluation<CpuBackend, BaseField, BitReversedOrder>> {
            unimplemented!("TestTraceGenerator::write_trace")
        }

        fn add_inputs(&mut self, inputs: &ComponentACpuInput) {
            self.inputs.extend(inputs)
        }

        fn component(&self) -> ComponentA {
            ComponentA {
                n_instances: self.inputs.len(),
            }
        }
    }

    type ComponentASimdInput = Vec<(PackedM31, PackedM31)>;
    struct ComponentASimdTraceGenerator {
        inputs: ComponentASimdInput,
    }
    impl ComponentGen for ComponentASimdTraceGenerator {}

    impl TraceGenerator<SimdBackend> for ComponentASimdTraceGenerator {
        type Component = ComponentA;
        type ComponentInputs = ComponentASimdInput;

        fn write_trace(
            &mut self,
            _registry: &mut ComponentRegistry,
        ) -> ColumnVec<CircleEvaluation<SimdBackend, BaseField, BitReversedOrder>> {
            unimplemented!("TestTraceGenerator::write_trace")
        }

        fn add_inputs(&mut self, inputs: &ComponentASimdInput) {
            self.inputs.extend(inputs)
        }

        fn component(&self) -> ComponentA {
            ComponentA {
                n_instances: self.inputs.len() * 16,
            }
        }
    }

    #[test]
    fn test_component_registry() {
        let mut registry = ComponentRegistry::default();
        let cpu_generator_id = "cpu";
        let simd_generator_id = "simd";

        let cpu_trace_generator = ComponentACpuTraceGenerator { inputs: vec![] };
        let simd_trace_generator = ComponentASimdTraceGenerator { inputs: vec![] };
        registry.register_component(cpu_generator_id, cpu_trace_generator);
        registry.register_component(simd_generator_id, simd_trace_generator);

        let cpu_inputs = vec![(m31!(1), m31!(1)), (m31!(2), m31!(2))];
        let simd_inputs = vec![(PackedM31::broadcast(m31!(1)), PackedM31::broadcast(m31!(1)))];

        registry
            .get_component_mut::<ComponentACpuTraceGenerator>(cpu_generator_id)
            .add_inputs(&cpu_inputs);
        registry
            .get_component_mut::<ComponentASimdTraceGenerator>(simd_generator_id)
            .add_inputs(&simd_inputs);
    }
}
