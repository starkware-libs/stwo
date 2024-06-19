use std::collections::HashMap;

use super::{ComponentGen, InputsType, TraceGenerator};
use crate::core::backend::Backend;

#[derive(Default)]
pub struct ComponentRegistry {
    components: HashMap<String, Box<dyn ComponentGen>>,
    inputs: HashMap<String, Box<dyn InputsType>>,
}

impl ComponentRegistry {
    pub fn register_component(&mut self, component_id: &str, component: impl ComponentGen) {
        self.components
            .insert(component_id.to_string(), Box::new(component));
    }

    pub fn register_component_and_init_input<B: Backend>(
        &mut self,
        component_id: &str,
        component: impl ComponentGen + TraceGenerator<B>,
    ) {
        self.inputs
            .insert(component_id.to_string(), component.initialize_inputs());
        self.components
            .insert(component_id.to_string(), Box::new(component));
    }

    pub fn get_component<T: ComponentGen>(&self, component_id: &str) -> &T {
        self.components
            .get(component_id)
            .unwrap_or_else(|| panic!("Component name {} not found.", component_id))
            .downcast_ref()
            .unwrap()
    }

    pub fn get_component_mut<T: ComponentGen>(&mut self, component_id: &str) -> &mut T {
        self.components
            .get_mut(component_id)
            .unwrap_or_else(|| panic!("Component name {} not found.", component_id))
            .downcast_mut()
            .unwrap()
    }

    pub fn get_inputs<T: InputsType>(&self, component_id: &str) -> &T {
        self.inputs
            .get(component_id)
            .unwrap_or_else(|| panic!("No inputs found for Component name {}", component_id))
            .downcast_ref()
            .unwrap()
    }

    pub fn get_inputs_mut<T: InputsType>(&mut self, component_id: &str) -> &mut T {
        self.inputs
            .get_mut(component_id)
            .unwrap_or_else(|| panic!("No inputs found for Component name {}", component_id))
            .downcast_mut()
            .unwrap()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::backend::CpuBackend;
    use crate::core::fields::m31::BaseField;
    use crate::core::poly::circle::CircleEvaluation;
    use crate::core::poly::BitReversedOrder;
    use crate::core::ColumnVec;
    use crate::trace_generation::TraceGenerator;

    #[derive(Default)]
    struct ComponentA {
        inputs: Vec<u32>,
    }

    impl ComponentGen for ComponentA {}

    impl TraceGenerator<CpuBackend> for ComponentA {
        type ComponentInputs = u32;

        fn add_inputs(
            _component_id: &str,
            _registry: &mut ComponentRegistry,
            _inputs: &Self::ComponentInputs,
        ) {
            unimplemented!("TestTraceGenerator::add_inputs")
        }

        fn write_trace(
            _component_id: &str,
            _registry: &mut ComponentRegistry,
        ) -> ColumnVec<CircleEvaluation<CpuBackend, BaseField, BitReversedOrder>> {
            unimplemented!("TestTraceGenerator::write_trace")
        }

        fn initialize_inputs(&self) -> Box<dyn InputsType> {
            unimplemented!("TestTraceGenerator::initialize_inputs")
        }
    }

    #[test]
    fn test_component_registry() {
        let mut registry = ComponentRegistry::default();
        let component = ComponentA { inputs: vec![1] };

        registry.register_component("test", component);

        assert_eq!(registry.get_component::<ComponentA>("test").inputs, vec![1]);
    }
}
