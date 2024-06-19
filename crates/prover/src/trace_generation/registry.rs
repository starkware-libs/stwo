use std::collections::HashMap;

use super::{ComponentGen, ExtendableInputs, InputsType, TraceGenerator};
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

    pub fn register_component_with_input<B: Backend, TG: TraceGenerator<B> + ComponentGen>(
        &mut self,
        component_id: &str,
        component: TG,
    ) {
        self.inputs.insert(
            component_id.to_string(),
            Box::<<TG as TraceGenerator<B>>::ComponentInputs>::default(),
        );
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

    /// Add inputs for the trace generation of the component.
    /// This function should be called from the caller components before calling `write_trace` of
    /// the given component.
    pub fn add_inputs<T: ExtendableInputs>(&mut self, component_id: &str, inputs: &T) {
        self.inputs
            .get_mut(component_id)
            .unwrap_or_else(|| panic!("No inputs found for Component name {}", component_id))
            .downcast_mut::<T>()
            .unwrap()
            .extend(inputs)
    }

    pub fn get_inputs<T: InputsType>(&self, component_id: &str) -> &T {
        self.inputs
            .get(component_id)
            .unwrap_or_else(|| panic!("No inputs found for Component name {}", component_id))
            .downcast_ref()
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
    struct ComponentA {}

    #[derive(Default)]
    struct ComponentAInputs(Vec<u32>);
    impl InputsType for ComponentAInputs {}

    impl ExtendableInputs for ComponentAInputs {
        fn extend(&mut self, other: &Self) {
            self.0.extend(other.0.iter().cloned());
        }
    }

    impl ComponentGen for ComponentA {}

    impl TraceGenerator<CpuBackend> for ComponentA {
        type ComponentInputs = ComponentAInputs;

        fn write_trace(
            _component_id: &str,
            _registry: &mut ComponentRegistry,
        ) -> ColumnVec<CircleEvaluation<CpuBackend, BaseField, BitReversedOrder>> {
            unimplemented!("TestTraceGenerator::write_trace")
        }
    }

    #[test]
    fn test_component_registry() {
        let mut registry = ComponentRegistry::default();
        let component = ComponentA {};
        let component_id = "test";
        let inputs = ComponentAInputs(vec![1]);

        registry.register_component_with_input(component_id, component);
        registry.add_inputs(component_id, &inputs);

        assert_eq!(registry.get_inputs::<ComponentAInputs>("test").0, vec![1]);
    }
}
