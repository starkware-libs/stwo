use std::collections::HashMap;

use downcast_rs::{impl_downcast, Downcast};

use crate::core::backend::Backend;
use crate::core::fields::m31::BaseField;
use crate::core::poly::circle::CircleEvaluation;
use crate::core::poly::BitReversedOrder;
use crate::core::ColumnVec;

pub trait ComponentGen: Downcast {}
impl_downcast!(ComponentGen);

#[derive(Default)]
pub struct ComponentRegistry {
    components: HashMap<String, Box<dyn ComponentGen>>,
}

impl ComponentRegistry {
    pub fn register_component(&mut self, component_id: String, component: Box<dyn ComponentGen>) {
        self.components.insert(component_id, component);
    }

    pub fn get_component<T: ComponentGen>(&self, component_id: &str) -> &T {
        self.components
            .get(component_id)
            .unwrap()
            .downcast_ref()
            .unwrap()
    }

    pub fn get_component_mut<T: ComponentGen>(&mut self, component_id: &str) -> &mut T {
        self.components
            .get_mut(component_id)
            .unwrap()
            .downcast_mut()
            .unwrap()
    }
}

// A trait to generate a a trace.
// Generates the trace given a list of inputs collects inputs for subcomponents.
trait TraceGenerator<B: Backend> {
    type ComponentInputs;

    /// Add inputs for the trace generation of the component.
    /// This function should be called from the caller components before calling `write_trace` of
    /// this component.
    fn add_inputs(&mut self, inputs: &Self::ComponentInputs);

    /// Allocates and returns the trace of the component and updates the
    /// subcomponents with the corresponding inputs.
    /// Should be called only after all the inputs are available.
    // TODO(ShaharS): change `component_id` to a struct that contains the id and the component name.
    fn write_trace(
        component_id: &str,
        registry: &mut ComponentRegistry,
    ) -> ColumnVec<CircleEvaluation<B, BaseField, BitReversedOrder>>;
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::backend::CpuBackend;

    #[derive(Default)]
    struct ComponentA {
        inputs: Vec<u32>,
    }

    impl ComponentGen for ComponentA {}

    impl TraceGenerator<CpuBackend> for ComponentA {
        type ComponentInputs = u32;

        fn add_inputs(&mut self, _inputs: &Self::ComponentInputs) {
            unimplemented!("TestTraceGenerator::add_inputs")
        }

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
        let component = Box::new(ComponentA { inputs: vec![1] });
        registry.register_component("test".to_string(), component);
        let component = registry.get_component::<ComponentA>("test");
        assert_eq!(component.inputs, vec![1]);
    }
}
