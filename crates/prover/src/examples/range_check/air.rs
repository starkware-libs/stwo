use super::component::{RangeCheckComponent, RangeCheckInput, RangeCheckTraceGenerator};
use crate::core::air::{Air, AirProver, Component, ComponentProver};
use crate::core::backend::CpuBackend;
use crate::core::channel::Blake2sChannel;
use crate::core::fields::m31::BaseField;
use crate::core::poly::circle::CircleEvaluation;
use crate::core::poly::BitReversedOrder;
use crate::core::prover::VerificationError;
use crate::core::{ColumnVec, InteractionElements, LookupValues};
use crate::trace_generation::registry::ComponentGenerationRegistry;
use crate::trace_generation::{AirTraceGenerator, AirTraceVerifier, ComponentTraceGenerator};

pub struct RangeCheckAirGenerator {
    pub registry: ComponentGenerationRegistry,
}

impl RangeCheckAirGenerator {
    pub fn new(inputs: &RangeCheckInput) -> Self {
        let mut component_generator = RangeCheckTraceGenerator::new();
        component_generator.add_inputs(inputs);
        let mut registry = ComponentGenerationRegistry::default();
        registry.register("range_check", component_generator);
        Self { registry }
    }
}

impl AirTraceVerifier for RangeCheckAirGenerator {
    fn interaction_elements(&self, _channel: &mut Blake2sChannel) -> InteractionElements {
        InteractionElements::default()
    }
}

impl AirTraceGenerator<CpuBackend> for RangeCheckAirGenerator {
    fn write_trace(&mut self) -> Vec<CircleEvaluation<CpuBackend, BaseField, BitReversedOrder>> {
        RangeCheckTraceGenerator::write_trace("range_check", &mut self.registry)
    }

    fn interact(
        &self,
        _trace: &ColumnVec<CircleEvaluation<CpuBackend, BaseField, BitReversedOrder>>,
        _elements: &InteractionElements,
    ) -> Vec<CircleEvaluation<CpuBackend, BaseField, BitReversedOrder>> {
        vec![]
    }

    fn to_air_prover(&self) -> impl AirProver<CpuBackend> {
        let component_generator = self
            .registry
            .get_generator::<RangeCheckTraceGenerator>("range_check");
        RangeCheckAir {
            component: component_generator.component(),
        }
    }

    fn composition_log_degree_bound(&self) -> u32 {
        let component_generator = self
            .registry
            .get_generator::<RangeCheckTraceGenerator>("range_check");
        assert!(
            component_generator.inputs_set(),
            "range_check input not set."
        );
        component_generator
            .component()
            .max_constraint_log_degree_bound()
    }
}

#[derive(Clone)]
pub struct RangeCheckAir {
    pub component: RangeCheckComponent,
}

impl RangeCheckAir {
    pub fn new(component: RangeCheckComponent) -> Self {
        Self { component }
    }
}

impl Air for RangeCheckAir {
    fn components(&self) -> Vec<&dyn Component> {
        vec![&self.component]
    }

    fn verify_lookups(&self, _lookup_values: &LookupValues) -> Result<(), VerificationError> {
        Ok(())
    }
}

impl AirTraceVerifier for RangeCheckAir {
    fn interaction_elements(&self, _channel: &mut Blake2sChannel) -> InteractionElements {
        InteractionElements::default()
    }
}

impl AirTraceGenerator<CpuBackend> for RangeCheckAir {
    fn interact(
        &self,
        _trace: &ColumnVec<CircleEvaluation<CpuBackend, BaseField, BitReversedOrder>>,
        _elements: &InteractionElements,
    ) -> Vec<CircleEvaluation<CpuBackend, BaseField, BitReversedOrder>> {
        vec![]
    }

    fn to_air_prover(&self) -> impl AirProver<CpuBackend> {
        self.clone()
    }

    fn composition_log_degree_bound(&self) -> u32 {
        self.component.max_constraint_log_degree_bound()
    }
}

impl AirProver<CpuBackend> for RangeCheckAir {
    fn prover_components(&self) -> Vec<&dyn ComponentProver<CpuBackend>> {
        vec![&self.component]
    }
}
