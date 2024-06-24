use itertools::{zip_eq, Itertools};

use super::component::{FibonacciComponent, FibonacciInput, FibonacciTraceGenerator};
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

pub struct FibonacciAirGenerator {
    pub registry: ComponentGenerationRegistry,
}

impl FibonacciAirGenerator {
    pub fn new(inputs: &FibonacciInput) -> Self {
        let mut component_generator = FibonacciTraceGenerator::new();
        component_generator.add_inputs(inputs);
        let mut registry = ComponentGenerationRegistry::default();
        registry.register("fibonacci", component_generator);
        Self { registry }
    }
}

impl AirTraceVerifier for FibonacciAirGenerator {
    fn interaction_elements(&self, _channel: &mut Blake2sChannel) -> InteractionElements {
        InteractionElements::default()
    }
}

impl AirTraceGenerator<CpuBackend> for FibonacciAirGenerator {
    fn write_trace(&mut self) -> Vec<CircleEvaluation<CpuBackend, BaseField, BitReversedOrder>> {
        FibonacciTraceGenerator::write_trace("fibonacci", &mut self.registry)
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
            .get_generator::<FibonacciTraceGenerator>("fibonacci");
        FibonacciAir {
            component: component_generator.component(),
        }
    }

    fn composition_log_degree_bound(&self) -> u32 {
        let component_generator = self
            .registry
            .get_generator::<FibonacciTraceGenerator>("fibonacci");
        assert!(component_generator.inputs_set(), "Fibonacci input not set.");
        component_generator
            .component()
            .max_constraint_log_degree_bound()
    }
}

#[derive(Clone)]
pub struct FibonacciAir {
    pub component: FibonacciComponent,
}

impl FibonacciAir {
    pub fn new(component: FibonacciComponent) -> Self {
        Self { component }
    }
}

impl Air for FibonacciAir {
    fn components(&self) -> Vec<&dyn Component> {
        vec![&self.component]
    }

    fn verify_lookups(&self, _lookup_values: &LookupValues) -> Result<(), VerificationError> {
        Ok(())
    }
}

impl AirTraceVerifier for FibonacciAir {
    fn interaction_elements(&self, _channel: &mut Blake2sChannel) -> InteractionElements {
        InteractionElements::default()
    }
}

impl AirTraceGenerator<CpuBackend> for FibonacciAir {
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

impl AirProver<CpuBackend> for FibonacciAir {
    fn prover_components(&self) -> Vec<&dyn ComponentProver<CpuBackend>> {
        vec![&self.component]
    }
}

#[derive(Clone)]
pub struct MultiFibonacciAir {
    pub components: Vec<FibonacciComponent>,
}

impl MultiFibonacciAir {
    pub fn new(log_sizes: &[u32], claim: &[BaseField]) -> Self {
        let mut components = Vec::new();
        for (log_size, claim) in zip_eq(log_sizes.iter(), claim.iter()) {
            components.push(FibonacciComponent::new(*log_size, *claim));
        }
        Self { components }
    }
}

impl Air for MultiFibonacciAir {
    fn components(&self) -> Vec<&dyn Component> {
        self.components
            .iter()
            .map(|c| c as &dyn Component)
            .collect_vec()
    }

    fn verify_lookups(&self, _lookup_values: &LookupValues) -> Result<(), VerificationError> {
        Ok(())
    }
}

impl AirTraceVerifier for MultiFibonacciAir {
    fn interaction_elements(&self, _channel: &mut Blake2sChannel) -> InteractionElements {
        InteractionElements::default()
    }
}

impl AirTraceGenerator<CpuBackend> for MultiFibonacciAir {
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
        self.components
            .iter()
            .map(|component| component.max_constraint_log_degree_bound())
            .max()
            .unwrap()
    }
}

impl AirProver<CpuBackend> for MultiFibonacciAir {
    fn prover_components(&self) -> Vec<&dyn ComponentProver<CpuBackend>> {
        self.components
            .iter()
            .map(|c| c as &dyn ComponentProver<CpuBackend>)
            .collect_vec()
    }
}
