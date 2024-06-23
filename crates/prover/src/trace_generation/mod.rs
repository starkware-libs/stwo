pub mod registry;

use downcast_rs::{impl_downcast, Downcast};
use registry::ComponentGenerationRegistry;

use crate::core::air::{AirProver, Component};
use crate::core::backend::Backend;
use crate::core::channel::Blake2sChannel;
use crate::core::fields::m31::BaseField;
use crate::core::poly::circle::CircleEvaluation;
use crate::core::poly::BitReversedOrder;
use crate::core::{ColumnVec, InteractionElements};

pub trait ComponentGen: Downcast {}
impl_downcast!(ComponentGen);

// A trait to generate a a trace.
// Generates the trace given a list of inputs collects inputs for subcomponents.
pub trait ComponentTraceGenerator<B: Backend> {
    type Component: Component;
    type Inputs;

    /// Add inputs for the trace generation of the component.
    /// This function should be called from the caller components before calling `write_trace` of
    /// this component.
    fn add_inputs(&mut self, inputs: &Self::Inputs);

    /// Allocates and returns the trace of the component and updates the
    /// subcomponents with the corresponding inputs.
    /// Should be called only after all the inputs are available.
    // TODO(ShaharS): change `component_id` to a struct that contains the id and the component name.
    fn write_trace(
        component_id: &str,
        registry: &mut ComponentGenerationRegistry,
    ) -> ColumnVec<CircleEvaluation<B, BaseField, BitReversedOrder>>;

    fn write_interaction_trace(
        &self,
        trace: &ColumnVec<&CircleEvaluation<B, BaseField, BitReversedOrder>>,
        elements: &InteractionElements,
    ) -> ColumnVec<CircleEvaluation<B, BaseField, BitReversedOrder>>;

    fn component(self) -> Self::Component;
}

pub trait AirTraceVerifier {
    fn interaction_elements(&self, channel: &mut Blake2sChannel) -> InteractionElements;
}

pub trait AirTraceGenerator<B: Backend>: AirTraceVerifier {
    fn interact(
        &self,
        trace: &ColumnVec<CircleEvaluation<B, BaseField, BitReversedOrder>>,
        elements: &InteractionElements,
    ) -> Vec<CircleEvaluation<B, BaseField, BitReversedOrder>>;

    fn to_air_prover(&self) -> &impl AirProver<B>;
}
