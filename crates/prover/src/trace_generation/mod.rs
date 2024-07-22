mod prove;
pub mod registry;

use downcast_rs::{impl_downcast, Downcast};
pub use prove::{commit_and_prove, commit_and_verify};
use registry::ComponentGenerationRegistry;

use crate::core::air::{AirProver, Component};
use crate::core::backend::Backend;
use crate::core::channel::Channel;
use crate::core::fields::m31::BaseField;
use crate::core::poly::circle::CircleEvaluation;
use crate::core::poly::BitReversedOrder;
use crate::core::{ColumnVec, InteractionElements};

pub const BASE_TRACE: usize = 0;
pub const INTERACTION_TRACE: usize = 1;

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

    /// Allocates and returns the interaction trace of the component.
    fn write_interaction_trace(
        &self,
        trace: &ColumnVec<&CircleEvaluation<B, BaseField, BitReversedOrder>>,
        elements: &InteractionElements,
    ) -> ColumnVec<CircleEvaluation<B, BaseField, BitReversedOrder>>;

    fn component(&self) -> Self::Component;
}

pub trait AirTraceVerifier {
    fn interaction_elements(&self, channel: &mut impl Channel) -> InteractionElements;
}

pub trait AirTraceGenerator<B: Backend>: AirTraceVerifier {
    fn composition_log_degree_bound(&self) -> u32;

    // TODO(AlonH): Remove default implementation once all the components are implemented.
    fn write_trace(&mut self) -> Vec<CircleEvaluation<B, BaseField, BitReversedOrder>> {
        vec![]
    }

    fn interact(
        &self,
        trace: &ColumnVec<CircleEvaluation<B, BaseField, BitReversedOrder>>,
        elements: &InteractionElements,
    ) -> Vec<CircleEvaluation<B, BaseField, BitReversedOrder>>;

    fn to_air_prover(&self) -> impl AirProver<B>;
}
