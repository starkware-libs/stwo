pub mod registry;

use downcast_rs::{impl_downcast, Downcast};
use registry::ComponentRegistry;

use crate::core::backend::Backend;
use crate::core::fields::m31::BaseField;
use crate::core::poly::circle::CircleEvaluation;
use crate::core::poly::BitReversedOrder;
use crate::core::ColumnVec;

pub trait ComponentGen: Downcast {}
impl_downcast!(ComponentGen);

/// Input types are spe
pub trait InputsType: Downcast {}
impl_downcast!(InputsType);

pub trait ExtendableInputs: InputsType + Default {
    fn extend(&mut self, other: &Self);
}

// A trait to generate a a trace.
// Generates the trace given a list of inputs collects inputs for subcomponents.
pub trait TraceGenerator<B: Backend> {
    type ComponentInputs: ExtendableInputs;

    /// Allocates and returns the trace of the component and updates the
    /// subcomponents with the corresponding inputs.
    /// Should be called only after all the inputs are available.
    // TODO(ShaharS): change `component_id` to a struct that contains the id and the component name.
    fn write_trace(
        component_id: &str,
        registry: &mut ComponentRegistry,
    ) -> ColumnVec<CircleEvaluation<B, BaseField, BitReversedOrder>>;
}
