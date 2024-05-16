use itertools::{zip_eq, Itertools};

use super::component::FibonacciComponent;
use crate::core::air::{Air, AirProver, Component, ComponentProver};
use crate::core::backend::CpuBackend;
use crate::core::fields::m31::BaseField;

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
}
impl AirProver<CpuBackend> for FibonacciAir {
    fn prover_components(&self) -> Vec<&dyn ComponentProver<CpuBackend>> {
        vec![&self.component]
    }
}

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
}
impl AirProver<CpuBackend> for MultiFibonacciAir {
    fn prover_components(&self) -> Vec<&dyn ComponentProver<CpuBackend>> {
        self.components
            .iter()
            .map(|c| c as &dyn ComponentProver<CpuBackend>)
            .collect_vec()
    }
}
