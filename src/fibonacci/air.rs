use super::component::FibonacciComponent;
use crate::core::air::{Air, Component, ComponentVisitor};
use crate::core::backend::CPUBackend;

pub struct FibonacciAir {
    pub component: FibonacciComponent,
}

impl FibonacciAir {
    pub fn new(component: FibonacciComponent) -> Self {
        Self { component }
    }
}

impl Air<CPUBackend> for FibonacciAir {
    fn visit_components<V: ComponentVisitor<CPUBackend>>(&self, v: &mut V) {
        v.visit(&self.component);
    }

    fn max_constraint_log_degree_bound(&self) -> u32 {
        self.component.max_constraint_log_degree_bound()
    }
}
