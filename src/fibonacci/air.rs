use super::component::FibonacciComponent;
use crate::core::air::{Air, ComponentVisitor};

pub struct FibonacciAir {
    pub component: FibonacciComponent,
}

impl FibonacciAir {
    pub fn new(component: FibonacciComponent) -> Self {
        Self { component }
    }
}

impl Air for FibonacciAir {
    fn visit_components<V: ComponentVisitor>(&self, v: &mut V) {
        v.visit(&self.component);
    }
}
