use crate::core::air::{Air, Component, ComponentVisitor};
use crate::core::backend::CPUBackend;
use crate::core::commitment_scheme::TreeColumns;
use crate::core::ComponentVec;

pub fn component_wise_to_tree_wise<T>(
    _air: &impl Air<CPUBackend>,
    values: ComponentVec<T>,
) -> TreeColumns<T> {
    TreeColumns::new(vec![values.0.into_iter().flatten().collect()])
}

pub fn tree_wise_to_component_wise<T>(
    air: &impl Air<CPUBackend>,
    mut values: TreeColumns<T>,
) -> ComponentVec<T> {
    // Recombine the trace values by component, and not by tree.
    struct Visitor<'a, T> {
        by_tree: &'a mut std::vec::IntoIter<T>,
        by_component: ComponentVec<T>,
    }
    impl<T> ComponentVisitor<CPUBackend> for Visitor<'_, T> {
        fn visit<C: Component<CPUBackend>>(&mut self, component: &C) {
            self.by_component
                .0
                .push(self.by_tree.take(component.mask().len()).collect());
        }
    }
    let values = values.0.remove(0);
    let mut recombiner = Visitor {
        by_tree: &mut values.into_iter(),
        by_component: ComponentVec::default(),
    };
    air.visit_components(&mut recombiner);
    recombiner.by_component
}
