use crate::core::air::Air;
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
    let values = &mut values.0.remove(0).into_iter();
    let mut by_component = ComponentVec::default();
    for component in air.components() {
        by_component
            .0
            .push(values.take(component.mask().len()).collect());
    }
    by_component
}
