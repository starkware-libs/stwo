use super::definition::ComponentInstance;
use super::materialize::{MaterializedArray, MaterializedComputation};

pub fn get_component_inputs(component: &ComponentInstance) -> Vec<MaterializedArray> {
    let mut inputs = vec![];
    for node in &component.generation_graph {
        if node.op == "generation_input" {
            inputs.push(MaterializedArray {
                name: node.name.clone(),
                size: node.size,
                ty: node.ty.clone(),
            });
        }
    }
    inputs
}

pub fn get_component_outputs(component: &ComponentInstance) -> Vec<MaterializedArray> {
    let mut outputs = vec![];
    for column in &component.columns {
        // find node in generation graph
        let column_generation_node = component
            .generation_graph
            .iter()
            .find(|n| n.name == column.generation_node)
            .unwrap();
        outputs.push(MaterializedArray {
            name: column_generation_node.name.clone(),
            size: column_generation_node.size,
            ty: column_generation_node.ty.clone(),
        });
    }
    outputs
}

pub fn get_component_computations(
    _component_instance: &ComponentInstance,
) -> Vec<MaterializedComputation> {
    unimplemented!()
}
