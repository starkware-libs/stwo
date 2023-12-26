use super::definition::ComponentInstance;
use super::graph::{GraphNode, PointwiseOp};
use super::materialize::{
    FusedNode, FusedOp, MaskItem, MaterializedArray, MaterializedComputation, Ordering,
};

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
    component_instance: &ComponentInstance,
) -> Vec<MaterializedComputation> {
    let ordering = vec![1_usize, 2];
    let mut computations = vec![];
    for index in ordering {
        let node = &component_instance.generation_graph[index];
        println!("node: {:?}", node);
        let computation = get_materialized_computation(node);
        computations.push(computation);
    }
    computations
}

pub fn get_materialized_computation(node: &GraphNode) -> MaterializedComputation {
    let node_params: Vec<(String, String)> =
        node.params.iter().map(|param| param.unwrap()).collect();
    match node.op.as_str() {
        "constant" => MaterializedComputation {
            output_tile: vec![],
            input_tile: vec![],
            n_repeats: 1,
            fused_op: FusedOp {
                ops: node_params
                    .iter()
                    .map(|(_param_ty, param_value)| FusedNode {
                        name: node.name.clone(),
                        op: PointwiseOp::Const {
                            value: param_value.clone(),
                            ty: node.ty.clone(),
                        },
                        ty: node.ty.clone(),
                    })
                    .collect(),
            },
            ordering: Ordering::Sequential,
        },
        "generation_input" => MaterializedComputation {
            output_tile: vec![],
            input_tile: node_params
                .iter()
                .map(|(_param_ty, param_value)| MaskItem {
                    item_name: node.name.clone(),
                    array_name: param_value.clone(),
                    offset: 0,
                    step: 1,
                    modulus: None,
                })
                .collect(),
            n_repeats: 1,
            fused_op: FusedOp { ops: vec![] },
            ordering: Ordering::Sequential,
        },
        &_ => unimplemented!(),
    }
}
