use std::collections::{BTreeMap, BTreeSet};

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
    let mut computations = vec![];
    let output_tiles = get_output_tiles(component_instance);
    let ordering: Vec<Vec<usize>> = vec![vec![1], vec![2]];

    for indices in ordering {
        let nodes: Vec<&GraphNode> = indices
            .into_iter()
            .map(|index| &component_instance.generation_graph[index])
            .collect();
        computations.extend(get_materialized_computation(nodes, &output_tiles));
    }
    computations
}

pub fn get_materialized_computation(
    nodes: Vec<&GraphNode>,
    output_tiles: &Vec<MaskItem>,
) -> Vec<MaterializedComputation> {
    let mut computations = vec![];
    for (_node_index, node) in nodes.iter().enumerate() {
        let node_params: Vec<(String, String)> =
            node.params.iter().map(|param| param.unwrap()).collect();
        let node_tile = output_tiles.iter().find(|tile| tile.item_name == node.name);
        let node_computation = match node.op.as_str() {
            "constant" => MaterializedComputation {
                output_tile: match node_tile {
                    Some(tile) => vec![tile.clone()],
                    None => vec![],
                },
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
                output_tile: match node_tile {
                    Some(tile) => vec![tile.clone()],
                    None => vec![],
                },
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
        };
        computations.push(node_computation);
    }
    computations
}

pub fn get_output_tiles(component_instance: &ComponentInstance) -> Vec<MaskItem> {
    let mut output_tiles: Vec<MaskItem> = vec![];
    let column_nodes = get_column_generation_nodes(component_instance);
    for column_generation_node in &column_nodes {
        let column_inputs_nodes: Vec<&GraphNode> = column_generation_node
            .inputs
            .iter()
            .map(|node_name| component_instance.get_generation_node(node_name))
            .collect();

        match column_generation_node.op.as_str() {
            "concat" => {
                let mut offset = 0;
                for (_input_node_index, input_node) in column_inputs_nodes {
                    output_tiles.push(MaskItem {
                        item_name: input_node.name.clone(),
                        array_name: column_generation_node.name.clone(),
                        offset,
                        step: 1,
                        modulus: None,
                    });

                    offset += input_node.size;
                }
            }
            "interleave" => {
                let first_size = column_inputs_nodes[0].1.size;
                assert!(column_inputs_nodes
                    .iter()
                    .all(|node| node.1.size == first_size));
                assert_eq!(
                    column_generation_node.size,
                    column_inputs_nodes.len() as u64 * first_size
                );
                for (offset, (_node_index, input_node)) in
                    column_inputs_nodes.into_iter().enumerate()
                {
                    output_tiles.push(MaskItem {
                        item_name: input_node.name.clone(),
                        array_name: column_generation_node.name.clone(),
                        offset: offset as u64,
                        step: input_node.size,
                        modulus: None,
                    });
                }
            }
            "pad" => {
                unimplemented!()
            }
            "repeat" => {
                unimplemented!()
            }
            &_ => unimplemented!(),
        }
    }
    output_tiles
}

pub fn get_intersection(
    nodes: Vec<(usize, &GraphNode)>,
    output_tiles: &Vec<MaskItem>,
) -> Vec<usize> {
    let mut intersection = vec![];
    for (index, node) in nodes.iter() {
        let node_tile = output_tiles.iter().find(|tile| tile.item_name == node.name);
        if node_tile.is_some() {
            intersection.push(*index);
        }
    }
    intersection
}

pub fn get_column_generation_nodes(
    component_instance: &ComponentInstance,
) -> BTreeMap<usize, &GraphNode> {
    let mut column_nodes: BTreeMap<usize, &GraphNode> = BTreeMap::new();
    for column in &component_instance.columns {
        let (node_index, column_generation_node) =
            component_instance.get_generation_node(&column.generation_node);
        column_nodes.insert(node_index, column_generation_node);
    }
    column_nodes
}

pub fn get_input_indices(
    component_instance: &ComponentInstance,
    nodes: BTreeMap<usize, &GraphNode>,
) -> Vec<usize> {
    let mut node_input_names = nodes
        .iter()
        .flat_map(|(_index, node)| node.inputs.clone())
        .collect::<BTreeSet<String>>();
    for node in nodes.values() {
        node_input_names.remove(&node.name);
    }
    component_instance
        .generation_graph
        .iter()
        .enumerate()
        .filter(|(_index, node)| node_input_names.contains(&node.name))
        .map(|(index, _node)| index)
        .collect::<Vec<usize>>()
}
