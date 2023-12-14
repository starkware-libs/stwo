use crate::core::air::definition::{Column, ColumnKind, Component, ComponentInstance, Constraint, InteractionElement};
use crate::core::air::graph::{GraphNode, OpParam};

pub fn create_diluted_component_definition(n_bits: u32) -> Component {
    Component {
        name: "Diluted".to_string(),
        version: "0.1".to_string(),
        description: "Hand written diluted component".to_string(),
        instances: vec![ComponentInstance {
            n_bits,
            generation_graph: vec![
                GraphNode {
                    name: "values".to_string(),
                    description: "Input values to check of diluted form".to_string(),
                    size: 1 << n_bits,
                    ty: "M31".to_string(),
                    op: "generation_input".to_string(),
                    params: vec![OpParam::String("values".to_string())],
                    inputs: vec![],
                },
                GraphNode {
                    name: "one".to_string(),
                    description: "One value in column".to_string(),
                    size: 1 << n_bits,
                    ty: "M31".to_string(),
                    op: "constant".to_string(),
                    params: vec![OpParam::Int(1)],
                    inputs: vec![],
                },
                GraphNode {
                    name: "log_up_random_shift".to_string(),
                    description: "log_up_random_shift interaction element in column".to_string(),
                    size: 1 << n_bits,
                    ty: "QM31".to_string(),
                    op: "constant".to_string(),
                    params: vec![OpParam::String("log_up_shift_element".to_string())],
                    inputs: vec![],
                },
                GraphNode {
                    name: "shifted_values".to_string(),
                    description: "Shifted values".to_string(),
                    size: 1 << n_bits,
                    ty: "QM31".to_string(),
                    op: "sub".to_string(),
                    params: vec![],
                    inputs: vec!["values".to_string(), "log_up_random_shift".to_string()],
                },
                GraphNode {
                    name: "inv_shifted_values".to_string(),
                    description: "Inverses of the shifted values".to_string(),
                    size: 1 << n_bits,
                    ty: "QM31".to_string(),
                    op: "div".to_string(),
                    params: vec![],
                    inputs: vec!["one".to_string(), "shifted_values".to_string()],
                },
                GraphNode {
                    name: "zero".to_string(),
                    description: "One value in column".to_string(),
                    size: 1,
                    ty: "QM31".to_string(),
                    op: "constant".to_string(),
                    params: vec![OpParam::Int(0)],
                    inputs: vec![],
                },
                GraphNode {
                    name: "partial_sums".to_string(),
                    description: "Sum of the inverses of the shifted values".to_string(),
                    size: (1 << n_bits) + 1,
                    ty: "QM31".to_string(),
                    op: "concat".to_string(),
                    params: vec![],
                    inputs: vec!["zero".to_string(), "partial_sums_rec".to_string()],
                },
                GraphNode {
                    name: "partial_sums0".to_string(),
                    description: "Recursive Sum of the inverses of the shifted values".to_string(),
                    size: 1 << n_bits,
                    ty: "QM31".to_string(),
                    op: "slice".to_string(),
                    params: vec![
                        OpParam::Int(0),
                        OpParam::Int(1 << n_bits),
                        OpParam::Int(1),
                    ],
                    inputs: vec!["partial_sums".to_string()],
                },
                GraphNode {
                    name: "partial_sums_rec".to_string(),
                    description: "Recursive Sum of the inverses of the shifted values".to_string(),
                    size: 1 << n_bits,
                    ty: "QM31".to_string(),
                    op: "add".to_string(),
                    params: vec![],
                    inputs: vec!["partial_sums0".to_string(), "inv_shifted_values".to_string()],
                },
                GraphNode {
                    name: "total_sum".to_string(),
                    description: "Total Sum of the inverses of the shifted values".to_string(),
                    size: 1,
                    ty: "QM31".to_string(),
                    op: "slice".to_string(),
                    params: vec![
                        OpParam::Int(1 << n_bits),
                        OpParam::Int(1 << n_bits + 1),
                        OpParam::Int(1),
                    ],
                    inputs: vec!["partial_sums".to_string()],
                },
            ],
            columns: vec![
                Column {
                    name: "Diluted input values".to_string(),
                    description: "values".to_string(),
                    generation_node: "values".to_string(),
                    kind: ColumnKind::Witness,
                },
                Column {
                    name: "partial_sums".to_string(),
                    description: "The partial sums of the shifted inverses of the diluted values"
                        .to_string(),
                    generation_node: "partial_sums".to_string(),
                    kind: ColumnKind::Witness,
                },
            ],
            outputs: vec!["total_sum".to_string()],
            constraint_graph: vec![],
            constraints: vec![],
            interaction_elements: vec![
                InteractionElement {
                    name: "log_up_shift_element".to_string(),
                    description: "Random element for shifting the logup".to_string(),
                    witness_dependencies: vec!["values".to_string()],
                },
            ],
        }],
    }
}
