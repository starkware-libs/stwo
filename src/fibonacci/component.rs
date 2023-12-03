use crate::core::air::definition::{Column, ColumnKind, Component, ComponentInstance, Constraint};
use crate::core::air::graph::{GraphNode, OpParam};

pub fn create_fibonacci_component_definition(n_bits: u32) -> Component {
    Component {
        name: "Fibonacci".to_string(),
        version: "0.1".to_string(),
        description: "Hand written fibonacci component".to_string(),
        instances: vec![ComponentInstance {
            n_bits,
            generation_graph: vec![
                // Fibonacci generation.
                GraphNode {
                    name: "f".to_string(),
                    description: "Fibonacci values".to_string(),
                    size: 1 << n_bits,
                    ty: "M31".to_string(),
                    op: "concat".to_string(),
                    params: vec![],
                    inputs: vec!["one".to_string(), "secret".to_string(), "f_rec".to_string()],
                },
                // One generation.
                GraphNode {
                    name: "one".to_string(),
                    description: "One value".to_string(),
                    size: 1,
                    ty: "M31".to_string(),
                    op: "constant".to_string(),
                    params: vec![OpParam::Int(1)],
                    inputs: vec![],
                },
                // Secret generation.
                GraphNode {
                    name: "secret".to_string(),
                    description: "Secret value".to_string(),
                    size: 1,
                    ty: "M31".to_string(),
                    op: "generation_input".to_string(),
                    params: vec![OpParam::String("secret".to_string())],
                    inputs: vec![],
                },
                // Recursive fibonacci generation.
                GraphNode {
                    name: "f_rec".to_string(),
                    description: "f[:2]**2 + f[1:-1]**2".to_string(),
                    size: (1 << n_bits) - 2,
                    ty: "M31".to_string(),
                    op: "add".to_string(),
                    params: vec![],
                    inputs: vec!["f0sq".to_string(), "f1sq".to_string()],
                },
                // f0sq generation.
                GraphNode {
                    name: "f0sq".to_string(),
                    description: "f[:2]**2".to_string(),
                    size: (1 << n_bits) - 2,
                    ty: "M31".to_string(),
                    op: "mul".to_string(),
                    params: vec![],
                    inputs: vec!["f0".to_string(), "f0".to_string()],
                },
                // f1sq generation.
                GraphNode {
                    name: "f1sq".to_string(),
                    description: "f[1:-1]**2".to_string(),
                    size: (1 << n_bits) - 2,
                    ty: "M31".to_string(),
                    op: "mul".to_string(),
                    params: vec![],
                    inputs: vec!["f1".to_string(), "f1".to_string()],
                },
                // f0 generation.
                GraphNode {
                    name: "f0".to_string(),
                    description: "f[:-2]".to_string(),
                    size: (1 << n_bits) - 2,
                    ty: "M31".to_string(),
                    op: "slice".to_string(),
                    params: vec![
                        OpParam::Int(0),
                        OpParam::Int((1 << n_bits) - 2),
                        OpParam::Int(1),
                    ],
                    inputs: vec!["f".to_string()],
                },
                // f1 generation.
                GraphNode {
                    name: "f1".to_string(),
                    description: "f[1:-1]".to_string(),
                    size: (1 << n_bits) - 2,
                    ty: "M31".to_string(),
                    op: "slice".to_string(),
                    params: vec![
                        OpParam::Int(1),
                        OpParam::Int((1 << n_bits) - 2),
                        OpParam::Int(1),
                    ],
                    inputs: vec!["f".to_string()],
                },
            ],
            columns: vec![Column {
                name: "f".to_string(),
                description: "Fibonacci values".to_string(),
                generation_node: "f".to_string(),
                kind: ColumnKind::Witness,
            }],
            outputs: vec![],
            constraint_graph: vec![
                // Initial 1.
                GraphNode {
                    name: "initial_1".to_string(),
                    description: "Check that the first fibonacci value is 1".to_string(),
                    size: 1,
                    ty: "M31".to_string(),
                    op: "sub".to_string(),
                    params: vec![],
                    inputs: vec!["f0".to_string(), "one".to_string()],
                },
                // one.
                GraphNode {
                    name: "one".to_string(),
                    description: "One value".to_string(),
                    size: 1,
                    ty: "M31".to_string(),
                    op: "constant".to_string(),
                    params: vec![OpParam::Int(1)],
                    inputs: vec![],
                },
                // f.
                GraphNode {
                    name: "f".to_string(),
                    description: "Fibonacci values".to_string(),
                    size: 1 << n_bits,
                    ty: "M31".to_string(),
                    op: "commited_column".to_string(),
                    params: vec![OpParam::String("f".to_string())],
                    inputs: vec![],
                },
                // f0.
                GraphNode {
                    name: "f0".to_string(),
                    description: "f[0]".to_string(),
                    size: 1,
                    ty: "M31".to_string(),
                    op: "slice".to_string(),
                    params: vec![OpParam::Int(0), OpParam::Int(1), OpParam::Int(1)],
                    inputs: vec!["f".to_string()],
                },
                // claim.
                GraphNode {
                    name: "claim".to_string(),
                    description: "f[claim_index] - claim".to_string(),
                    size: 1,
                    ty: "M31".to_string(),
                    op: "sub".to_string(),
                    params: vec![],
                    inputs: vec!["f_at_claim_index".to_string(), "claim".to_string()],
                },
                // f_at_claim_index.
                GraphNode {
                    name: "f_at_claim_index".to_string(),
                    description: "f[claim_index]".to_string(),
                    size: 1,
                    ty: "M31".to_string(),
                    op: "slice".to_string(),
                    params: vec![
                        OpParam::String("claim_index".to_string()),
                        OpParam::Int(1),
                        OpParam::Int(1),
                    ],
                    inputs: vec!["f".to_string()],
                },
                // claim_index.
                GraphNode {
                    name: "claim_index".to_string(),
                    description: "Claim index".to_string(),
                    size: 1,
                    ty: "u64".to_string(),
                    op: "public_input".to_string(),
                    params: vec![OpParam::String("claim_index".to_string())],
                    inputs: vec![],
                },
                // Step.
                GraphNode {
                    name: "step".to_string(),
                    description: "f[2:]**2 - f[1:-1]**2 - f[:-2]".to_string(),
                    size: 1,
                    ty: "M31".to_string(),
                    op: "sub".to_string(),
                    params: vec![],
                    inputs: vec!["f2".to_string(), "f_rec".to_string()],
                },
                // f2.
                GraphNode {
                    name: "f2".to_string(),
                    description: "f[2:]".to_string(),
                    size: (1 << n_bits) - 2,
                    ty: "M31".to_string(),
                    op: "slice".to_string(),
                    params: vec![
                        OpParam::Int(2),
                        OpParam::Int((1 << n_bits) - 2),
                        OpParam::Int(1),
                    ],
                    inputs: vec!["f".to_string()],
                },
                // f_rec.
                GraphNode {
                    name: "f_rec".to_string(),
                    description: "f[1:-1]**2 + f[:-2]".to_string(),
                    size: (1 << n_bits) - 2,
                    ty: "M31".to_string(),
                    op: "add".to_string(),
                    params: vec![],
                    inputs: vec!["f0sq".to_string(), "f1sq".to_string()],
                },
                // f0sq.
                GraphNode {
                    name: "f0sq".to_string(),
                    description: "f[:-2]**2".to_string(),
                    size: (1 << n_bits) - 2,
                    ty: "M31".to_string(),
                    op: "mul".to_string(),
                    params: vec![],
                    inputs: vec!["f0".to_string(), "f0".to_string()],
                },
                // f1sq.
                GraphNode {
                    name: "f1sq".to_string(),
                    description: "f[1:-1]**2".to_string(),
                    size: (1 << n_bits) - 2,
                    ty: "M31".to_string(),
                    op: "mul".to_string(),
                    params: vec![],
                    inputs: vec!["f1".to_string(), "f1".to_string()],
                },
                // f0.
                GraphNode {
                    name: "f0".to_string(),
                    description: "f[:-2]".to_string(),
                    size: (1 << n_bits) - 2,
                    ty: "M31".to_string(),
                    op: "slice".to_string(),
                    params: vec![
                        OpParam::Int(0),
                        OpParam::Int((1 << n_bits) - 2),
                        OpParam::Int(1),
                    ],
                    inputs: vec!["f".to_string()],
                },
                // f1.
                GraphNode {
                    name: "f1".to_string(),
                    description: "f[1:-1]".to_string(),
                    size: (1 << n_bits) - 2,
                    ty: "M31".to_string(),
                    op: "slice".to_string(),
                    params: vec![
                        OpParam::Int(1),
                        OpParam::Int((1 << n_bits) - 2),
                        OpParam::Int(1),
                    ],
                    inputs: vec!["f".to_string()],
                },
            ],
            constraints: vec![
                // Initial 1.
                Constraint {
                    name: "initial_1".to_string(),
                    description: "Check that the first fibonacci value is 1".to_string(),
                    constraint_node: "initial_1".to_string(),
                },
                // Initial secret.
                Constraint {
                    name: "initial_secret".to_string(),
                    description: "Check that the second fibonacci value is the secret".to_string(),
                    constraint_node: "initial_secret".to_string(),
                },
                // Step.
                Constraint {
                    name: "step".to_string(),
                    description:
                        "Check that the next fibonacci value is the sum of the previous two."
                            .to_string(),
                    constraint_node: "step".to_string(),
                },
                // Claim.
                Constraint {
                    name: "claim".to_string(),
                    description: "Check that the claim matches the values at the specified mask."
                        .to_string(),
                    constraint_node: "claim".to_string(),
                },
            ],
            interaction_elements: vec![],
        }],
    }
}

#[test]
fn test_component_file() {
    let component = create_fibonacci_component_definition(5);
    let json = serde_json::to_string_pretty(&component).unwrap() + "\n";

    // Compute the path to a nearby file.
    // Use the cargo environment variable to get the correct path to the source directory.
    let mut path = std::path::PathBuf::from(std::env::var("CARGO_MANIFEST_DIR").unwrap());
    path.push("src/examples/fibonacci_component.json");

    // Compare against the local file fibonacci_component.json.
    let expected_json = std::fs::read_to_string(path.clone()).unwrap();
    if json != expected_json {
        // Fix the component file if the FIX_TESTS environment variable is set.
        if std::env::var("FIX_TESTS").is_ok() {
            std::fs::write(path, json).unwrap();
        } else {
            panic!("Fibonacci component file is not up to date. Run with FIX_TESTS=1 to fix.");
        }
    }
}
