use std::path::PathBuf;

use genco::prelude::*;
use xshell::{cmd, Shell};

use crate::core::air::materialize::{
    MaterializedArray, MaterializedComputation, MaterializedGraph,
};

pub fn project_root() -> PathBuf {
    std::path::PathBuf::from(std::env::var("CARGO_MANIFEST_DIR").unwrap())
}

pub fn reformat_rust_code(text: String) -> String {
    // Since rustfmt is used with nightly features, it takes 2 runs to reach a fixed point.
    reformat_rust_code_inner(reformat_rust_code_inner(text))
}
pub fn reformat_rust_code_inner(text: String) -> String {
    let sh = Shell::new().unwrap();
    sh.set_var("RUSTUP_TOOLCHAIN", "nightly-2023-07-05");
    let rustfmt_toml = project_root().join("rustfmt.toml");
    let mut stdout = cmd!(sh, "rustfmt --config-path {rustfmt_toml}")
        .stdin(text)
        .read()
        .unwrap();
    if !stdout.ends_with('\n') {
        stdout.push('\n');
    }
    stdout
}

pub fn generate_code(graph: &MaterializedGraph) -> rust::Tokens {
    let mut tokens = quote! {
        $("// Autogenerated file. To regenerate, please run `FIX_TESTS=1 cargo test test_fibonacci_rust_codegen`.\n")
        use crate::core::fields::m31::M31;
        use super::ops::*;
    };

    // Structs.
    append_arrays_struct(&mut tokens, &graph.inputs, "Input");
    append_arrays_struct(&mut tokens, &graph.outputs, "Output");

    // Generation.
    append_body_codegen(&mut tokens, graph);

    tokens
}

fn append_body_codegen(tokens: &mut Tokens<Rust>, graph: &MaterializedGraph) {
    // Unpack inputs.
    let mut input_unpacking = rust::Tokens::new();
    for array in &graph.inputs {
        input_unpacking.extend(quote! {
            let $(&array.name) = &input.$(&array.name);
        });
    }
    // Allocate outputs.
    let mut output_allocations = rust::Tokens::new();
    for array in &graph.outputs {
        output_allocations.extend(quote! {
            // Support parallelism.
            let mut $(&array.name) = Vec::<$(&array.ty)>::with_capacity($(array.size));
            unsafe { $(&array.name).set_len($(array.size)); }
        });
    }
    // Computation body.
    let mut body_tokens = rust::Tokens::new();
    for computation in &graph.computations {
        append_computation(&mut body_tokens, computation);
    }
    // Return output struct.
    let mut output_struct_tokens = rust::Tokens::new();
    for array in &graph.outputs {
        output_struct_tokens.extend(quote! {
            $(&array.name): $(&array.name),
        });
    }
    tokens.extend(quote! {
        pub fn compute(input: &Input) -> Output {
            $(input_unpacking)
            $(output_allocations)
            $(body_tokens)
            Output {
                $(output_struct_tokens)
            }
        }
    });
}

fn append_computation(body_tokens: &mut Tokens<Rust>, computation: &MaterializedComputation) {
    let mut input_bindings = rust::Tokens::new();
    for item in &computation.input_tile {
        let index_tokens = if let Some(modulus) = item.modulus {
            quote! {
                (i * $(item.step) + $(item.offset)) % $(modulus)
            }
        } else {
            quote! {i * $(item.step) + $(item.offset)}
        };
        input_bindings.extend(quote! {
            let $(&item.item_name) =
                $(&item.array_name)[$(index_tokens)];
        });
    }

    let mut fused_op = rust::Tokens::new();
    for node in &computation.fused_op.ops {
        let mut params = rust::Tokens::new();
        for param in &node.op.params {
            params.extend(quote! {
                $(param.to_string()),
            });
        }
        for input in &node.inputs {
            params.extend(quote! {
                $(input),
            });
        }
        fused_op.extend(quote! {
            let $(&node.name) = $(&node.op.name)($(&params));
        });
    }

    let mut output_assignments = rust::Tokens::new();
    for item in &computation.output_tile {
        output_assignments.extend(quote! {
            $(&item.array_name)[i * $(item.step) + $(item.offset)] = $(&item.item_name);
        });
    }
    body_tokens.extend(quote! {
        for i in 0..$(computation.n_repeats) {
            $(input_bindings)
            $(fused_op)
            $(output_assignments)
        }
    });
}

fn append_arrays_struct(tokens: &mut Tokens<Rust>, arrays: &[MaterializedArray], name: &str) {
    let mut input_struct_tokens = rust::Tokens::new();
    for array in arrays {
        input_struct_tokens.extend(quote! {
            pub $(&array.name): Vec<$(&array.ty)>,
        });
    }
    tokens.extend(quote! {
        pub struct $(name) {
            $(input_struct_tokens)
        }
    });
}

#[test]
fn test_fibonacci_rust_codegen() {
    use std::fs;

    use crate::core::air::graph::{OpParam, PointwiseOp};
    use crate::core::air::materialize::{
        FusedNode, FusedOp, MaskItem, MaterializedArray, MaterializedComputation,
        MaterializedGraph, Ordering,
    };
    let graph = MaterializedGraph {
        inputs: vec![MaterializedArray {
            name: "secret".into(),
            size: 1,
            ty: "M31".into(),
        }],
        outputs: vec![MaterializedArray {
            name: "f".into(),
            size: 32,
            ty: "M31".into(),
        }],
        computations: vec![
            MaterializedComputation {
                output_tile: vec![MaskItem {
                    item_name: "one".into(),
                    array_name: "f".into(),
                    offset: 0,
                    step: 1,
                    modulus: None,
                }],
                input_tile: vec![],
                n_repeats: 1,
                fused_op: FusedOp {
                    ops: vec![FusedNode {
                        name: "one".into(),
                        op: PointwiseOp {
                            name: "M31::from_u32_unchecked".into(),
                            params: vec![OpParam::Int(1)],
                        },
                        ty: "M31".into(),
                        inputs: vec![],
                    }],
                },
                ordering: Ordering::Sequential,
            },
            MaterializedComputation {
                output_tile: vec![MaskItem {
                    item_name: "f_secret".into(),
                    array_name: "f".into(),
                    offset: 1,
                    step: 1,
                    modulus: None,
                }],
                input_tile: vec![MaskItem {
                    item_name: "f_secret".into(),
                    array_name: "secret".into(),
                    offset: 1,
                    step: 1,
                    modulus: Some(1),
                }],
                n_repeats: 1,
                fused_op: FusedOp { ops: vec![] },
                ordering: Ordering::Sequential,
            },
            MaterializedComputation {
                output_tile: vec![MaskItem {
                    item_name: "f_rec".into(),
                    array_name: "f".into(),
                    offset: 2,
                    step: 1,
                    modulus: None,
                }],
                input_tile: vec![
                    MaskItem {
                        item_name: "f0".into(),
                        array_name: "f".into(),
                        offset: 0,
                        step: 1,
                        modulus: None,
                    },
                    MaskItem {
                        item_name: "f1".into(),
                        array_name: "f".into(),
                        offset: 1,
                        step: 1,
                        modulus: None,
                    },
                ],
                n_repeats: 32 - 2,
                fused_op: FusedOp {
                    ops: vec![
                        FusedNode {
                            name: "f0sq".into(),
                            op: PointwiseOp {
                                name: "mul".into(),
                                params: vec![],
                            },
                            ty: "M31".into(),
                            inputs: vec!["f0".into(), "f0".into()],
                        },
                        FusedNode {
                            name: "f1sq".into(),
                            op: PointwiseOp {
                                name: "mul".into(),
                                params: vec![],
                            },
                            ty: "M31".into(),
                            inputs: vec!["f1".into(), "f1".into()],
                        },
                        FusedNode {
                            name: "f_rec".into(),
                            op: PointwiseOp {
                                name: "add".into(),
                                params: vec![],
                            },
                            ty: "M31".into(),
                            inputs: vec!["f0sq".into(), "f1sq".into()],
                        },
                    ],
                },
                ordering: Ordering::Sequential,
            },
        ],
    };
    let tokens = generate_code(&graph);
    let text = reformat_rust_code(tokens.to_string().expect("Could not format Rust code."));

    let mut path = project_root();
    path.push("src/examples/fibonacci_code.rs");

    let expected = fs::read_to_string(&path).unwrap();
    if text != expected {
        if std::env::var("FIX_TESTS").is_ok() {
            fs::write(path, text).unwrap();
        } else {
            panic!("Fibonacci codegen file is not up to date. Run with FIX_TESTS=1 to fix");
        }
    }
}
