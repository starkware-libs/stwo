use num_traits::One;

use crate::core::air::definition::{
    Column, ColumnKind, Component, ComponentInstance, UnivariateConstraint,
};
use crate::core::air::expr::{
    BinaryOp, UnivariateExprExtension, UnivariateMaskItem, UnivariatePolyExpression,
};
use crate::core::air::generation::{GenerationFormula, SubcolumnGeneration};
use crate::core::air::slice::{Slice, SliceDomain};
use crate::core::fields::m31::BaseField;

pub fn create_fibonacci_component_definition(n_bits: u32) -> Component {
    Component {
        name: "Fibonacci".to_string(),
        version: "0.1".to_string(),
        description: "Hand written fibonacci component".to_string(),
        instances: vec![ComponentInstance {
            n_bits,
            columns: vec![
                Column {
                    name: "f".to_string(),
                    description: "Fibonacci values".to_string(),
                    n_bits,
                    kind: ColumnKind::Witness {
                        generation: vec![
                            SubcolumnGeneration {
                                domain: SliceDomain {
                                    inclusions: vec![Slice {
                                        offset: 0,
                                        log_steps: n_bits,
                                    }],
                                    exclusions: vec![],
                                },
                                formula: GenerationFormula::Explicit(
                                    UnivariatePolyExpression::Value(BaseField::one()),
                                ),
                            },
                            SubcolumnGeneration {
                                domain: SliceDomain {
                                    inclusions: vec![Slice {
                                        offset: 1,
                                        log_steps: n_bits,
                                    }],
                                    exclusions: vec![],
                                },
                                formula: GenerationFormula::Explicit(
                                    UnivariatePolyExpression::Extension(
                                        UnivariateExprExtension::MaskItem(UnivariateMaskItem {
                                            column: "secret".to_string(),
                                            log_expand: 0,
                                            offset: 0, // TODO(spapini): Think about the offset.
                                        }),
                                    ),
                                ),
                            },
                            SubcolumnGeneration {
                                domain: SliceDomain {
                                    inclusions: vec![Slice {
                                        offset: 0,
                                        log_steps: 0,
                                    }],
                                    exclusions: vec![Slice {
                                        offset: 0,
                                        log_steps: n_bits,
                                    }],
                                },
                                formula: GenerationFormula::Explicit(
                                    UnivariatePolyExpression::Extension(
                                        UnivariateExprExtension::MaskItem(UnivariateMaskItem {
                                            column: "f".to_string(),
                                            log_expand: 0,
                                            offset: -1,
                                        }),
                                    ),
                                ),
                            },
                        ],
                    },
                },
                Column {
                    name: "secret".to_string(),
                    description: "Secret value".to_string(),
                    n_bits: 0,
                    kind: ColumnKind::GenerationInput,
                },
                Column {
                    name: "claim".to_string(),
                    description: "Claim value".to_string(),
                    n_bits,
                    kind: ColumnKind::Constant,
                },
                Column {
                    name: "claim_mask".to_string(),
                    description: "Claim mask".to_string(),
                    n_bits,
                    kind: ColumnKind::Constant,
                },
            ],
            constraints: vec![
                // Initial 1.
                UnivariateConstraint {
                    name: "initial_1".to_string(),
                    description: "Check that the first fibonacci value is 1".to_string(),
                    domain: SliceDomain {
                        inclusions: vec![Slice {
                            offset: 0,
                            log_steps: 0,
                        }],
                        exclusions: vec![],
                    },
                    // The expression f[0] - 1.
                    expr: UnivariatePolyExpression::BinaryOp {
                        op: BinaryOp::Sub,
                        lhs: Box::new(UnivariatePolyExpression::Extension(
                            UnivariateExprExtension::MaskItem(UnivariateMaskItem {
                                column: "f".to_string(),
                                log_expand: 0,
                                offset: 0,
                            }),
                        )),
                        rhs: Box::new(UnivariatePolyExpression::Value(BaseField::one())),
                    },
                },
                // Initial secret.
                UnivariateConstraint {
                    name: "initial_secret".to_string(),
                    description: "Check that the second fibonacci value is the secret".to_string(),
                    domain: SliceDomain {
                        inclusions: vec![Slice {
                            offset: 1,
                            log_steps: 0,
                        }],
                        exclusions: vec![],
                    },
                    // The expression f[1] - secret.
                    expr: UnivariatePolyExpression::BinaryOp {
                        op: BinaryOp::Sub,
                        lhs: Box::new(UnivariatePolyExpression::Extension(
                            UnivariateExprExtension::MaskItem(UnivariateMaskItem {
                                column: "f".to_string(),
                                log_expand: 0,
                                offset: 1,
                            }),
                        )),
                        rhs: Box::new(UnivariatePolyExpression::Extension(
                            UnivariateExprExtension::MaskItem(UnivariateMaskItem {
                                column: "secret".to_string(),
                                log_expand: 0,
                                offset: 0,
                            }),
                        )),
                    },
                },
                // Step.
                UnivariateConstraint {
                    name: "step".to_string(),
                    description:
                        "Check that the next fibonacci value is the sum of the previous two."
                            .to_string(),
                    domain: SliceDomain {
                        inclusions: vec![Slice {
                            offset: 0,
                            log_steps: 0,
                        }],
                        exclusions: vec![
                            Slice {
                                offset: -2,
                                log_steps: n_bits,
                            },
                            Slice {
                                offset: -1,
                                log_steps: n_bits,
                            },
                        ],
                    },
                    // The expression f[i]**2 + f[i + 1] **2 - f[i + 2].
                    expr: UnivariatePolyExpression::BinaryOp {
                        op: BinaryOp::Sub,
                        lhs: Box::new(UnivariatePolyExpression::BinaryOp {
                            op: BinaryOp::Add,
                            lhs: Box::new(UnivariatePolyExpression::Pow {
                                base: Box::new(UnivariatePolyExpression::Extension(
                                    UnivariateExprExtension::MaskItem(UnivariateMaskItem {
                                        column: "f".to_string(),
                                        log_expand: 0,
                                        offset: 0,
                                    }),
                                )),
                                exp: 2,
                            }),
                            rhs: Box::new(UnivariatePolyExpression::Pow {
                                base: Box::new(UnivariatePolyExpression::Extension(
                                    UnivariateExprExtension::MaskItem(UnivariateMaskItem {
                                        column: "f".to_string(),
                                        log_expand: 0,
                                        offset: 1,
                                    }),
                                )),
                                exp: 2,
                            }),
                        }),
                        rhs: Box::new(UnivariatePolyExpression::Extension(
                            UnivariateExprExtension::MaskItem(UnivariateMaskItem {
                                column: "f".to_string(),
                                log_expand: 0,
                                offset: 2,
                            }),
                        )),
                    },
                },
                // Claim.
                UnivariateConstraint {
                    name: "claim".to_string(),
                    description: "Check that the claim matches the values at the specified mask."
                        .to_string(),
                    domain: SliceDomain {
                        inclusions: vec![Slice {
                            offset: 0,
                            log_steps: 0,
                        }],
                        exclusions: vec![],
                    },
                    // The expression (f - claim) * claim_mask.
                    expr: UnivariatePolyExpression::BinaryOp {
                        op: BinaryOp::Mul,
                        lhs: Box::new(UnivariatePolyExpression::BinaryOp {
                            op: BinaryOp::Sub,
                            lhs: Box::new(UnivariatePolyExpression::Extension(
                                UnivariateExprExtension::MaskItem(UnivariateMaskItem {
                                    column: "f".to_string(),
                                    log_expand: 0,
                                    offset: 0,
                                }),
                            )),
                            rhs: Box::new(UnivariatePolyExpression::Extension(
                                UnivariateExprExtension::MaskItem(UnivariateMaskItem {
                                    column: "claim".to_string(),
                                    log_expand: 0,
                                    offset: 0,
                                }),
                            )),
                        }),
                        rhs: Box::new(UnivariatePolyExpression::Extension(
                            UnivariateExprExtension::MaskItem(UnivariateMaskItem {
                                column: "claim_mask".to_string(),
                                log_expand: 0,
                                offset: 0,
                            }),
                        )),
                    },
                },
            ],
            interaction_elements: vec![],
            outputs: vec![],
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
