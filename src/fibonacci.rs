use num_traits::One;

use crate::core::air::definition::{
    Column, ColumnKind, Component, ComponentInstance, UnivariateConstraint,
};
use crate::core::air::expr::{
    BinaryOp, UnivariateExprExtension, UnivariateMaskItem, UnivariatePolyExpression,
};
use crate::core::air::generation::{GenerationFormula, SubcolumnGeneration};
use crate::core::air::mask::{Mask, MaskItem};
use crate::core::air::slice::{Slice, SliceDomain};
use crate::core::circle::Coset;
use crate::core::constraints::{coset_vanishing, point_excluder, point_vanishing, PolyOracle};
use crate::core::fields::m31::BaseField;
use crate::core::fields::Field;
use crate::core::poly::circle::{CanonicCoset, CircleDomain, CircleEvaluation};

pub struct Fibonacci {
    pub trace_coset: CanonicCoset,
    pub eval_domain: CircleDomain,
    pub constraint_coset: Coset,
    pub constraint_eval_domain: CircleDomain,
    pub claim: BaseField,
}

impl Fibonacci {
    pub fn new(n_bits: usize, claim: BaseField) -> Self {
        let trace_coset = CanonicCoset::new(n_bits);
        let eval_domain = trace_coset.eval_domain(n_bits + 1);
        let constraint_coset = Coset::subgroup(n_bits);
        let constraint_eval_domain = CircleDomain::constraint_domain(n_bits + 1);
        Self {
            trace_coset,
            eval_domain,
            constraint_coset,
            constraint_eval_domain,
            claim,
        }
    }

    pub fn get_trace(&self) -> CircleEvaluation {
        // Trace.
        let mut trace = Vec::with_capacity(self.trace_coset.len());

        // Fill trace with fibonacci squared.
        let mut a = BaseField::one();
        let mut b = BaseField::one();
        for _ in 0..self.trace_coset.len() {
            trace.push(a);
            let tmp = a.square() + b.square();
            a = b;
            b = tmp;
        }

        // Returns as a CircleEvaluation.
        CircleEvaluation::new_canonical_ordered(self.trace_coset, trace)
    }

    pub fn eval_step_constraint(&self, trace: impl PolyOracle) -> BaseField {
        trace.get_at(self.trace_coset.index_at(0)).square()
            + trace.get_at(self.trace_coset.index_at(1)).square()
            - trace.get_at(self.trace_coset.index_at(2))
    }

    pub fn eval_step_quotient(&self, trace: impl PolyOracle) -> BaseField {
        let excluded0 = self.constraint_coset.at(self.constraint_coset.len() - 2);
        let excluded1 = self.constraint_coset.at(self.constraint_coset.len() - 1);
        let num = self.eval_step_constraint(trace)
            * point_excluder(excluded0, trace.point())
            * point_excluder(excluded1, trace.point());
        let denom = coset_vanishing(self.constraint_coset, trace.point());
        num / denom
    }

    pub fn eval_boundary_constraint(&self, trace: impl PolyOracle, value: BaseField) -> BaseField {
        trace.get_at(self.trace_coset.index_at(0)) - value
    }

    pub fn eval_boundary_quotient(
        &self,
        trace: impl PolyOracle,
        point_index: usize,
        value: BaseField,
    ) -> BaseField {
        let num = self.eval_boundary_constraint(trace, value);
        let denom = point_vanishing(self.constraint_coset.at(point_index), trace.point());
        num / denom
    }

    pub fn eval_quotient(&self, random_coeff: BaseField, trace: impl PolyOracle) -> BaseField {
        let mut quotient = random_coeff.pow(0) * self.eval_step_quotient(trace);
        quotient += random_coeff.pow(1) * self.eval_boundary_quotient(trace, 0, BaseField::one());
        quotient += random_coeff.pow(2)
            * self.eval_boundary_quotient(trace, self.constraint_coset.len() - 1, self.claim);
        quotient
    }

    pub fn get_mask(&self) -> Mask {
        Mask::new(
            (0..3)
                .map(|offset| MaskItem {
                    column_index: 0,
                    offset,
                })
                .collect(),
        )
    }
}

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

#[cfg(test)]
mod tests {
    use num_traits::One;

    use super::Fibonacci;
    use crate::core::circle::{CirclePoint, CirclePointIndex};
    use crate::core::constraints::{EvalByEvaluation, EvalByPoly};
    use crate::core::fields::m31::{BaseField, M31};
    use crate::core::poly::circle::CircleEvaluation;
    use crate::m31;

    #[test]
    fn test_constraint_on_trace() {
        use num_traits::Zero;

        let fib = Fibonacci::new(3, m31!(1056169651));
        let trace = fib.get_trace();

        // Assert that the step constraint is satisfied on the trace.
        for p_ind in fib
            .constraint_coset
            .iter_indices()
            .take(fib.constraint_coset.len() - 2)
        {
            let res = fib.eval_step_constraint(EvalByEvaluation {
                offset: p_ind,
                eval: &trace,
            });
            assert_eq!(res, BaseField::zero());
        }

        // Assert that the first trace value is 1.
        assert_eq!(
            fib.eval_boundary_constraint(
                EvalByEvaluation {
                    offset: fib.constraint_coset.index_at(0),
                    eval: &trace,
                },
                BaseField::one()
            ),
            BaseField::zero()
        );

        // Assert that the last trace value is the fibonacci claim.
        assert_eq!(
            fib.eval_boundary_constraint(
                EvalByEvaluation {
                    offset: fib
                        .constraint_coset
                        .index_at(fib.constraint_coset.len() - 1),
                    eval: &trace,
                },
                fib.claim
            ),
            BaseField::zero()
        );
    }

    #[test]
    fn test_quotient_is_low_degree() {
        let fib = Fibonacci::new(5, m31!(443693538));
        let trace = fib.get_trace();
        let trace_poly = trace.interpolate();

        let extended_evaluation = trace_poly.clone().evaluate(fib.eval_domain);

        // TODO(ShaharS), Change to a channel implementation to retrieve the random
        // coefficients from extension field.
        let random_coeff = m31!(2213980);

        // Compute quotient on the evaluation domain.
        let mut quotient_values = Vec::with_capacity(fib.constraint_eval_domain.len());
        for p_ind in fib.constraint_eval_domain.iter_indices() {
            quotient_values.push(fib.eval_quotient(
                random_coeff,
                EvalByEvaluation {
                    offset: p_ind,
                    eval: &extended_evaluation,
                },
            ));
        }
        let quotient_eval = CircleEvaluation::new(fib.constraint_eval_domain, quotient_values);
        // Interpolate the poly. The the poly is indeed of degree lower than the size of
        // eval_domain, then it should interpolate correctly.
        let quotient_poly = quotient_eval.interpolate();

        // Evaluate this polynomial at another point, out of eval_domain and compare to what we
        // expect.
        let oods_point_index = CirclePointIndex::generator() * 2;
        assert!(fib.constraint_eval_domain.find(oods_point_index).is_none());
        let oods_point = oods_point_index.to_point();

        let oods_evaluation = fib.get_mask().get_evaluation(
            &[fib.trace_coset],
            &[EvalByPoly {
                point: CirclePoint::zero(),
                poly: &trace_poly,
            }],
            oods_point_index,
        );
        let oods_evaluation = EvalByEvaluation {
            offset: oods_point_index,
            eval: &oods_evaluation,
        };

        assert_eq!(
            quotient_poly.eval_at_point(oods_point),
            fib.eval_quotient(random_coeff, oods_evaluation)
        );
    }

    #[test]
    fn test_component_file() {
        let component = super::create_fibonacci_component_definition(5);
        let json = serde_json::to_string_pretty(&component).unwrap();

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
}
