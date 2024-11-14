use std::ops::{Add, AddAssign, Mul, MulAssign, Neg, Sub};

use num_traits::{One, Zero};

use super::logup::{LogupAtRow, LogupSums};
use super::{EvalAtRow, RelationEntry, RelationType, INTERACTION_TRACE_IDX};
use crate::core::fields::m31::BaseField;
use crate::core::fields::qm31::SecureField;
use crate::core::fields::FieldExpOps;
use crate::core::lookups::utils::Fraction;

/// A single base field column at index `idx` of interaction `interaction`, at mask offset `offset`.
#[derive(Clone, Debug, PartialEq)]
pub struct ColumnExpr {
    interaction: usize,
    idx: usize,
    offset: usize,
}

#[derive(Clone, Debug, PartialEq)]
pub enum Expr {
    Col(ColumnExpr),
    /// An atomic secure column constructed from 4 expressions.
    /// Expressions on the secure column are not reduced, i.e,
    /// if `a = SecureCol(a0, a1, a2, a3)`, `b = SecureCol(b0, b1, b2, b3)` then
    /// `a + b` evaluates to `Add(a, b)` rather than
    /// `SecureCol(Add(a0, b0), Add(a1, b1), Add(a2, b2), Add(a3, b3))`
    SecureCol([Box<Expr>; 4]),
    Const(BaseField),
    Var(String),
    Add(Box<Expr>, Box<Expr>),
    Sub(Box<Expr>, Box<Expr>),
    Mul(Box<Expr>, Box<Expr>),
    Neg(Box<Expr>),
    Inv(Box<Expr>),
}

impl Expr {
    #[allow(dead_code)]
    pub fn format_expr(&self) -> String {
        match self {
            Expr::Col(ColumnExpr {
                interaction,
                idx,
                offset,
            }) => {
                format!("col_{interaction}_{idx}[{offset}]")
            }
            Expr::SecureCol([a, b, c, d]) => format!(
                "SecureCol({}, {}, {}, {})",
                a.format_expr(),
                b.format_expr(),
                c.format_expr(),
                d.format_expr()
            ),
            Expr::Const(c) => c.0.to_string(),
            Expr::Var(v) => v.to_string(),
            Expr::Add(a, b) => format!("{} + {}", a.format_expr(), b.format_expr()),
            Expr::Sub(a, b) => format!("{} - ({})", a.format_expr(), b.format_expr()),
            Expr::Mul(a, b) => format!("({}) * ({})", a.format_expr(), b.format_expr()),
            Expr::Neg(a) => format!("-({})", a.format_expr()),
            Expr::Inv(a) => format!("1/({})", a.format_expr()),
        }
    }
}

impl From<BaseField> for Expr {
    fn from(val: BaseField) -> Self {
        Expr::Const(val)
    }
}

impl From<SecureField> for Expr {
    fn from(val: SecureField) -> Self {
        Expr::SecureCol([
            Box::new(val.0 .0.into()),
            Box::new(val.0 .1.into()),
            Box::new(val.1 .0.into()),
            Box::new(val.1 .1.into()),
        ])
    }
}

impl Add for Expr {
    type Output = Self;
    fn add(self, rhs: Self) -> Self {
        Expr::Add(Box::new(self), Box::new(rhs))
    }
}

impl Sub for Expr {
    type Output = Self;
    fn sub(self, rhs: Self) -> Self {
        Expr::Sub(Box::new(self), Box::new(rhs))
    }
}

impl Mul for Expr {
    type Output = Self;
    fn mul(self, rhs: Self) -> Self {
        Expr::Mul(Box::new(self), Box::new(rhs))
    }
}

impl AddAssign for Expr {
    fn add_assign(&mut self, rhs: Self) {
        *self = self.clone() + rhs
    }
}

impl MulAssign for Expr {
    fn mul_assign(&mut self, rhs: Self) {
        *self = self.clone() * rhs
    }
}

impl Neg for Expr {
    type Output = Self;
    fn neg(self) -> Self {
        Expr::Neg(Box::new(self))
    }
}

impl Zero for Expr {
    fn zero() -> Self {
        Expr::Const(BaseField::zero())
    }
    fn is_zero(&self) -> bool {
        // TODO(alont): consider replacing `Zero` in the trait bound with a custom trait
        // that only has `zero()`.
        panic!("Can't check if an expression is zero.");
    }
}

impl One for Expr {
    fn one() -> Self {
        Expr::Const(BaseField::one())
    }
}

impl FieldExpOps for Expr {
    fn inverse(&self) -> Self {
        Expr::Inv(Box::new(self.clone()))
    }
}

impl Mul<BaseField> for Expr {
    type Output = Self;
    fn mul(self, rhs: BaseField) -> Self {
        self * Expr::from(rhs)
    }
}

impl Mul<SecureField> for Expr {
    type Output = Self;
    fn mul(self, rhs: SecureField) -> Self {
        self * Expr::from(rhs)
    }
}

impl Add<SecureField> for Expr {
    type Output = Self;
    fn add(self, rhs: SecureField) -> Self {
        self + Expr::from(rhs)
    }
}

impl Sub<SecureField> for Expr {
    type Output = Self;
    fn sub(self, rhs: SecureField) -> Self {
        self - Expr::from(rhs)
    }
}

impl AddAssign<BaseField> for Expr {
    fn add_assign(&mut self, rhs: BaseField) {
        *self = self.clone() + Expr::from(rhs)
    }
}

fn combine_formal<R: RelationType<Expr, Expr>>(relation: &R, values: &[Expr]) -> Expr {
    let z = Expr::Var(relation.get_name().to_owned() + "_z");
    let alpha_powers = (0..relation.get_size())
        .map(|i| Expr::Var(relation.get_name().to_owned() + "_alpha" + &i.to_string()));
    values
        .iter()
        .zip(alpha_powers)
        .fold(Expr::zero(), |acc, (value, power)| {
            acc + power * value.clone()
        })
        - z
}

/// An Evaluator that saves all constraint expressions.
pub struct ExprEvaluator {
    pub cur_var_index: usize,
    pub constraints: Vec<Expr>,
    pub logup: LogupAtRow<Self>,
}

impl ExprEvaluator {
    #[allow(dead_code)]
    pub fn new(log_size: u32, logup_sums: LogupSums) -> Self {
        Self {
            cur_var_index: Default::default(),
            constraints: Default::default(),
            logup: LogupAtRow::new(INTERACTION_TRACE_IDX, logup_sums.0, logup_sums.1, log_size),
        }
    }
}

impl EvalAtRow for ExprEvaluator {
    // TODO(alont): Should there be a version of this that disallows Secure fields for F?
    type F = Expr;
    type EF = Expr;

    fn next_interaction_mask<const N: usize>(
        &mut self,
        interaction: usize,
        offsets: [isize; N],
    ) -> [Self::F; N] {
        std::array::from_fn(|i| {
            let col = ColumnExpr {
                interaction,
                idx: self.cur_var_index,
                offset: offsets[i] as usize,
            };
            self.cur_var_index += 1;
            Expr::Col(col)
        })
    }

    fn add_constraint<G>(&mut self, constraint: G)
    where
        Self::EF: std::ops::Mul<G, Output = Self::EF>,
    {
        self.constraints.push(Expr::one() * constraint);
    }

    fn combine_ef(values: [Self::F; 4]) -> Self::EF {
        Expr::SecureCol([
            Box::new(values[0].clone()),
            Box::new(values[1].clone()),
            Box::new(values[2].clone()),
            Box::new(values[3].clone()),
        ])
    }

    fn add_to_relation<Relation: RelationType<Self::F, Self::EF>>(
        &mut self,
        entries: &[RelationEntry<'_, Self::F, Self::EF, Relation>],
    ) {
        let fracs: Vec<Fraction<Self::EF, Self::EF>> = entries
            .iter()
            .map(
                |RelationEntry {
                     relation,
                     multiplicity,
                     values,
                 }| {
                    Fraction::new(multiplicity.clone(), combine_formal(*relation, values))
                },
            )
            .collect();
        self.write_logup_frac(fracs.into_iter().sum());
    }

    super::logup_proxy!();
}

#[cfg(test)]
mod tests {
    use num_traits::{One, Zero};

    use crate::constraint_framework::expr::{ColumnExpr, Expr, ExprEvaluator};
    use crate::constraint_framework::{
        relation, EvalAtRow, FrameworkEval, RelationEntry, ORIGINAL_TRACE_IDX,
    };
    use crate::core::fields::m31::M31;
    use crate::core::fields::qm31::SecureField;
    use crate::core::fields::FieldExpOps;

    #[test]
    fn test_expr_eval() {
        let test_struct = TestStruct {};
        let eval = test_struct.evaluate(ExprEvaluator::new(16, (SecureField::zero(), None)));
        assert_eq!(eval.constraints.len(), 2);
        assert_eq!(
            eval.constraints[0],
            Expr::Mul(
                Box::new(Expr::one()),
                Box::new(Expr::Mul(
                    Box::new(Expr::Mul(
                        Box::new(Expr::Mul(
                            Box::new(Expr::Col(ColumnExpr {
                                interaction: ORIGINAL_TRACE_IDX,
                                idx: 0,
                                offset: 0
                            })),
                            Box::new(Expr::Col(ColumnExpr {
                                interaction: ORIGINAL_TRACE_IDX,
                                idx: 1,
                                offset: 0
                            }))
                        )),
                        Box::new(Expr::Col(ColumnExpr {
                            interaction: ORIGINAL_TRACE_IDX,
                            idx: 2,
                            offset: 0
                        }))
                    )),
                    Box::new(Expr::Inv(Box::new(Expr::Add(
                        Box::new(Expr::Col(ColumnExpr {
                            interaction: ORIGINAL_TRACE_IDX,
                            idx: 0,
                            offset: 0
                        })),
                        Box::new(Expr::Col(ColumnExpr {
                            interaction: ORIGINAL_TRACE_IDX,
                            idx: 1,
                            offset: 0
                        }))
                    ))))
                ))
            )
        );

        assert_eq!(
            eval.constraints[1],
            Expr::Mul(
                Box::new(Expr::Const(M31(1))),
                Box::new(Expr::Sub(
                    Box::new(Expr::Mul(
                        Box::new(Expr::Sub(
                            Box::new(Expr::Sub(
                                Box::new(Expr::SecureCol([
                                    Box::new(Expr::Col(ColumnExpr {
                                        interaction: 2,
                                        idx: 4,
                                        offset: 0
                                    })),
                                    Box::new(Expr::Col(ColumnExpr {
                                        interaction: 2,
                                        idx: 6,
                                        offset: 0
                                    })),
                                    Box::new(Expr::Col(ColumnExpr {
                                        interaction: 2,
                                        idx: 8,
                                        offset: 0
                                    })),
                                    Box::new(Expr::Col(ColumnExpr {
                                        interaction: 2,
                                        idx: 10,
                                        offset: 0
                                    }))
                                ])),
                                Box::new(Expr::Sub(
                                    Box::new(Expr::SecureCol([
                                        Box::new(Expr::Col(ColumnExpr {
                                            interaction: 2,
                                            idx: 5,
                                            offset: 18446744073709551615
                                        })),
                                        Box::new(Expr::Col(ColumnExpr {
                                            interaction: 2,
                                            idx: 7,
                                            offset: 18446744073709551615
                                        })),
                                        Box::new(Expr::Col(ColumnExpr {
                                            interaction: 2,
                                            idx: 9,
                                            offset: 18446744073709551615
                                        })),
                                        Box::new(Expr::Col(ColumnExpr {
                                            interaction: 2,
                                            idx: 11,
                                            offset: 18446744073709551615
                                        }))
                                    ])),
                                    Box::new(Expr::Mul(
                                        Box::new(Expr::Col(ColumnExpr {
                                            interaction: 0,
                                            idx: 3,
                                            offset: 0
                                        })),
                                        Box::new(Expr::SecureCol([
                                            Box::new(Expr::Const(M31(0))),
                                            Box::new(Expr::Const(M31(0))),
                                            Box::new(Expr::Const(M31(0))),
                                            Box::new(Expr::Const(M31(0)))
                                        ]))
                                    ))
                                ))
                            )),
                            Box::new(Expr::Const(M31(0)))
                        )),
                        Box::new(Expr::Sub(
                            Box::new(Expr::Add(
                                Box::new(Expr::Add(
                                    Box::new(Expr::Add(
                                        Box::new(Expr::Const(M31(0))),
                                        Box::new(Expr::Mul(
                                            Box::new(Expr::Var("TestRelation_alpha0".to_string())),
                                            Box::new(Expr::Col(ColumnExpr {
                                                interaction: 1,
                                                idx: 0,
                                                offset: 0
                                            }))
                                        ))
                                    )),
                                    Box::new(Expr::Mul(
                                        Box::new(Expr::Var("TestRelation_alpha1".to_string())),
                                        Box::new(Expr::Col(ColumnExpr {
                                            interaction: 1,
                                            idx: 1,
                                            offset: 0
                                        }))
                                    ))
                                )),
                                Box::new(Expr::Mul(
                                    Box::new(Expr::Var("TestRelation_alpha2".to_string())),
                                    Box::new(Expr::Col(ColumnExpr {
                                        interaction: 1,
                                        idx: 2,
                                        offset: 0
                                    }))
                                ))
                            )),
                            Box::new(Expr::Var("TestRelation_z".to_string()))
                        ))
                    )),
                    Box::new(Expr::Const(M31(1)))
                ))
            )
        );
    }

    #[test]
    fn test_format_expr() {
        let test_struct = TestStruct {};
        let eval = test_struct.evaluate(ExprEvaluator::new(16, (SecureField::zero(), None)));
        assert_eq!(eval.constraints[0].format_expr(), "(1) * ((((col_1_0[0]) * (col_1_1[0])) * (col_1_2[0])) * (1/(col_1_0[0] + col_1_1[0])))");
        assert_eq!(eval.constraints[1].format_expr(), "(1) * ((SecureCol(col_2_4[0], col_2_6[0], col_2_8[0], col_2_10[0]) - (SecureCol(col_2_5[18446744073709551615], col_2_7[18446744073709551615], col_2_9[18446744073709551615], col_2_11[18446744073709551615]) - ((col_0_3[0]) * (SecureCol(0, 0, 0, 0)))) - (0)) * (0 + (TestRelation_alpha0) * (col_1_0[0]) + (TestRelation_alpha1) * (col_1_1[0]) + (TestRelation_alpha2) * (col_1_2[0]) - (TestRelation_z)) - (1))");
    }

    relation!(TestRelation, 3);

    struct TestStruct {}
    impl FrameworkEval for TestStruct {
        fn log_size(&self) -> u32 {
            0
        }
        fn max_constraint_log_degree_bound(&self) -> u32 {
            0
        }
        fn evaluate<E: EvalAtRow>(&self, mut eval: E) -> E {
            let x0 = eval.next_trace_mask();
            let x1 = eval.next_trace_mask();
            let x2 = eval.next_trace_mask();
            eval.add_constraint(
                x0.clone() * x1.clone() * x2.clone() * (x0.clone() + x1.clone()).inverse(),
            );
            eval.add_to_relation(&[RelationEntry::new(
                &TestRelation::dummy(),
                E::EF::one(),
                &[x0, x1, x2],
            )]);
            eval.finalize_logup();
            eval
        }
    }
}
