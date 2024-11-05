use std::ops::{Add, AddAssign, Mul, MulAssign, Neg, Sub};

use num_traits::{One, Zero};

use super::EvalAtRow;
use crate::core::fields::m31::BaseField;
use crate::core::fields::qm31::SecureField;
use crate::core::fields::FieldExpOps;

/// A single base field column at index `idx` of interaction `interaction`, at mask offset `offset`.
#[derive(Clone, Debug, PartialEq)]
struct ColumnExpr {
    interaction: usize,
    idx: usize,
    offset: usize,
}

#[derive(Clone, Debug, PartialEq)]
enum Expr {
    Col(ColumnExpr),
    /// An atomic secure column constructed from 4 expressions.
    /// Expressions on the secure column are not reduced, i.e,
    /// if `a = SecureCol(a0, a1, a2, a3)`, `b = SecureCol(b0, b1, b2, b3)` then
    /// `a + b` evaluates to `Add(a, b)` rather than
    /// `SecureCol(Add(a0, b0), Add(a1, b1), Add(a2, b2), Add(a3, b3))`
    SecureCol([Box<Expr>; 4]),
    Const(BaseField),
    Add(Box<Expr>, Box<Expr>),
    Sub(Box<Expr>, Box<Expr>),
    Mul(Box<Expr>, Box<Expr>),
    Neg(Box<Expr>),
    Inv(Box<Expr>),
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

/// An Evaluator that saves all constraint expressions.
#[derive(Default)]
struct ExprEvaluator {
    cur_var_index: usize,
    constraints: Vec<Expr>,
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
}

#[cfg(test)]
mod tests {
    use num_traits::One;

    use crate::constraint_framework::expr::{ColumnExpr, Expr, ExprEvaluator};
    use crate::constraint_framework::{EvalAtRow, FrameworkEval, ORIGINAL_TRACE_IDX};
    use crate::core::fields::FieldExpOps;
    #[test]
    fn test_expr_eval() {
        let test_struct = TestStruct {};
        let eval = test_struct.evaluate(ExprEvaluator::default());
        assert_eq!(eval.constraints.len(), 1);
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
    }

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
            eval.add_constraint(x0.clone() * x1.clone() * x2 * (x0 + x1).inverse());
            eval
        }
    }
}
