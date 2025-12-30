use num_traits::{One, Zero};
use stwo::core::fields::m31::BaseField;
use stwo::core::fields::qm31::SecureField;

use super::{with_arena, BaseExpr, BaseExprNode, ExtExpr, ExtExprNode};

impl BaseExpr {
    /// Helper function, use [`simplify`] instead.
    ///
    /// Simplifies an expression by applying basic arithmetic rules.
    fn unchecked_simplify(&self) -> Self {
        let node = self.node();
        match node {
            BaseExprNode::Add(a, b) => {
                let a = a.unchecked_simplify();
                let b = b.unchecked_simplify();
                let a_node = a.node();
                let b_node = b.node();
                match (&a_node, &b_node) {
                    // Simplify constants.
                    (BaseExprNode::Const(av), BaseExprNode::Const(bv)) => {
                        BaseExpr::from(*av + *bv)
                    }
                    (BaseExprNode::Const(av), _) if av.is_zero() => b, // 0 + b = b
                    (_, BaseExprNode::Const(bv)) if bv.is_zero() => a, // a + 0 = a
                    // Simplify Negs.
                    (BaseExprNode::Neg(minus_a), BaseExprNode::Neg(minus_b)) => {
                        -(*minus_a + *minus_b)
                    } // (-a + -b) = -(a + b)
                    (BaseExprNode::Neg(minus_a), _) => b - *minus_a, // -a + b = b - a
                    (_, BaseExprNode::Neg(minus_b)) => a - *minus_b, // a + -b = a - b
                    // No simplification.
                    _ => a + b,
                }
            }
            BaseExprNode::Sub(a, b) => {
                let a = a.unchecked_simplify();
                let b = b.unchecked_simplify();
                let a_node = a.node();
                let b_node = b.node();
                match (&a_node, &b_node) {
                    // Simplify constants.
                    (BaseExprNode::Const(av), BaseExprNode::Const(bv)) => {
                        BaseExpr::from(*av - *bv)
                    }
                    (BaseExprNode::Const(av), _) if av.is_zero() => -b, // 0 - b = -b
                    (_, BaseExprNode::Const(bv)) if bv.is_zero() => a,  // a - 0 = a
                    // Simplify Negs.
                    (BaseExprNode::Neg(minus_a), BaseExprNode::Neg(minus_b)) => {
                        *minus_b - *minus_a
                    } // (-a - -b) = b - a
                    (BaseExprNode::Neg(minus_a), _) => -(*minus_a + b), // -a - b = -(a + b)
                    (_, BaseExprNode::Neg(minus_b)) => a + *minus_b,    // a - -b = a + b
                    // No simplification.
                    _ => a - b,
                }
            }
            BaseExprNode::Mul(a, b) => {
                let a = a.unchecked_simplify();
                let b = b.unchecked_simplify();
                let a_node = a.node();
                let b_node = b.node();
                match (&a_node, &b_node) {
                    // Simplify constants.
                    (BaseExprNode::Const(av), BaseExprNode::Const(bv)) => {
                        BaseExpr::from(*av * *bv)
                    }
                    (BaseExprNode::Const(av), _) if av.is_zero() => BaseExpr::zero(), // 0 * b = 0
                    (_, BaseExprNode::Const(bv)) if bv.is_zero() => BaseExpr::zero(), // a * 0 = 0
                    (BaseExprNode::Const(av), _) if *av == BaseField::one() => b,     // 1 * b = b
                    (_, BaseExprNode::Const(bv)) if *bv == BaseField::one() => a,     // a * 1 = a
                    (BaseExprNode::Const(av), _) if -*av == BaseField::one() => -b,   // -1 * b = -b
                    (_, BaseExprNode::Const(bv)) if -*bv == BaseField::one() => -a,   // a * -1 = -a
                    // Simplify Negs.
                    (BaseExprNode::Neg(minus_a), BaseExprNode::Neg(minus_b)) => {
                        *minus_a * *minus_b
                    } // (-a) * (-b) = a * b
                    (BaseExprNode::Neg(minus_a), _) => -(*minus_a * b), // (-a) * b = -(a * b)
                    (_, BaseExprNode::Neg(minus_b)) => -(a * *minus_b), // a * (-b) = -(a * b)
                    // No simplification.
                    _ => a * b,
                }
            }
            BaseExprNode::Neg(a) => {
                let a = a.unchecked_simplify();
                let a_node = a.node();
                match a_node {
                    BaseExprNode::Const(c) => BaseExpr::from(-c),
                    BaseExprNode::Neg(minus_a) => minus_a,   // -(-a) = a
                    BaseExprNode::Sub(a, b) => b - a,        // -(a - b) = b - a
                    _ => -a,                                 // No simplification.
                }
            }
            BaseExprNode::Inv(a) => {
                let a = a.unchecked_simplify();
                let a_node = a.node();
                match a_node {
                    BaseExprNode::Inv(inv_a) => inv_a, // 1 / (1 / a) = a
                    BaseExprNode::Const(c) => BaseExpr::from(c.inverse()),
                    _ => with_arena(|arena| arena.base_inv(a)),
                }
            }
            // No simplification for Col, Const, Param.
            _ => *self,
        }
    }

    /// Simplifies an expression by applying basic arithmetic rules and ensures that the result is
    /// equivalent to the original expression by assigning random values.
    pub fn simplify(&self) -> Self {
        let simplified = self.unchecked_simplify();
        assert_eq!(self.random_eval(), simplified.random_eval());
        simplified
    }

    pub fn simplify_and_format(&self) -> String {
        self.simplify().format_expr()
    }
}

impl ExtExpr {
    /// Helper function, use [`simplify`] instead.
    ///
    /// Simplifies an expression by applying basic arithmetic rules.
    fn unchecked_simplify(&self) -> Self {
        let node = self.node();
        match node {
            ExtExprNode::Add(a, b) => {
                let a = a.unchecked_simplify();
                let b = b.unchecked_simplify();
                let a_node = a.node();
                let b_node = b.node();
                match (&a_node, &b_node) {
                    // Simplify constants.
                    (ExtExprNode::Const(av), ExtExprNode::Const(bv)) => ExtExpr::from(*av + *bv),
                    (ExtExprNode::Const(av), _) if av.is_zero() => b, // 0 + b = b
                    (_, ExtExprNode::Const(bv)) if bv.is_zero() => a, // a + 0 = a
                    // Simplify Negs.
                    (ExtExprNode::Neg(minus_a), ExtExprNode::Neg(minus_b)) => -(*minus_a + *minus_b), // (-a + -b) = -(a + b)
                    (ExtExprNode::Neg(minus_a), _) => b - *minus_a, // -a + b = b - a
                    (_, ExtExprNode::Neg(minus_b)) => a - *minus_b, // a + -b = a - b
                    // No simplification.
                    _ => a + b,
                }
            }
            ExtExprNode::Sub(a, b) => {
                let a = a.unchecked_simplify();
                let b = b.unchecked_simplify();
                let a_node = a.node();
                let b_node = b.node();
                match (&a_node, &b_node) {
                    // Simplify constants.
                    (ExtExprNode::Const(av), ExtExprNode::Const(bv)) => ExtExpr::from(*av - *bv),
                    (ExtExprNode::Const(av), _) if av.is_zero() => -b, // 0 - b = -b
                    (_, ExtExprNode::Const(bv)) if bv.is_zero() => a,  // a - 0 = a
                    // Simplify Negs.
                    (ExtExprNode::Neg(minus_a), ExtExprNode::Neg(minus_b)) => *minus_b - *minus_a, // (-a - -b) = b - a
                    (ExtExprNode::Neg(minus_a), _) => -(*minus_a + b), // -a - b = -(a + b)
                    (_, ExtExprNode::Neg(minus_b)) => a + *minus_b,    // a - -b = a + b
                    // No simplification.
                    _ => a - b,
                }
            }
            ExtExprNode::Mul(a, b) => {
                let a = a.unchecked_simplify();
                let b = b.unchecked_simplify();
                let a_node = a.node();
                let b_node = b.node();
                match (&a_node, &b_node) {
                    // Simplify constants.
                    (ExtExprNode::Const(av), ExtExprNode::Const(bv)) => ExtExpr::from(*av * *bv),
                    (ExtExprNode::Const(av), _) if av.is_zero() => ExtExpr::zero(), // 0 * b = 0
                    (_, ExtExprNode::Const(bv)) if bv.is_zero() => ExtExpr::zero(), // a * 0 = 0
                    (ExtExprNode::Const(av), _) if *av == SecureField::one() => b,  // 1 * b = b
                    (_, ExtExprNode::Const(bv)) if *bv == SecureField::one() => a,  // a * 1 = a
                    (ExtExprNode::Const(av), _) if -*av == SecureField::one() => -b, // -1 * b = -b
                    (_, ExtExprNode::Const(bv)) if -*bv == SecureField::one() => -a, // a * -1 = -a
                    // Simplify Negs.
                    (ExtExprNode::Neg(minus_a), ExtExprNode::Neg(minus_b)) => *minus_a * *minus_b, // (-a) * (-b) = a * b
                    (ExtExprNode::Neg(minus_a), _) => -(*minus_a * b), // (-a) * b = -(a * b)
                    (_, ExtExprNode::Neg(minus_b)) => -(a * *minus_b), // a * (-b) = -(a * b)
                    // No simplification.
                    _ => a * b,
                }
            }
            ExtExprNode::Neg(a) => {
                let a = a.unchecked_simplify();
                let a_node = a.node();
                match a_node {
                    ExtExprNode::Const(c) => ExtExpr::from(-c),
                    ExtExprNode::Neg(minus_a) => minus_a,    // -(-a) = a
                    ExtExprNode::Sub(a, b) => b - a,         // -(a - b) = b - a
                    _ => -a,                                 // No simplification.
                }
            }
            ExtExprNode::SecureCol([a, b, c, d]) => {
                let a = a.unchecked_simplify();
                let b = b.unchecked_simplify();
                let c = c.unchecked_simplify();
                let d = d.unchecked_simplify();
                let a_node = a.node();
                let b_node = b.node();
                let c_node = c.node();
                let d_node = d.node();
                match (&a_node, &b_node, &c_node, &d_node) {
                    (
                        BaseExprNode::Const(av),
                        BaseExprNode::Const(bv),
                        BaseExprNode::Const(cv),
                        BaseExprNode::Const(dv),
                    ) => ExtExpr::from(SecureField::from_m31_array([*av, *bv, *cv, *dv])),
                    _ => ExtExpr::secure_col([a, b, c, d]),
                }
            }
            // No simplification for Const, Param.
            _ => *self,
        }
    }

    /// Simplifies an expression by applying basic arithmetic rules and ensures that the result is
    /// equivalent to the original expression by assigning random values.
    pub fn simplify(&self) -> Self {
        let simplified = self.unchecked_simplify();
        assert_eq!(self.random_eval(), simplified.random_eval());
        simplified
    }

    pub fn simplify_and_format(&self) -> String {
        self.simplify().format_expr()
    }
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use rand::rngs::SmallRng;
    use rand::{Rng, SeedableRng};
    use stwo::core::fields::m31::BaseField;
    use stwo::core::fields::qm31::SecureField;

    use crate::expr::utils::*;
    use crate::expr::init_arena;
    use crate::AssertEvaluator;

    #[test]
    fn test_simplify_expr() {
        init_arena();

        let c0 = col!(1, 0, 0);
        let c1 = col!(1, 1, 0);
        let a = var!("a");
        let b = qvar!("b");
        let zero = felt!(0);
        let qzero = qfelt!(0, 0, 0, 0);
        let one = felt!(1);
        let qone = qfelt!(1, 0, 0, 0);
        let minus_one = felt!(stwo::core::fields::m31::P - 1);
        let qminus_one = qfelt!(stwo::core::fields::m31::P - 1, 0, 0, 0);

        let mut rng = SmallRng::seed_from_u64(0);
        let columns: HashMap<(usize, usize, isize), BaseField> =
            HashMap::from([((1, 0, 0), rng.gen()), ((1, 1, 0), rng.gen())]);
        let vars: HashMap<String, BaseField> = HashMap::from([("a".to_string(), rng.gen())]);
        let ext_vars: HashMap<String, SecureField> = HashMap::from([("b".to_string(), rng.gen())]);

        let base_expr = (((zero + c0) + (a + zero))
            * ((-c1) + (-c0))
            + (-(-(a + a + c0)))
            - zero)
            + (a - zero)
            + (-c1 - (a * a))
            + (a * zero)
            - (zero * c1)
            + one
                * a
                * one
                * c1
                * (-a)
                * c1
                * (minus_one * c0);

        let expr = (qzero
            + secure_col!(
                base_expr,
                base_expr,
                zero,
                one
            )
            - qzero)
            * qone
            * b
            * qminus_one;

        let full_eval = expr.eval_expr::<AssertEvaluator<'_>, _, _, _>(&columns, &vars, &ext_vars);
        let simplified_eval = expr
            .simplify()
            .eval_expr::<AssertEvaluator<'_>, _, _, _>(&columns, &vars, &ext_vars);

        assert_eq!(full_eval, simplified_eval);
    }
}
