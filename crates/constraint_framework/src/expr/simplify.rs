use num_traits::{One, Zero};

use super::{BaseExpr, ExtExpr};
use crate::core::fields::qm31::SecureField;

/// Applies simplifications to arithmetic expressions that can be used both for `BaseExpr` and for
/// `ExtExpr`.
macro_rules! simplify_arithmetic {
    ($self:tt) => {
        match $self.clone() {
            Self::Add(a, b) => {
                let a = a.simplify();
                let b = b.simplify();
                match (a.clone(), b.clone()) {
                    // Simplify constants.
                    (Self::Const(a), Self::Const(b)) => Self::Const(a + b),
                    (Self::Const(a_val), _) if a_val.is_zero() => b, // 0 + b = b
                    (_, Self::Const(b_val)) if b_val.is_zero() => a, // a + 0 = a
                    // Simplify Negs.
                    // (-a + -b) = -(a + b)
                    (Self::Neg(minus_a), Self::Neg(minus_b)) => -(*minus_a + *minus_b),
                    (Self::Neg(minus_a), _) => b - *minus_a, // -a + b = b - a
                    (_, Self::Neg(minus_b)) => a - *minus_b, // a + -b = a - b
                    // No simplification.
                    _ => a + b,
                }
            }
            Self::Sub(a, b) => {
                let a = a.simplify();
                let b = b.simplify();
                match (a.clone(), b.clone()) {
                    // Simplify constants.
                    (Self::Const(a), Self::Const(b)) => Self::Const(a - b), // Simplify consts.
                    (Self::Const(a_val), _) if a_val.is_zero() => -b,       // 0 - b = -b
                    (_, Self::Const(b_val)) if b_val.is_zero() => a,        // a - 0 = a
                    // Simplify Negs.
                    // (-a - -b) = b - a
                    (Self::Neg(minus_a), Self::Neg(minus_b)) => *minus_b - *minus_a,
                    (Self::Neg(minus_a), _) => -(*minus_a + b), // -a - b = -(a + b)
                    (_, Self::Neg(minus_b)) => a + *minus_b,    // a + -b = a - b
                    // No Simplification.
                    _ => a - b,
                }
            }
            Self::Mul(a, b) => {
                let a = a.simplify();
                let b = b.simplify();
                match (a.clone(), b.clone()) {
                    // Simplify consts.
                    (Self::Const(a), Self::Const(b)) => Self::Const(a * b),
                    (Self::Const(a_val), _) if a_val.is_zero() => Self::zero(), // 0 * b = 0
                    (_, Self::Const(b_val)) if b_val.is_zero() => Self::zero(), // a * 0 = 0
                    (Self::Const(a_val), _) if a_val == One::one() => b,        // 1 * b = b
                    (_, Self::Const(b_val)) if b_val == One::one() => a,        // a * 1 = a
                    (Self::Const(a_val), _) if -a_val == One::one() => -b,      // -1 * b = -b
                    (_, Self::Const(b_val)) if -b_val == One::one() => -a,      // a * -1 = -a
                    // Simplify Negs.
                    // (-a) * (-b) = a * b
                    (Self::Neg(minus_a), Self::Neg(minus_b)) => *minus_a * *minus_b,
                    (Self::Neg(minus_a), _) => -(*minus_a * b), // (-a) * b = -(a * b)
                    (_, Self::Neg(minus_b)) => -(a * *minus_b), // a * (-b) = -(a * b)
                    // No simplification.
                    _ => a * b,
                }
            }
            Self::Neg(a) => {
                let a = a.simplify();
                match a {
                    Self::Const(c) => Self::Const(-c),
                    Self::Neg(minus_a) => *minus_a,     // -(-a) = a
                    Self::Sub(a, b) => Self::Sub(b, a), // -(a - b) = b - a
                    _ => -a,                            // No simplification.
                }
            }
            other => other, // No simplification.
        }
    };
}

impl BaseExpr {
    /// Helper function, use [`simplify`] instead.
    ///
    /// Simplifies an expression by applying basic arithmetic rules.
    fn unchecked_simplify(&self) -> Self {
        let simple = simplify_arithmetic!(self);
        match simple {
            Self::Inv(a) => {
                let a = a.unchecked_simplify();
                match a {
                    Self::Inv(inv_a) => *inv_a, // 1 / (1 / a) = a
                    Self::Const(c) => Self::Const(c.inverse()),
                    _ => Self::Inv(Box::new(a)),
                }
            }
            other => other,
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
        let simple = simplify_arithmetic!(self);
        match simple {
            Self::SecureCol([a, b, c, d]) => {
                let a = a.unchecked_simplify();
                let b = b.unchecked_simplify();
                let c = c.unchecked_simplify();
                let d = d.unchecked_simplify();
                match (a.clone(), b.clone(), c.clone(), d.clone()) {
                    (
                        BaseExpr::Const(a_val),
                        BaseExpr::Const(b_val),
                        BaseExpr::Const(c_val),
                        BaseExpr::Const(d_val),
                    ) => ExtExpr::Const(SecureField::from_m31_array([a_val, b_val, c_val, d_val])),
                    _ => Self::SecureCol([Box::new(a), Box::new(b), Box::new(c), Box::new(d)]),
                }
            }
            other => other,
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

    use crate::constraint_framework::expr::utils::*;
    use crate::constraint_framework::AssertEvaluator;
    use crate::core::fields::m31::BaseField;
    use crate::core::fields::qm31::SecureField;
    #[test]
    fn test_simplify_expr() {
        let c0 = col!(1, 0, 0);
        let c1 = col!(1, 1, 0);
        let a = var!("a");
        let b = qvar!("b");
        let zero = felt!(0);
        let qzero = qfelt!(0, 0, 0, 0);
        let one = felt!(1);
        let qone = qfelt!(1, 0, 0, 0);
        let minus_one = felt!(crate::core::fields::m31::P - 1);
        let qminus_one = qfelt!(crate::core::fields::m31::P - 1, 0, 0, 0);

        let mut rng = SmallRng::seed_from_u64(0);
        let columns: HashMap<(usize, usize, isize), BaseField> =
            HashMap::from([((1, 0, 0), rng.gen()), ((1, 1, 0), rng.gen())]);
        let vars: HashMap<String, BaseField> = HashMap::from([("a".to_string(), rng.gen())]);
        let ext_vars: HashMap<String, SecureField> = HashMap::from([("b".to_string(), rng.gen())]);

        let base_expr = (((zero.clone() + c0.clone()) + (a.clone() + zero.clone()))
            * ((-c1.clone()) + (-c0.clone()))
            + (-(-(a.clone() + a.clone() + c0.clone())))
            - zero.clone())
            + (a.clone() - zero.clone())
            + (-c1.clone() - (a.clone() * a.clone()))
            + (a.clone() * zero.clone())
            - (zero.clone() * c1.clone())
            + one.clone()
                * a.clone()
                * one.clone()
                * c1.clone()
                * (-a.clone())
                * c1.clone()
                * (minus_one.clone() * c0.clone());

        let expr = (qzero.clone()
            + secure_col!(
                base_expr.clone(),
                base_expr.clone(),
                zero.clone(),
                one.clone()
            )
            - qzero.clone())
            * qone.clone()
            * b.clone()
            * qminus_one.clone();

        let full_eval = expr.eval_expr::<AssertEvaluator<'_>, _, _, _>(&columns, &vars, &ext_vars);
        let simplified_eval = expr
            .simplify()
            .eval_expr::<AssertEvaluator<'_>, _, _, _>(&columns, &vars, &ext_vars);

        assert_eq!(full_eval, simplified_eval);
    }
}
