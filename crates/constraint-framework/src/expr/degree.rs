/// Finds a degree bound for an expressions. The degree is given with respect to columns as
/// variables.
/// Computes the actual degree with the following caveats:
///     1. The constant expression 0 receives degree 0 like all other constants rather than the
///        mathematically correct -infinity. This means, for example, that expresisons of the
///        type 0 * expr will return degree deg expr. This should be mitigated by
///        simplification.
///     2. If expressions p and q cancel out under some operation, this will not be accounted
///        for, so that (x^2 + 1) - (x^2 + x) will return degree 2.
use std::collections::HashMap;

use num_traits::Zero;

use super::{BaseExpr, BaseExprNode, ExtExpr, ExtExprNode};

type Degree = usize;

/// A struct of named expressions that can be searched when determining the degree bound for an
/// expression that contains parameters.
/// Required because expressions that contain parameters that are actually intermediates have to
/// account for the degree of the intermediate.
pub struct NamedExprs {
    exprs: HashMap<String, BaseExpr>,
    ext_exprs: HashMap<String, ExtExpr>,
}

impl NamedExprs {
    pub const fn new(
        exprs: HashMap<String, BaseExpr>,
        ext_exprs: HashMap<String, ExtExpr>,
    ) -> Self {
        Self { exprs, ext_exprs }
    }

    pub fn degree_bound(&self, name: String) -> Degree {
        if let Some(expr) = self.exprs.get(&name) {
            expr.degree_bound(self)
        } else if let Some(expr) = self.ext_exprs.get(&name) {
            expr.degree_bound(self)
        } else if name.starts_with("preprocessed.") {
            // TODO(alont): Fix this hack.
            1
        } else {
            // If expression isn't found assume it's an external variable, effectively a const.
            0
        }
    }
}

impl BaseExpr {
    pub fn degree_bound(&self, named_exprs: &NamedExprs) -> Degree {
        let node = self.node();
        match node {
            BaseExprNode::Col(_) => 1,
            BaseExprNode::Const(_) => 0,
            BaseExprNode::Param(name) => named_exprs.degree_bound(name.as_str()),
            BaseExprNode::Add(a, b) => a.degree_bound(named_exprs).max(b.degree_bound(named_exprs)),
            BaseExprNode::Sub(a, b) => a.degree_bound(named_exprs).max(b.degree_bound(named_exprs)),
            BaseExprNode::Mul(a, b) => a.degree_bound(named_exprs) + b.degree_bound(named_exprs),
            BaseExprNode::Neg(a) => a.degree_bound(named_exprs),
            BaseExprNode::Inv(expr) => {
                let expr_node = expr.node();
                match expr_node {
                    BaseExprNode::Param(name) if named_exprs.degree_bound(name.as_str()).is_zero() => 0,
                    BaseExprNode::Const(_) => 0,
                    _ => panic!("Cannot compute the degree of an inverse"),
                }
            }
        }
    }
}

impl ExtExpr {
    pub fn degree_bound(&self, named_exprs: &NamedExprs) -> Degree {
        let node = self.node();
        match node {
            ExtExprNode::SecureCol(coefs) => coefs
                .iter()
                .map(|coef| coef.degree_bound(named_exprs))
                .max()
                .unwrap(),
            ExtExprNode::Const(_) => 0,
            ExtExprNode::Param(name) => named_exprs.degree_bound(name.as_str()),
            ExtExprNode::Add(a, b) => a.degree_bound(named_exprs).max(b.degree_bound(named_exprs)),
            ExtExprNode::Sub(a, b) => a.degree_bound(named_exprs).max(b.degree_bound(named_exprs)),
            ExtExprNode::Mul(a, b) => a.degree_bound(named_exprs) + b.degree_bound(named_exprs),
            ExtExprNode::Neg(a) => a.degree_bound(named_exprs),
        }
    }
}

#[cfg(test)]
mod tests {
    use stwo::core::fields::FieldExpOps;

    use crate::expr::degree::NamedExprs;
    use crate::expr::init_arena;
    use crate::expr::utils::*;

    #[test]
    fn test_degree_bound() {
        init_arena();

        let intermediate = (felt!(12) + col!(1, 1, 0)) * var!("a") * col!(1, 0, 0);
        let qintermediate = secure_col!(intermediate, felt!(12), var!("b"), felt!(0));

        let low_degree_intermediate = felt!(12345);

        let named_exprs = NamedExprs {
            exprs: [
                ("intermediate".to_string(), intermediate),
                (
                    "low_degree_intermediate".to_string(),
                    low_degree_intermediate,
                ),
            ]
            .into(),
            ext_exprs: [("qintermediate".to_string(), qintermediate)].into(),
        };

        let expr = var!("intermediate") * col!(2, 1, 0);
        let qexpr =
            var!("qintermediate") * secure_col!(col!(2, 1, 0), expr, felt!(0), felt!(1));

        assert_eq!(intermediate.degree_bound(&named_exprs), 2);
        assert_eq!(qintermediate.degree_bound(&named_exprs), 2);
        assert_eq!(expr.degree_bound(&named_exprs), 3);
        assert_eq!(qexpr.degree_bound(&named_exprs), 5);

        // Multiplty by the inverse of a constant and a constant intermediate.
        assert_eq!(
            (expr * felt!(3141).inverse() * var!("low_degree_intermediate").inverse())
                .degree_bound(&named_exprs),
            3
        );
    }
}
