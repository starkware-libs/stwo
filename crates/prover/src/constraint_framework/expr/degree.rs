/// Finds a degree bound for an expressions. The degree is given with respect to columns as
/// variables.
/// Computes the actual degree with the following caveats:
///     1. The constant expression 0 receives degree 0 like all other constants rather than the
///        mathematically correcy -infinity. This means, for example, that expresisons of the
///        type 0 * expr will return degree deg expr. This should be mitigated by
///        simplification.
///     2. If expressions p and q cancel out under some operation, this will not be accounted
///        for, so that (x^2 + 1) - (x^2 + x) will return degree 2.
use std::collections::HashMap;

use super::{BaseExpr, ExtExpr};

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
        match self {
            BaseExpr::Col(_) => 1,
            BaseExpr::Const(_) => 0,
            BaseExpr::Param(name) => named_exprs.degree_bound(name.clone()),
            BaseExpr::Add(a, b) => a.degree_bound(named_exprs).max(b.degree_bound(named_exprs)),
            BaseExpr::Sub(a, b) => a.degree_bound(named_exprs).max(b.degree_bound(named_exprs)),
            BaseExpr::Mul(a, b) => a.degree_bound(named_exprs) + b.degree_bound(named_exprs),
            BaseExpr::Neg(a) => a.degree_bound(named_exprs),
            // TODO(alont): Consider handling this in the type system.
            BaseExpr::Inv(_) => panic!("Cannot compute the degree of an inverse."),
        }
    }
}

impl ExtExpr {
    pub fn degree_bound(&self, named_exprs: &NamedExprs) -> Degree {
        match self {
            ExtExpr::SecureCol(coefs) => coefs
                .iter()
                .cloned()
                .map(|coef| coef.degree_bound(named_exprs))
                .max()
                .unwrap(),
            ExtExpr::Const(_) => 0,
            ExtExpr::Param(name) => named_exprs.degree_bound(name.clone()),
            ExtExpr::Add(a, b) => a.degree_bound(named_exprs).max(b.degree_bound(named_exprs)),
            ExtExpr::Sub(a, b) => a.degree_bound(named_exprs).max(b.degree_bound(named_exprs)),
            ExtExpr::Mul(a, b) => a.degree_bound(named_exprs) + b.degree_bound(named_exprs),
            ExtExpr::Neg(a) => a.degree_bound(named_exprs),
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::constraint_framework::expr::degree::NamedExprs;
    use crate::constraint_framework::expr::utils::*;

    #[test]
    fn test_degree_bound() {
        let intermediate = (felt!(12) + col!(1, 1, 0)) * var!("a") * col!(1, 0, 0);
        let qintermediate = secure_col!(intermediate.clone(), felt!(12), var!("b"), felt!(0));

        let named_exprs = NamedExprs {
            exprs: [("intermediate".to_string(), intermediate.clone())].into(),
            ext_exprs: [("qintermediate".to_string(), qintermediate.clone())].into(),
        };

        let expr = var!("intermediate") * col!(2, 1, 0);
        let qexpr =
            var!("qintermediate") * secure_col!(col!(2, 1, 0), expr.clone(), felt!(0), felt!(1));

        assert_eq!(intermediate.degree_bound(&named_exprs), 2);
        assert_eq!(qintermediate.degree_bound(&named_exprs), 2);
        assert_eq!(expr.degree_bound(&named_exprs), 3);
        assert_eq!(qexpr.degree_bound(&named_exprs), 5);
    }
}
