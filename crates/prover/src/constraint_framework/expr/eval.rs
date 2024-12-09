use std::collections::{HashMap, HashSet};
use std::hash::{DefaultHasher, Hash, Hasher};
use std::ops::{Add, Index};

use itertools::sorted;

use super::{BaseExpr, ColumnExpr, ExtExpr};
use crate::constraint_framework::{AssertEvaluator, EvalAtRow};
use crate::core::fields::m31::BaseField;
use crate::core::fields::qm31::SecureField;
use crate::core::fields::FieldExpOps;

/// An assignment to the variables that may appear in an expression.
pub type ExprVarAssignment = (
    HashMap<(usize, usize, isize), BaseField>,
    HashMap<String, BaseField>,
    HashMap<String, SecureField>,
);

/// Three sets representing all the variables that can appear in an expression:
///    * `cols`: The columns of the AIR.
///    * `params`: The formal parameters to the AIR.
///    * `ext_params`: The extension field parameters to the AIR.
#[derive(Default)]
pub struct ExprVariables {
    pub cols: HashSet<ColumnExpr>,
    pub params: HashSet<String>,
    pub ext_params: HashSet<String>,
}

impl ExprVariables {
    pub fn col(col: ColumnExpr) -> Self {
        Self {
            cols: vec![col].into_iter().collect(),
            params: HashSet::new(),
            ext_params: HashSet::new(),
        }
    }

    pub fn param(param: String) -> Self {
        Self {
            cols: HashSet::new(),
            params: vec![param].into_iter().collect(),
            ext_params: HashSet::new(),
        }
    }

    pub fn ext_param(param: String) -> Self {
        Self {
            cols: HashSet::new(),
            params: HashSet::new(),
            ext_params: vec![param].into_iter().collect(),
        }
    }

    /// Generates a random assignment to the variables.
    /// Note that the assignment is deterministic in the sets of variables (disregarding their
    /// order), and this is required.
    pub fn random_assignment(&self, salt: usize) -> ExprVarAssignment {
        let cols = sorted(self.cols.iter())
            .map(|col| {
                ((col.interaction, col.idx, col.offset), {
                    let mut hasher = DefaultHasher::new();
                    (salt, col).hash(&mut hasher);
                    (hasher.finish() as u32).into()
                })
            })
            .collect();

        let params = sorted(self.params.iter())
            .map(|param| {
                (param.clone(), {
                    let mut hasher = DefaultHasher::new();
                    (salt, param).hash(&mut hasher);
                    (hasher.finish() as u32).into()
                })
            })
            .collect();

        let ext_params = sorted(self.ext_params.iter())
            .map(|param| {
                (param.clone(), {
                    let mut hasher = DefaultHasher::new();
                    (salt, param).hash(&mut hasher);
                    (hasher.finish() as u32).into()
                })
            })
            .collect();

        (cols, params, ext_params)
    }
}

impl Add for ExprVariables {
    type Output = Self;
    fn add(self, rhs: Self) -> Self {
        Self {
            cols: self.cols.union(&rhs.cols).cloned().collect(),
            params: self.params.union(&rhs.params).cloned().collect(),
            ext_params: self.ext_params.union(&rhs.ext_params).cloned().collect(),
        }
    }
}

impl BaseExpr {
    /// Evaluates a base field expression.
    /// Takes:
    ///     * `columns`: A mapping from triplets (interaction, idx, offset) to base field values.
    ///     * `vars`: A mapping from variable names to base field values.
    pub fn eval_expr<E, C, V>(&self, columns: &C, vars: &V) -> E::F
    where
        C: for<'a> Index<&'a (usize, usize, isize), Output = E::F>,
        V: for<'a> Index<&'a String, Output = E::F>,
        E: EvalAtRow,
    {
        match self {
            Self::Col(col) => columns[&(col.interaction, col.idx, col.offset)].clone(),
            Self::Const(c) => E::F::from(*c),
            Self::Param(var) => vars[&var.to_string()].clone(),
            Self::Add(a, b) => {
                a.eval_expr::<E, C, V>(columns, vars) + b.eval_expr::<E, C, V>(columns, vars)
            }
            Self::Sub(a, b) => {
                a.eval_expr::<E, C, V>(columns, vars) - b.eval_expr::<E, C, V>(columns, vars)
            }
            Self::Mul(a, b) => {
                a.eval_expr::<E, C, V>(columns, vars) * b.eval_expr::<E, C, V>(columns, vars)
            }
            Self::Neg(a) => -a.eval_expr::<E, C, V>(columns, vars),
            Self::Inv(a) => a.eval_expr::<E, C, V>(columns, vars).inverse(),
        }
    }

    pub fn collect_variables(&self) -> ExprVariables {
        match self {
            BaseExpr::Col(col) => ExprVariables::col(col.clone()),
            BaseExpr::Const(_) => ExprVariables::default(),
            BaseExpr::Param(param) => ExprVariables::param(param.to_string()),
            BaseExpr::Add(a, b) => a.collect_variables() + b.collect_variables(),
            BaseExpr::Sub(a, b) => a.collect_variables() + b.collect_variables(),
            BaseExpr::Mul(a, b) => a.collect_variables() + b.collect_variables(),
            BaseExpr::Neg(a) => a.collect_variables(),
            BaseExpr::Inv(a) => a.collect_variables(),
        }
    }

    pub fn random_eval(&self) -> BaseField {
        let assignment = self.collect_variables().random_assignment(0);
        assert!(assignment.2.is_empty());
        self.eval_expr::<AssertEvaluator<'_>, _, _>(&assignment.0, &assignment.1)
    }
}

impl ExtExpr {
    /// Evaluates an extension field expression.
    /// Takes:
    ///     * `columns`: A mapping from triplets (interaction, idx, offset) to base field values.
    ///     * `vars`: A mapping from variable names to base field values.
    ///     * `ext_vars`: A mapping from variable names to extension field values.
    pub fn eval_expr<E, C, V, EV>(&self, columns: &C, vars: &V, ext_vars: &EV) -> E::EF
    where
        C: for<'a> Index<&'a (usize, usize, isize), Output = E::F>,
        V: for<'a> Index<&'a String, Output = E::F>,
        EV: for<'a> Index<&'a String, Output = E::EF>,
        E: EvalAtRow,
    {
        match self {
            Self::SecureCol([a, b, c, d]) => {
                let a = a.eval_expr::<E, C, V>(columns, vars);
                let b = b.eval_expr::<E, C, V>(columns, vars);
                let c = c.eval_expr::<E, C, V>(columns, vars);
                let d = d.eval_expr::<E, C, V>(columns, vars);
                E::combine_ef([a, b, c, d])
            }
            Self::Const(c) => E::EF::from(*c),
            Self::Param(var) => ext_vars[&var.to_string()].clone(),
            Self::Add(a, b) => {
                a.eval_expr::<E, C, V, EV>(columns, vars, ext_vars)
                    + b.eval_expr::<E, C, V, EV>(columns, vars, ext_vars)
            }
            Self::Sub(a, b) => {
                a.eval_expr::<E, C, V, EV>(columns, vars, ext_vars)
                    - b.eval_expr::<E, C, V, EV>(columns, vars, ext_vars)
            }
            Self::Mul(a, b) => {
                a.eval_expr::<E, C, V, EV>(columns, vars, ext_vars)
                    * b.eval_expr::<E, C, V, EV>(columns, vars, ext_vars)
            }
            Self::Neg(a) => -a.eval_expr::<E, C, V, EV>(columns, vars, ext_vars),
        }
    }

    pub fn collect_variables(&self) -> ExprVariables {
        match self {
            ExtExpr::SecureCol([a, b, c, d]) => {
                a.collect_variables()
                    + b.collect_variables()
                    + c.collect_variables()
                    + d.collect_variables()
            }
            ExtExpr::Const(_) => ExprVariables::default(),
            ExtExpr::Param(param) => ExprVariables::ext_param(param.to_string()),
            ExtExpr::Add(a, b) => a.collect_variables() + b.collect_variables(),
            ExtExpr::Sub(a, b) => a.collect_variables() + b.collect_variables(),
            ExtExpr::Mul(a, b) => a.collect_variables() + b.collect_variables(),
            ExtExpr::Neg(a) => a.collect_variables(),
        }
    }

    pub fn random_eval(&self) -> SecureField {
        let assignment = self.collect_variables().random_assignment(0);
        self.eval_expr::<AssertEvaluator<'_>, _, _, _>(&assignment.0, &assignment.1, &assignment.2)
    }
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use num_traits::One;

    use crate::constraint_framework::expr::utils::*;
    use crate::constraint_framework::AssertEvaluator;
    use crate::core::fields::m31::BaseField;
    use crate::core::fields::qm31::SecureField;
    use crate::core::fields::FieldExpOps;

    #[test]
    fn test_eval_expr() {
        let col_1_0_0 = BaseField::from(12);
        let col_1_1_0 = BaseField::from(5);
        let var_a = BaseField::from(3);
        let var_b = BaseField::from(4);
        let var_c = SecureField::from_m31_array([
            BaseField::from(1),
            BaseField::from(2),
            BaseField::from(3),
            BaseField::from(4),
        ]);

        let columns: HashMap<(usize, usize, isize), BaseField> =
            HashMap::from([((1, 0, 0), col_1_0_0), ((1, 1, 0), col_1_1_0)]);
        let vars = HashMap::from([("a".to_string(), var_a), ("b".to_string(), var_b)]);
        let ext_vars = HashMap::from([("c".to_string(), var_c)]);

        let expr = secure_col!(
            col!(1, 0, 0) - col!(1, 1, 0),
            col!(1, 1, 0) * (-var!("a")),
            var!("a") + var!("a").inverse(),
            var!("b") * felt!(7)
        ) + qvar!("c") * qvar!("c")
            - qfelt!(1, 0, 0, 0);

        let expected = SecureField::from_m31_array([
            col_1_0_0 - col_1_1_0,
            col_1_1_0 * (-var_a),
            var_a + var_a.inverse(),
            var_b * BaseField::from(7),
        ]) + var_c * var_c
            - SecureField::one();

        assert_eq!(
            expr.eval_expr::<AssertEvaluator<'_>, _, _, _>(&columns, &vars, &ext_vars),
            expected
        );
    }
}
