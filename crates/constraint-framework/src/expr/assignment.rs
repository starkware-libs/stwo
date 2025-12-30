use std::collections::{HashMap, HashSet};
use std::hash::{DefaultHasher, Hash, Hasher};
use std::iter::Sum;
use std::ops::{Add, AddAssign, Index, Sub};

use itertools::sorted;
use stwo::core::fields::m31::BaseField;
use stwo::core::fields::qm31::SecureField;
use stwo::core::fields::FieldExpOps;

use super::{BaseExpr, BaseExprNode, ColumnExpr, ExtExpr, ExtExprNode};
use crate::{AssertEvaluator, EvalAtRow};

/// An assignment to the variables that may appear in an expression.
/// Maps are:
///     columns: (interaction, index, offset) -> value
///     base field expressions: name -> value
///     extension field expressions: name -> extension field value
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

impl AddAssign for ExprVariables {
    fn add_assign(&mut self, rhs: Self) {
        self.cols = self.cols.union(&rhs.cols).cloned().collect();
        self.params = self.params.union(&rhs.params).cloned().collect();
        self.ext_params = self.ext_params.union(&rhs.ext_params).cloned().collect();
    }
}

impl Sum for ExprVariables {
    fn sum<I: Iterator<Item = Self>>(iter: I) -> Self {
        iter.fold(Self::default(), |acc, x| acc + x)
    }
}

impl Sub for ExprVariables {
    type Output = Self;
    fn sub(self, rhs: Self) -> Self {
        Self {
            cols: &self.cols - &rhs.cols,
            params: &self.params - &rhs.params,
            ext_params: &self.ext_params - &rhs.ext_params,
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
        E::F: Clone,
    {
        let node = self.node();
        match node {
            BaseExprNode::Col(col) => columns[&(col.interaction, col.idx, col.offset)].clone(),
            BaseExprNode::Const(c) => E::F::from(c),
            BaseExprNode::Param(var) => {
                let var_str = var.as_str();
                vars[&var_str].clone()
            }
            BaseExprNode::Add(a, b) => {
                a.eval_expr::<E, C, V>(columns, vars) + b.eval_expr::<E, C, V>(columns, vars)
            }
            BaseExprNode::Sub(a, b) => {
                a.eval_expr::<E, C, V>(columns, vars) - b.eval_expr::<E, C, V>(columns, vars)
            }
            BaseExprNode::Mul(a, b) => {
                a.eval_expr::<E, C, V>(columns, vars) * b.eval_expr::<E, C, V>(columns, vars)
            }
            BaseExprNode::Neg(a) => -a.eval_expr::<E, C, V>(columns, vars),
            BaseExprNode::Inv(a) => a.eval_expr::<E, C, V>(columns, vars).inverse(),
        }
    }

    pub fn collect_variables(&self) -> ExprVariables {
        let node = self.node();
        match node {
            BaseExprNode::Col(col) => ExprVariables::col(col),
            BaseExprNode::Const(_) => ExprVariables::default(),
            BaseExprNode::Param(param) => ExprVariables::param(param.as_str()),
            BaseExprNode::Add(a, b) => a.collect_variables() + b.collect_variables(),
            BaseExprNode::Sub(a, b) => a.collect_variables() + b.collect_variables(),
            BaseExprNode::Mul(a, b) => a.collect_variables() + b.collect_variables(),
            BaseExprNode::Neg(a) => a.collect_variables(),
            BaseExprNode::Inv(a) => a.collect_variables(),
        }
    }

    pub fn assign(&self, assignment: &ExprVarAssignment) -> BaseField {
        self.eval_expr::<AssertEvaluator<'_>, _, _>(&assignment.0, &assignment.1)
    }

    pub fn random_eval(&self) -> BaseField {
        let assignment = self.collect_variables().random_assignment(0);
        assert!(assignment.2.is_empty());
        self.assign(&assignment)
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
        E::F: Clone,
        E::EF: Clone,
    {
        let node = self.node();
        match node {
            ExtExprNode::SecureCol([a, b, c, d]) => {
                let a = a.eval_expr::<E, C, V>(columns, vars);
                let b = b.eval_expr::<E, C, V>(columns, vars);
                let c = c.eval_expr::<E, C, V>(columns, vars);
                let d = d.eval_expr::<E, C, V>(columns, vars);
                E::combine_ef([a, b, c, d])
            }
            ExtExprNode::Const(c) => E::EF::from(c),
            ExtExprNode::Param(var) => {
                let var_str = var.as_str();
                ext_vars[&var_str].clone()
            }
            ExtExprNode::Add(a, b) => {
                a.eval_expr::<E, C, V, EV>(columns, vars, ext_vars)
                    + b.eval_expr::<E, C, V, EV>(columns, vars, ext_vars)
            }
            ExtExprNode::Sub(a, b) => {
                a.eval_expr::<E, C, V, EV>(columns, vars, ext_vars)
                    - b.eval_expr::<E, C, V, EV>(columns, vars, ext_vars)
            }
            ExtExprNode::Mul(a, b) => {
                a.eval_expr::<E, C, V, EV>(columns, vars, ext_vars)
                    * b.eval_expr::<E, C, V, EV>(columns, vars, ext_vars)
            }
            ExtExprNode::Neg(a) => -a.eval_expr::<E, C, V, EV>(columns, vars, ext_vars),
        }
    }

    pub fn collect_variables(&self) -> ExprVariables {
        let node = self.node();
        match node {
            ExtExprNode::SecureCol([a, b, c, d]) => {
                a.collect_variables()
                    + b.collect_variables()
                    + c.collect_variables()
                    + d.collect_variables()
            }
            ExtExprNode::Const(_) => ExprVariables::default(),
            ExtExprNode::Param(param) => ExprVariables::ext_param(param.as_str()),
            ExtExprNode::Add(a, b) => a.collect_variables() + b.collect_variables(),
            ExtExprNode::Sub(a, b) => a.collect_variables() + b.collect_variables(),
            ExtExprNode::Mul(a, b) => a.collect_variables() + b.collect_variables(),
            ExtExprNode::Neg(a) => a.collect_variables(),
        }
    }

    pub fn assign(&self, assignment: &ExprVarAssignment) -> SecureField {
        self.eval_expr::<AssertEvaluator<'_>, _, _, _>(&assignment.0, &assignment.1, &assignment.2)
    }

    pub fn random_eval(&self) -> SecureField {
        self.assign(&self.collect_variables().random_assignment(0))
    }
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use num_traits::One;
    use stwo::core::fields::m31::BaseField;
    use stwo::core::fields::qm31::SecureField;
    use stwo::core::fields::FieldExpOps;

    use crate::expr::utils::*;
    use crate::expr::init_arena;
    use crate::AssertEvaluator;

    #[test]
    fn test_eval_expr() {
        init_arena();

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
