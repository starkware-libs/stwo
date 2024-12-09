use std::collections::{HashMap, HashSet};
use std::hash::{DefaultHasher, Hash, Hasher};
use std::ops::{Add, AddAssign, Index, Mul, MulAssign, Neg, Sub};

use itertools::sorted;
use num_traits::{One, Zero};

use super::preprocessed_columns::PreprocessedColumn;
use super::{AssertEvaluator, EvalAtRow, Relation, RelationEntry, INTERACTION_TRACE_IDX};
use crate::core::fields::cm31::CM31;
use crate::core::fields::m31::{self, BaseField};
use crate::core::fields::qm31::{SecureField, QM31};
use crate::core::fields::FieldExpOps;
use crate::core::lookups::utils::Fraction;

/// A single base field column at index `idx` of interaction `interaction`, at mask offset `offset`.
#[derive(Clone, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct ColumnExpr {
    interaction: usize,
    idx: usize,
    offset: isize,
}

impl From<(usize, usize, isize)> for ColumnExpr {
    fn from((interaction, idx, offset): (usize, usize, isize)) -> Self {
        Self {
            interaction,
            idx,
            offset,
        }
    }
}

/// An expression representing a base field value. Can be either:
///     * A column indexed by a `ColumnExpr`.
///     * A base field constant.
///     * A formal parameter to the AIR.
///     * A sum, difference, or product of two base field expressions.
///     * A negation or inverse of a base field expression.
///
/// This type is meant to be used as an F associated type for EvalAtRow and interacts with
/// `ExtExpr`, `BaseField` and `SecureField` as expected.
#[derive(Clone, Debug, PartialEq)]
pub enum BaseExpr {
    Col(ColumnExpr),
    Const(BaseField),
    /// Formal parameter to the AIR, for example the interaction elements of a relation.
    Param(String),
    Add(Box<BaseExpr>, Box<BaseExpr>),
    Sub(Box<BaseExpr>, Box<BaseExpr>),
    Mul(Box<BaseExpr>, Box<BaseExpr>),
    Neg(Box<BaseExpr>),
    Inv(Box<BaseExpr>),
}

/// An expression representing a secure field value. Can be either:
///     * A secure column constructed from 4 base field expressions.
///     * A secure field constant.
///     * A formal parameter to the AIR.
///     * A sum, difference, or product of two secure field expressions.
///     * A negation of a secure field expression.
///
/// This type is meant to be used as an EF associated type for EvalAtRow and interacts with
/// `BaseExpr`, `BaseField` and `SecureField` as expected.
#[derive(Clone, Debug, PartialEq)]
pub enum ExtExpr {
    /// An atomic secure column constructed from 4 expressions.
    /// Expressions on the secure column are not reduced, i.e,
    /// if `a = SecureCol(a0, a1, a2, a3)`, `b = SecureCol(b0, b1, b2, b3)` then
    /// `a + b` evaluates to `Add(a, b)` rather than
    /// `SecureCol(Add(a0, b0), Add(a1, b1), Add(a2, b2), Add(a3, b3))`
    SecureCol([Box<BaseExpr>; 4]),
    Const(SecureField),
    /// Formal parameter to the AIR, for example the interaction elements of a relation.
    Param(String),
    Add(Box<ExtExpr>, Box<ExtExpr>),
    Sub(Box<ExtExpr>, Box<ExtExpr>),
    Mul(Box<ExtExpr>, Box<ExtExpr>),
    Neg(Box<ExtExpr>),
}

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
    pub fn format_expr(&self) -> String {
        match self {
            BaseExpr::Col(ColumnExpr {
                interaction,
                idx,
                offset,
            }) => {
                let offset_str = if *offset == CLAIMED_SUM_DUMMY_OFFSET as isize {
                    "claimed_sum_offset".to_string()
                } else {
                    offset.to_string()
                };
                format!("col_{interaction}_{idx}[{offset_str}]")
            }
            BaseExpr::Const(c) => c.to_string(),
            BaseExpr::Param(v) => v.to_string(),
            BaseExpr::Add(a, b) => format!("{} + {}", a.format_expr(), b.format_expr()),
            BaseExpr::Sub(a, b) => format!("{} - ({})", a.format_expr(), b.format_expr()),
            BaseExpr::Mul(a, b) => format!("({}) * ({})", a.format_expr(), b.format_expr()),
            BaseExpr::Neg(a) => format!("-({})", a.format_expr()),
            BaseExpr::Inv(a) => format!("1 / ({})", a.format_expr()),
        }
    }

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
    pub fn format_expr(&self) -> String {
        match self {
            ExtExpr::SecureCol([a, b, c, d]) => {
                // If the expression's non-base components are all constant zeroes, return the base
                // field representation of its first part.
                if **b == BaseExpr::zero() && **c == BaseExpr::zero() && **d == BaseExpr::zero() {
                    a.format_expr()
                } else {
                    format!(
                        "SecureCol({}, {}, {}, {})",
                        a.format_expr(),
                        b.format_expr(),
                        c.format_expr(),
                        d.format_expr()
                    )
                }
            }
            ExtExpr::Const(c) => {
                if c.0 .1.is_zero() && c.1 .0.is_zero() && c.1 .1.is_zero() {
                    // If the constant is in the base field, display it as such.
                    c.0 .0.to_string()
                } else {
                    c.to_string()
                }
            }
            ExtExpr::Param(v) => v.to_string(),
            ExtExpr::Add(a, b) => format!("{} + {}", a.format_expr(), b.format_expr()),
            ExtExpr::Sub(a, b) => format!("{} - ({})", a.format_expr(), b.format_expr()),
            ExtExpr::Mul(a, b) => format!("({}) * ({})", a.format_expr(), b.format_expr()),
            ExtExpr::Neg(a) => format!("-({})", a.format_expr()),
        }
    }

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
    /// Note that the assignment is deterministically dependent on every variable and that this is
    /// required.
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

impl From<BaseField> for BaseExpr {
    fn from(val: BaseField) -> Self {
        BaseExpr::Const(val)
    }
}

impl From<BaseField> for ExtExpr {
    fn from(val: BaseField) -> Self {
        ExtExpr::SecureCol([
            Box::new(BaseExpr::from(val)),
            Box::new(BaseExpr::zero()),
            Box::new(BaseExpr::zero()),
            Box::new(BaseExpr::zero()),
        ])
    }
}

impl From<SecureField> for ExtExpr {
    fn from(QM31(CM31(a, b), CM31(c, d)): SecureField) -> Self {
        ExtExpr::SecureCol([
            Box::new(BaseExpr::from(a)),
            Box::new(BaseExpr::from(b)),
            Box::new(BaseExpr::from(c)),
            Box::new(BaseExpr::from(d)),
        ])
    }
}

impl From<BaseExpr> for ExtExpr {
    fn from(expr: BaseExpr) -> Self {
        ExtExpr::SecureCol([
            Box::new(expr.clone()),
            Box::new(BaseExpr::zero()),
            Box::new(BaseExpr::zero()),
            Box::new(BaseExpr::zero()),
        ])
    }
}

impl Add for BaseExpr {
    type Output = Self;
    fn add(self, rhs: Self) -> Self {
        BaseExpr::Add(Box::new(self), Box::new(rhs))
    }
}

impl Sub for BaseExpr {
    type Output = Self;
    fn sub(self, rhs: Self) -> Self {
        BaseExpr::Sub(Box::new(self), Box::new(rhs))
    }
}

impl Mul for BaseExpr {
    type Output = Self;
    fn mul(self, rhs: Self) -> Self {
        BaseExpr::Mul(Box::new(self), Box::new(rhs))
    }
}

impl AddAssign for BaseExpr {
    fn add_assign(&mut self, rhs: Self) {
        *self = self.clone() + rhs
    }
}

impl MulAssign for BaseExpr {
    fn mul_assign(&mut self, rhs: Self) {
        *self = self.clone() * rhs
    }
}

impl Neg for BaseExpr {
    type Output = Self;
    fn neg(self) -> Self {
        BaseExpr::Neg(Box::new(self))
    }
}

impl Add for ExtExpr {
    type Output = Self;
    fn add(self, rhs: Self) -> Self {
        ExtExpr::Add(Box::new(self), Box::new(rhs))
    }
}

impl Sub for ExtExpr {
    type Output = Self;
    fn sub(self, rhs: Self) -> Self {
        ExtExpr::Sub(Box::new(self), Box::new(rhs))
    }
}

impl Mul for ExtExpr {
    type Output = Self;
    fn mul(self, rhs: Self) -> Self {
        ExtExpr::Mul(Box::new(self), Box::new(rhs))
    }
}

impl AddAssign for ExtExpr {
    fn add_assign(&mut self, rhs: Self) {
        *self = self.clone() + rhs
    }
}

impl MulAssign for ExtExpr {
    fn mul_assign(&mut self, rhs: Self) {
        *self = self.clone() * rhs
    }
}

impl Neg for ExtExpr {
    type Output = Self;
    fn neg(self) -> Self {
        ExtExpr::Neg(Box::new(self))
    }
}

impl Zero for BaseExpr {
    fn zero() -> Self {
        BaseExpr::from(BaseField::zero())
    }
    fn is_zero(&self) -> bool {
        // TODO(alont): consider replacing `Zero` in the trait bound with a custom trait
        // that only has `zero()`.
        panic!("Can't check if an expression is zero.");
    }
}

impl One for BaseExpr {
    fn one() -> Self {
        BaseExpr::from(BaseField::one())
    }
}

impl Zero for ExtExpr {
    fn zero() -> Self {
        ExtExpr::from(BaseField::zero())
    }
    fn is_zero(&self) -> bool {
        // TODO(alont): consider replacing `Zero` in the trait bound with a custom trait
        // that only has `zero()`.
        panic!("Can't check if an expression is zero.");
    }
}

impl One for ExtExpr {
    fn one() -> Self {
        ExtExpr::from(BaseField::one())
    }
}

impl FieldExpOps for BaseExpr {
    fn inverse(&self) -> Self {
        BaseExpr::Inv(Box::new(self.clone()))
    }
}

impl Add<BaseField> for BaseExpr {
    type Output = Self;
    fn add(self, rhs: BaseField) -> Self {
        self + BaseExpr::from(rhs)
    }
}

impl AddAssign<BaseField> for BaseExpr {
    fn add_assign(&mut self, rhs: BaseField) {
        *self = self.clone() + BaseExpr::from(rhs)
    }
}

impl Mul<BaseField> for BaseExpr {
    type Output = Self;
    fn mul(self, rhs: BaseField) -> Self {
        self * BaseExpr::from(rhs)
    }
}

impl Mul<SecureField> for BaseExpr {
    type Output = ExtExpr;
    fn mul(self, rhs: SecureField) -> ExtExpr {
        ExtExpr::from(self) * ExtExpr::from(rhs)
    }
}

impl Add<SecureField> for BaseExpr {
    type Output = ExtExpr;
    fn add(self, rhs: SecureField) -> ExtExpr {
        ExtExpr::from(self) + ExtExpr::from(rhs)
    }
}

impl Sub<SecureField> for BaseExpr {
    type Output = ExtExpr;
    fn sub(self, rhs: SecureField) -> ExtExpr {
        ExtExpr::from(self) - ExtExpr::from(rhs)
    }
}

impl Add<BaseField> for ExtExpr {
    type Output = Self;
    fn add(self, rhs: BaseField) -> Self {
        self + ExtExpr::from(rhs)
    }
}

impl AddAssign<BaseField> for ExtExpr {
    fn add_assign(&mut self, rhs: BaseField) {
        *self = self.clone() + ExtExpr::from(rhs)
    }
}

impl Mul<BaseField> for ExtExpr {
    type Output = Self;
    fn mul(self, rhs: BaseField) -> Self {
        self * ExtExpr::from(rhs)
    }
}

impl Mul<SecureField> for ExtExpr {
    type Output = Self;
    fn mul(self, rhs: SecureField) -> Self {
        self * ExtExpr::from(rhs)
    }
}

impl Add<SecureField> for ExtExpr {
    type Output = Self;
    fn add(self, rhs: SecureField) -> Self {
        self + ExtExpr::from(rhs)
    }
}

impl Sub<SecureField> for ExtExpr {
    type Output = Self;
    fn sub(self, rhs: SecureField) -> Self {
        self - ExtExpr::from(rhs)
    }
}

impl Add<BaseExpr> for ExtExpr {
    type Output = Self;
    fn add(self, rhs: BaseExpr) -> Self {
        self + ExtExpr::from(rhs)
    }
}

impl Mul<BaseExpr> for ExtExpr {
    type Output = Self;
    fn mul(self, rhs: BaseExpr) -> Self {
        self * ExtExpr::from(rhs)
    }
}

impl Mul<ExtExpr> for BaseExpr {
    type Output = ExtExpr;
    fn mul(self, rhs: ExtExpr) -> ExtExpr {
        rhs * self
    }
}

impl Sub<BaseExpr> for ExtExpr {
    type Output = Self;
    fn sub(self, rhs: BaseExpr) -> Self {
        self - ExtExpr::from(rhs)
    }
}

/// Returns the expression
/// `value[0] * <relation>_alpha0 + value[1] * <relation>_alpha1 + ... - <relation>_z.`
fn combine_formal<R: Relation<BaseExpr, ExtExpr>>(relation: &R, values: &[BaseExpr]) -> ExtExpr {
    const Z_SUFFIX: &str = "_z";
    const ALPHA_SUFFIX: &str = "_alpha";

    let z = ExtExpr::Param(relation.get_name().to_owned() + Z_SUFFIX);
    let alpha_powers = (0..relation.get_size())
        .map(|i| ExtExpr::Param(relation.get_name().to_owned() + ALPHA_SUFFIX + &i.to_string()));
    values
        .iter()
        .zip(alpha_powers)
        .fold(ExtExpr::zero(), |acc, (value, power)| {
            acc + power * value.clone()
        })
        - z
}

pub struct FormalLogupAtRow {
    pub interaction: usize,
    pub total_sum: ExtExpr,
    pub claimed_sum: Option<(ExtExpr, usize)>,
    pub fracs: Vec<Fraction<ExtExpr, ExtExpr>>,
    pub is_finalized: bool,
    pub is_first: BaseExpr,
    pub log_size: u32,
}

// P is an offset no column can reach, it signifies the variable
// offset, which is an input to the verifier.
const CLAIMED_SUM_DUMMY_OFFSET: usize = m31::P as usize;

impl FormalLogupAtRow {
    pub fn new(interaction: usize, has_partial_sum: bool, log_size: u32) -> Self {
        let total_sum_name = "total_sum".to_string();
        let claimed_sum_name = "claimed_sum".to_string();

        Self {
            interaction,
            // TODO(alont): Should these be Expr::SecureField?
            total_sum: ExtExpr::Param(total_sum_name),
            claimed_sum: has_partial_sum
                .then_some((ExtExpr::Param(claimed_sum_name), CLAIMED_SUM_DUMMY_OFFSET)),
            fracs: vec![],
            is_finalized: true,
            is_first: BaseExpr::zero(),
            log_size,
        }
    }
}

/// An Evaluator that saves all constraint expressions.
pub struct ExprEvaluator {
    pub cur_var_index: usize,
    pub constraints: Vec<ExtExpr>,
    pub logup: FormalLogupAtRow,
    pub intermediates: Vec<(String, BaseExpr)>,
    pub ext_intermediates: Vec<(String, ExtExpr)>,
}

impl ExprEvaluator {
    pub fn new(log_size: u32, has_partial_sum: bool) -> Self {
        Self {
            cur_var_index: Default::default(),
            constraints: Default::default(),
            logup: FormalLogupAtRow::new(INTERACTION_TRACE_IDX, has_partial_sum, log_size),
            intermediates: vec![],
            ext_intermediates: vec![],
        }
    }

    pub fn format_constraints(&self) -> String {
        let lets_string = self
            .intermediates
            .iter()
            .map(|(name, expr)| format!("let {} = {};", name, expr.simplify_and_format()))
            .collect::<Vec<String>>()
            .join("\n\n");

        let secure_lets_string = self
            .ext_intermediates
            .iter()
            .map(|(name, expr)| format!("let {} = {};", name, expr.simplify_and_format()))
            .collect::<Vec<String>>()
            .join("\n\n");

        let constraints_str = self
            .constraints
            .iter()
            .enumerate()
            .map(|(i, c)| format!("let constraint_{i} = ") + &c.simplify_and_format() + ";")
            .collect::<Vec<String>>()
            .join("\n\n");

        [lets_string, secure_lets_string, constraints_str]
            .iter()
            .filter(|x| !x.is_empty())
            .cloned()
            .collect::<Vec<_>>()
            .join("\n\n")
    }
}

impl EvalAtRow for ExprEvaluator {
    // TODO(alont): Should there be a version of this that disallows Secure fields for F?
    type F = BaseExpr;
    type EF = ExtExpr;

    fn next_interaction_mask<const N: usize>(
        &mut self,
        interaction: usize,
        offsets: [isize; N],
    ) -> [Self::F; N] {
        let res = std::array::from_fn(|i| {
            let col = ColumnExpr::from((interaction, self.cur_var_index, offsets[i]));
            BaseExpr::Col(col)
        });
        self.cur_var_index += 1;
        res
    }

    fn add_constraint<G>(&mut self, constraint: G)
    where
        Self::EF: From<G>,
    {
        self.constraints.push(constraint.into());
    }

    fn combine_ef(values: [Self::F; 4]) -> Self::EF {
        ExtExpr::SecureCol([
            Box::new(values[0].clone()),
            Box::new(values[1].clone()),
            Box::new(values[2].clone()),
            Box::new(values[3].clone()),
        ])
    }

    fn add_to_relation<R: Relation<Self::F, Self::EF>>(
        &mut self,
        entry: RelationEntry<'_, Self::F, Self::EF, R>,
    ) {
        let intermediate =
            self.add_extension_intermediate(combine_formal(entry.relation, entry.values));
        let frac = Fraction::new(entry.multiplicity.clone(), intermediate);
        self.write_logup_frac(frac);
    }

    fn add_intermediate(&mut self, expr: Self::F) -> Self::F {
        let name = format!(
            "intermediate{}",
            self.intermediates.len() + self.ext_intermediates.len()
        );
        let intermediate = BaseExpr::Param(name.clone());
        self.intermediates.push((name, expr));
        intermediate
    }

    fn add_extension_intermediate(&mut self, expr: Self::EF) -> Self::EF {
        let name = format!(
            "intermediate{}",
            self.intermediates.len() + self.ext_intermediates.len()
        );
        let intermediate = ExtExpr::Param(name.clone());
        self.ext_intermediates.push((name, expr));
        intermediate
    }

    fn get_preprocessed_column(&mut self, column: PreprocessedColumn) -> Self::F {
        BaseExpr::Param(column.name().to_string())
    }

    super::logup_proxy!();
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use num_traits::One;
    use rand::rngs::SmallRng;
    use rand::{Rng, SeedableRng};

    use super::{BaseExpr, ExtExpr};
    use crate::constraint_framework::expr::ExprEvaluator;
    use crate::constraint_framework::{
        relation, AssertEvaluator, EvalAtRow, FrameworkEval, RelationEntry,
    };
    use crate::core::fields::m31::{self, BaseField};
    use crate::core::fields::qm31::SecureField;
    use crate::core::fields::FieldExpOps;

    macro_rules! secure_col {
        ($a:expr, $b:expr, $c:expr, $d:expr) => {
            ExtExpr::SecureCol([
                Box::new($a.into()),
                Box::new($b.into()),
                Box::new($c.into()),
                Box::new($d.into()),
            ])
        };
    }

    macro_rules! col {
        ($interaction:expr, $idx:expr, $offset:expr) => {
            BaseExpr::Col(($interaction, $idx, $offset).into())
        };
    }

    macro_rules! var {
        ($var:expr) => {
            BaseExpr::Param($var.to_string())
        };
    }

    macro_rules! qvar {
        ($var:expr) => {
            ExtExpr::Param($var.to_string())
        };
    }

    macro_rules! felt {
        ($val:expr) => {
            BaseExpr::Const($val.into())
        };
    }

    macro_rules! qfelt {
        ($a:expr, $b:expr, $c:expr, $d:expr) => {
            ExtExpr::Const(SecureField::from_m31_array([
                $a.into(),
                $b.into(),
                $c.into(),
                $d.into(),
            ]))
        };
    }

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
        let minus_one = felt!(m31::P - 1);
        let qminus_one = qfelt!(m31::P - 1, 0, 0, 0);

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

    #[test]
    fn test_format_expr() {
        let test_struct = TestStruct {};
        let eval = test_struct.evaluate(ExprEvaluator::new(16, false));
        let expected = "let intermediate0 = (col_1_1[0]) * (col_1_2[0]);

\
        let intermediate1 = (TestRelation_alpha0) * (col_1_0[0]) \
            + (TestRelation_alpha1) * (col_1_1[0]) \
            + (TestRelation_alpha2) * (col_1_2[0]) \
            - (TestRelation_z);

\
        let constraint_0 = ((col_1_0[0]) * (intermediate0)) * (1 / (col_1_0[0] + col_1_1[0]));

\
        let constraint_1 = (SecureCol(col_2_3[0], col_2_4[0], col_2_5[0], col_2_6[0]) \
            - (SecureCol(col_2_3[-1], col_2_4[-1], col_2_5[-1], col_2_6[-1]) \
                - ((total_sum) * (preprocessed.is_first)))) \
            * (intermediate1) \
            - (1);"
            .to_string();

        assert_eq!(eval.format_constraints(), expected);
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
            let intermediate = eval.add_intermediate(x1.clone() * x2.clone());
            eval.add_constraint(x0.clone() * intermediate * (x0.clone() + x1.clone()).inverse());
            eval.add_to_relation(RelationEntry::new(
                &TestRelation::dummy(),
                E::EF::one(),
                &[x0, x1, x2],
            ));
            eval.finalize_logup();
            eval
        }
    }
}
