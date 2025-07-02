pub mod assignment;
pub mod degree;
pub mod evaluator;
pub mod format;
pub mod simplify;
pub mod utils;

use std::ops::{Add, AddAssign, Mul, MulAssign, Neg, Sub};
use std::rc::Rc;

pub use evaluator::ExprEvaluator;
use num_traits::{One, Zero};
use stwo::core::fields::cm31::CM31;
use stwo::core::fields::m31::BaseField;
use stwo::core::fields::qm31::{SecureField, QM31};
use stwo::core::fields::FieldExpOps;

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
pub struct BaseExpr(Rc<BaseExprInner>);

#[derive(Clone, Debug, PartialEq)]
pub enum BaseExprInner {
    Col(ColumnExpr),
    Const(BaseField),
    /// Formal parameter to the AIR, for example the interaction elements of a relation.
    Param(String),
    Add(BaseExpr, BaseExpr),
    Sub(BaseExpr, BaseExpr),
    Mul(BaseExpr, BaseExpr),
    Neg(BaseExpr),
    Inv(BaseExpr),
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
    SecureCol([Rc<BaseExpr>; 4]),
    Const(SecureField),
    /// Formal parameter to the AIR, for example the interaction elements of a relation.
    Param(String),
    Add(Rc<ExtExpr>, Rc<ExtExpr>),
    Sub(Rc<ExtExpr>, Rc<ExtExpr>),
    Mul(Rc<ExtExpr>, Rc<ExtExpr>),
    Neg(Rc<ExtExpr>),
}

impl Into<BaseExpr> for BaseExprInner {
    fn into(self) -> BaseExpr {
        BaseExpr(Rc::new(self))
    }
}

impl From<BaseField> for BaseExpr {
    fn from(val: BaseField) -> Self {
        BaseExprInner::Const(val).into()
    }
}

impl From<BaseField> for ExtExpr {
    fn from(val: BaseField) -> Self {
        ExtExpr::SecureCol([
            Rc::new(BaseExpr::from(val)),
            Rc::new(BaseExpr::zero()),
            Rc::new(BaseExpr::zero()),
            Rc::new(BaseExpr::zero()),
        ])
    }
}

impl From<SecureField> for ExtExpr {
    fn from(QM31(CM31(a, b), CM31(c, d)): SecureField) -> Self {
        ExtExpr::SecureCol([
            Rc::new(BaseExpr::from(a)),
            Rc::new(BaseExpr::from(b)),
            Rc::new(BaseExpr::from(c)),
            Rc::new(BaseExpr::from(d)),
        ])
    }
}

impl From<BaseExpr> for ExtExpr {
    fn from(expr: BaseExpr) -> Self {
        ExtExpr::SecureCol([
            Rc::new(expr.clone()),
            Rc::new(BaseExpr::zero()),
            Rc::new(BaseExpr::zero()),
            Rc::new(BaseExpr::zero()),
        ])
    }
}

impl Add for BaseExpr {
    type Output = Self;
    fn add(self, rhs: Self) -> Self {
        BaseExprInner::Add(self, rhs).into()
    }
}

impl Sub for BaseExpr {
    type Output = Self;
    fn sub(self, rhs: Self) -> Self {
        BaseExprInner::Sub(self, rhs).into()
    }
}

impl Mul for BaseExpr {
    type Output = Self;
    fn mul(self, rhs: Self) -> Self {
        BaseExprInner::Mul(self, rhs).into()
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
        BaseExprInner::Neg(self).into()
    }
}

impl Add for ExtExpr {
    type Output = Self;
    fn add(self, rhs: Self) -> Self {
        ExtExpr::Add(Rc::new(self), Rc::new(rhs))
    }
}

impl Sub for ExtExpr {
    type Output = Self;
    fn sub(self, rhs: Self) -> Self {
        ExtExpr::Sub(Rc::new(self), Rc::new(rhs))
    }
}

impl Mul for ExtExpr {
    type Output = Self;
    fn mul(self, rhs: Self) -> Self {
        ExtExpr::Mul(Rc::new(self), Rc::new(rhs))
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
        ExtExpr::Neg(Rc::new(self))
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
        BaseExprInner::Inv(self.clone()).into()
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
