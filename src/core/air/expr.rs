use serde::{Deserialize, Serialize};

use crate::core::fields::m31::BaseField;

// Univariate.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum UnivariateExprExtension {
    MaskItem(UnivariateMaskItem),
    X,
    Y,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct UnivariateMaskItem {
    pub column: String,
    pub log_expand: u64,
    pub offset: i64,
}

pub type UnivariatePolyExpression = Expression<UnivariateExprExtension>;

// Multivariate.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum MultivariateExprExtension {
    MaskItem {
        column: String,
        bit_indices: Vec<BitExpression>,
    },
}

pub type MultiVariatePolyExpression = Expression<MultivariateExprExtension>;

// Common.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum Expression<Extension> {
    Extension(Extension),
    Value(BaseField),
    BinaryOp {
        op: BinaryOp,
        lhs: Box<Expression<Extension>>,
        rhs: Box<Expression<Extension>>,
    },
    Pow {
        base: Box<Expression<Extension>>,
        exp: u64,
    },
    UnaryOp {
        op: UnaryOp,
        expr: Box<Expression<Extension>>,
    },
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum BinaryOp {
    Add,
    Sub,
    Mul,
    Div,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum UnaryOp {
    Neg,
}

// Bit expression.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum BitExpression {
    Value(bool),
    X {
        index: u64,
    },
    BinaryOp {
        op: BitBinaryOp,
        lhs: Box<BitExpression>,
        rhs: Box<BitExpression>,
    },
    UnaryOp {
        op: BitUnaryOp,
        expr: Box<BitExpression>,
    },
}
#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum BitBinaryOp {
    Add,
    Sub,
    Mul,
    Div,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum BitUnaryOp {
    Neg,
}
