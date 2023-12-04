use std::fmt::Display;

use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct GraphNode {
    pub name: String,
    pub description: String,
    pub size: u64,
    pub ty: String,
    pub op: String,
    pub params: Vec<OpParam>,
    pub inputs: Vec<String>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum Op {
    Input {
        name: String,
        ty: String,
    },
    Pad {
        input: String,
        prepend: Vec<String>,
        append: Vec<String>,
    },
    Slice {
        input: String,
        start: i64,
        end: i64,
        step: i64,
    },
    Interleave {
        inputs: Vec<String>,
    },
    Repeat {
        input: String,
        n: i64,
    },
    Pointwise {
        op: PointwiseOp,
        inputs: Vec<String>,
    },
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum PointwiseOp {
    Const {
        value: String,
        ty: String,
    },
    Unary {
        op: UnaryOp,
        a: String,
    },
    Binary {
        op: BinaryOp,
        a: String,
        b: String,
    },
    Custom {
        name: String,
        params: Vec<OpParam>,
        inputs: Vec<String>,
    },
}
impl PointwiseOp {
    pub fn name(&self) -> String {
        match self {
            PointwiseOp::Const { .. } => "const_val".to_string(),
            PointwiseOp::Unary { op, .. } => match op {
                UnaryOp::Neg => "neg".to_string(),
                UnaryOp::Not => "not".to_string(),
                UnaryOp::Sqrt => "sqrt".to_string(),
                UnaryOp::InvOrZero => "inv_or_zero".to_string(),
            },
            PointwiseOp::Binary { op, .. } => match op {
                BinaryOp::Add => "add".to_string(),
                BinaryOp::Sub => "sub".to_string(),
                BinaryOp::Mul => "mul".to_string(),
                BinaryOp::Div => "div".to_string(),
                BinaryOp::Pow => "pow".to_string(),
                BinaryOp::Eq => "eq".to_string(),
                BinaryOp::Neq => "neq".to_string(),
                BinaryOp::Lt => "lt".to_string(),
                BinaryOp::Gt => "gt".to_string(),
                BinaryOp::Leq => "leq".to_string(),
                BinaryOp::Geq => "geq".to_string(),
                BinaryOp::And => "and".to_string(),
                BinaryOp::Or => "or".to_string(),
                BinaryOp::Xor => "xor".to_string(),
                BinaryOp::Max => "max".to_string(),
                BinaryOp::Min => "min".to_string(),
            },
            PointwiseOp::Custom { name, .. } => name.clone(),
        }
    }
    pub fn params(&self) -> Vec<OpParam> {
        match self {
            PointwiseOp::Const { value, .. } => vec![OpParam::String(value.clone())],
            PointwiseOp::Unary { .. } => vec![],
            PointwiseOp::Binary { .. } => vec![],
            PointwiseOp::Custom { params, .. } => params.clone(),
        }
    }
    pub fn inputs(&self) -> Vec<String> {
        match self {
            PointwiseOp::Const { .. } => vec![],
            PointwiseOp::Unary { a, .. } => vec![a.clone()],
            PointwiseOp::Binary { a, b, .. } => vec![a.clone(), b.clone()],
            PointwiseOp::Custom { inputs, .. } => inputs.clone(),
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum BinaryOp {
    Add,
    Sub,
    Mul,
    Div,
    Pow,
    Eq,
    Neq,
    Lt,
    Gt,
    Leq,
    Geq,
    And,
    Or,
    Xor,
    Max,
    Min,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum UnaryOp {
    Neg,
    Not,
    Sqrt,
    InvOrZero,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum OpParam {
    Int(i64),
    String(String),
    Bool(bool),
    List(Vec<OpParam>),
}
impl Display for OpParam {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            OpParam::Int(i) => write!(f, "{}", i),
            OpParam::String(s) => write!(f, "{}", s),
            OpParam::Bool(b) => write!(f, "{}", b),
            OpParam::List(l) => {
                write!(f, "[")?;
                for (i, item) in l.iter().enumerate() {
                    if i > 0 {
                        write!(f, ", ")?;
                    }
                    write!(f, "{}", item)?;
                }
                write!(f, "]")
            }
        }
    }
}
