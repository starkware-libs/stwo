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
    Pointwise {
        op: PointwiseOp,
        inputs: Vec<String>,
    },
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct PointwiseOp {
    pub name: String,
    pub params: Vec<OpParam>,
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
