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
pub enum OpParam {
    Int(i64),
    String(String),
    Bool(bool),
    List(Vec<OpParam>),
}
