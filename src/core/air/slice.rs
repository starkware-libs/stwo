use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SliceDomain {
    pub inclusions: Vec<Slice>,
    pub exclusions: Vec<Slice>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Slice {
    pub offset: u64,
    pub log_steps: u64,
}
