use serde::{Deserialize, Serialize};

use super::graph::PointwiseOp;

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct MaterializedGraph {
    pub inputs: Vec<MaterializedArray>,
    pub outputs: Vec<MaterializedArray>,
    // Topologically sorted.
    pub computations: Vec<MaterializedComputation>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct MaterializedArray {
    pub name: String,
    pub size: u64,
    pub ty: String,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct MaterializedComputation {
    pub output_tile: Vec<MaskItem>,
    pub input_tile: Vec<MaskItem>,
    pub n_repeats: u64,
    pub fused_op: FusedOp,
    pub ordering: Ordering,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum Ordering {
    Sequential,
    Parallel,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct MaskItem {
    pub item_name: String,
    pub array_name: String,
    pub offset: u64,
    pub step: u64,
    pub modulus: Option<u64>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct FusedOp {
    pub ops: Vec<FusedNode>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct FusedNode {
    pub name: String,
    pub op: PointwiseOp,
    pub ty: String,
}
