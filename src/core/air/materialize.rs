use super::graph::PointwiseOp;

pub struct MaterializedGraph {
    pub inputs: Vec<MaterializedArray>,
    pub outputs: Vec<MaterializedArray>,
    // Topologically sorted.
    pub computations: Vec<MaterializedComputation>,
}

pub struct MaterializedArray {
    pub name: String,
    pub size: u64,
    pub ty: String,
}

pub struct MaterializedComputation {
    pub output_tile: Vec<MaskItem>,
    pub input_tile: Vec<MaskItem>,
    pub n_repeats: u64,
    pub fused_op: FusedOp,
    pub ordering: Ordering,
}

pub enum Ordering {
    Sequential,
    Parallel,
}

pub struct MaskItem {
    pub item_name: String,
    pub array_name: String,
    pub offset: u64,
    pub step: u64,
    pub modulus: Option<u64>,
}

pub struct FusedOp {
    pub ops: Vec<FusedNode>,
}

pub struct FusedNode {
    pub name: String,
    pub op: PointwiseOp,
    pub ty: String,
}
