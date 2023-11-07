use crate::core::{circle::CirclePointIndex, constraints::PolyOracle, fields::m31::Field};

#[derive(Clone, Debug)]
pub struct Column {
    pub name: String,
    pub n_bits: usize,
}

pub struct MaskItem {
    pub col: Column,
    pub index: CirclePointIndex,
}

pub struct Mask {
    pub items: Vec<MaskItem>,
}

pub fn eval_mask_at_point(poly_oracle: impl PolyOracle, mask: &Mask) -> Vec<Field> {
    let mut res = Vec::with_capacity(mask.items.len());
    for item in mask.items.iter() {
        res.push(poly_oracle.get_at(item.index));
    }
    res
}
