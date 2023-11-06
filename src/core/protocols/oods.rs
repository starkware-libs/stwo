use crate::core::{
    circle::CirclePointIndex,
    constraints::{EvalByPoly, PolyOracle},
    fields::m31::Field,
    poly::circle::CanonicCoset,
};

#[derive(Clone, Debug)]
pub struct Column {
    pub name: String,
    pub coset: CanonicCoset,
}

pub struct MaskItem {
    pub column_index: usize,
    pub index: CirclePointIndex,
}

pub struct Mask {
    pub items: Vec<MaskItem>,
}

pub fn eval_mask_at_point(poly_oracle: &EvalByPoly<'_>, mask: &Mask) -> Vec<Field> {
    let mut res = Vec::with_capacity(mask.items.len());
    for item in mask.items.iter() {
        res.push(poly_oracle.get_at(item.index));
    }
    res
}
