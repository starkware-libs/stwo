use crate::core::{
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
    pub offset: usize,
}

pub struct Mask {
    pub items: Vec<MaskItem>,
}

pub fn eval_mask_at_point(
    poly_oracle: Vec<&EvalByPoly<'_>>,
    trace: &[Column],
    mask: &Mask,
) -> Vec<Field> {
    let mut res = Vec::with_capacity(mask.items.len());
    for item in mask.items.iter() {
        let point = trace[item.column_index].coset.index_at(item.offset);
        res.push(poly_oracle[item.column_index].get_at(point));
    }
    res
}
