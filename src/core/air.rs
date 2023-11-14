use crate::core::{constraints::PolyOracle, fields::m31::Field, poly::circle::CanonicCoset};

pub struct MaskItem {
    pub column_index: usize,
    pub offset: usize,
}

pub struct Mask {
    pub items: Vec<MaskItem>,
}

impl Mask {
    pub fn new(items: Vec<MaskItem>) -> Self {
        Self { items }
    }

    pub fn eval(&self, cosets: &[CanonicCoset], poly_oracle: &[impl PolyOracle]) -> Vec<Field> {
        let mut res = Vec::with_capacity(self.items.len());
        for item in &self.items {
            let point = cosets[item.column_index].index_at(item.offset);
            res.push(poly_oracle[item.column_index].get_at(point));
        }
        res
    }
}
