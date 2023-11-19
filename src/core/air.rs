use super::circle::CirclePointIndex;
use crate::core::constraints::PolyOracle;
use crate::core::fields::m31::BaseField;
use crate::core::poly::circle::CanonicCoset;

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

    // TODO (ShaharS), Consider moving this functions to somewhere else and change the API.
    pub fn eval(
        &self,
        cosets: &[CanonicCoset],
        poly_oracles: &[impl PolyOracle],
    ) -> Vec<BaseField> {
        let mut res = Vec::with_capacity(self.items.len());
        for item in &self.items {
            let point_index = cosets[item.column_index].index_at(item.offset);
            res.push(poly_oracles[item.column_index].get_at(point_index));
        }
        res
    }

    pub fn get_point_indices(&self, cosets: &[CanonicCoset]) -> Vec<CirclePointIndex> {
        let mut res = Vec::with_capacity(self.items.len());
        for item in &self.items {
            res.push(cosets[item.column_index].index_at(item.offset));
        }
        res
    }
}
