use std::collections::BTreeMap;

use super::circle::CirclePointIndex;
use super::fields::m31::Field;
use super::poly::circle::PointSetEvaluation;
use crate::core::constraints::PolyOracle;
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
    pub fn get_evaluation(
        &self,
        mask_offsets: &[CirclePointIndex],
        poly_oracles: &[impl PolyOracle],
        evaluation_point: CirclePointIndex,
    ) -> PointSetEvaluation {
        let mut res: BTreeMap<CirclePointIndex, Field> = BTreeMap::new();
        for (mask_item, mask_offset) in self.items.iter().zip(mask_offsets) {
            let point = evaluation_point + *mask_offset;
            res.insert(point, poly_oracles[mask_item.column_index].get_at(point));
            res.insert(-point, poly_oracles[mask_item.column_index].get_at(-point));
        }
        PointSetEvaluation::new(res)
    }

    pub fn get_point_indices(&self, cosets: &[CanonicCoset]) -> Vec<CirclePointIndex> {
        let mut res = Vec::with_capacity(self.items.len());
        for item in &self.items {
            res.push(cosets[item.column_index].index_at(item.offset));
        }
        res
    }
}
