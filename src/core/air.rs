use std::collections::BTreeMap;

use super::circle::CirclePointIndex;
use super::fields::m31::BaseField;
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
        cosets: &[CanonicCoset],
        poly_oracles: &[impl PolyOracle],
        evaluation_point: CirclePointIndex,
    ) -> PointSetEvaluation {
        let mut res: BTreeMap<CirclePointIndex, BaseField> = BTreeMap::new();
        let mask_offsets = self.get_point_indices(cosets);
        for (mask_item, mask_offset) in self.items.iter().zip(mask_offsets) {
            let point = evaluation_point + mask_offset;
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

#[cfg(test)]
mod tests {
    use crate::core::air::{Mask, MaskItem};
    use crate::core::circle::{CirclePoint, CirclePointIndex};
    use crate::core::constraints::EvalByPoly;
    use crate::core::fields::m31::BaseField;
    use crate::core::poly::circle::{CanonicCoset, CircleEvaluation, Evaluation};

    #[test]
    fn test_mask() {
        let coset = CanonicCoset::new(3);
        let trace = CircleEvaluation::new_canonical_ordered(
            coset,
            (0..8).map(BaseField::from_u32_unchecked).collect(),
        );
        let trace_poly = trace.interpolate();
        let z_index = CirclePointIndex::generator() * 17;

        let mask = Mask::new(
            (0..3)
                .map(|i| MaskItem {
                    column_index: 0,
                    offset: i,
                })
                .collect(),
        );
        let mask_points = mask.get_point_indices(&[coset]);
        let oods_evaluation = mask.get_evaluation(
            &[coset],
            &[EvalByPoly {
                point: CirclePoint::zero(),
                poly: &trace_poly,
            }],
            z_index,
        );

        assert_eq!(mask.items[0].column_index, 0);
        assert_eq!(mask_points.len() * 2, oods_evaluation.len());
        for mask_point in mask_points {
            let point_index = mask_point + z_index;
            let value = oods_evaluation.get_at(point_index);
            let conjugate_value = oods_evaluation.get_at(-point_index);

            assert_eq!(value, trace_poly.eval_at_point(point_index.to_point()));
            assert_eq!(
                conjugate_value,
                trace_poly.eval_at_point(-point_index.to_point())
            );
        }
    }
}
