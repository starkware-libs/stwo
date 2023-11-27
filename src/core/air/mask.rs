use std::collections::BTreeMap;

use crate::core::circle::CirclePointIndex;
use crate::core::constraints::PolyOracle;
use crate::core::fields::m31::BaseField;
use crate::core::poly::circle::{CanonicCoset, PointSetEvaluation};

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
    use crate::core::air::mask::{Mask, MaskItem};
    use crate::core::circle::{CirclePoint, CirclePointIndex};
    use crate::core::constraints::EvalByPoly;
    use crate::core::fields::m31::BaseField;
    use crate::core::poly::circle::{CanonicCoset, CircleEvaluation, Evaluation};

    #[test]
    fn test_mask() {
        const N_TRACE_COLUMNS: u32 = 4;
        const COSET_SIZE: u32 = 8;
        let coset = CanonicCoset::new(3);
        let trace_cosets = [coset; N_TRACE_COLUMNS as usize];
        let trace: Vec<CircleEvaluation> = (0..N_TRACE_COLUMNS)
            .map(|i| {
                CircleEvaluation::new_canonical_ordered(
                    coset,
                    (COSET_SIZE * i..COSET_SIZE * (i + 1))
                        .map(BaseField::from_u32_unchecked)
                        .collect(),
                )
            })
            .collect();
        let trace_polys = trace
            .iter()
            .map(|column| column.clone().interpolate())
            .collect::<Vec<_>>();
        let mask = Mask::new(
            (0..3)
                .map(|i| MaskItem {
                    column_index: i,
                    offset: i,
                })
                .collect(),
        );
        let mask_points = mask.get_point_indices(&trace_cosets);
        let poly_oracles = (0..N_TRACE_COLUMNS)
            .map(|i| EvalByPoly {
                point: CirclePoint::zero(),
                poly: &trace_polys[i as usize],
            })
            .collect::<Vec<_>>();

        // Mask evaluations on the original trace coset.
        let mask_evaluation =
            mask.get_evaluation(&trace_cosets, &poly_oracles, CirclePointIndex(0));

        assert_eq!(mask_points.len() * 2, mask_evaluation.len());
        for (mask_item, mask_point) in mask.items.iter().zip(mask_points) {
            let value = mask_evaluation.get_at(mask_point);
            let conjugate_value = mask_evaluation.get_at(-mask_point);
            assert_eq!(value, trace[mask_item.column_index].get_at(mask_point));
            assert_eq!(
                conjugate_value,
                trace[mask_item.column_index].get_at(-mask_point)
            );
        }
    }
}
