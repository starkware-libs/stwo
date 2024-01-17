use std::collections::BTreeMap;
use std::ops::Deref;

use super::circle::{CirclePoint, CirclePointIndex};
use super::fields::m31::BaseField;
use super::fields::qm31::QM31;
use super::fields::ExtensionOf;
use super::poly::circle::{CircleEvaluation, CirclePoly, PointMapping};
use crate::core::constraints::PolyOracle;
use crate::core::poly::circle::CanonicCoset;

/// Arithmetic Intermediate Representation (AIR).
/// An Air instance is assumed to already contain all the information needed to
/// evaluate the constraints.
/// For instance, all interaction elements are assumed to be present in it.
/// Therefore, an AIR is generated only after the initial trace commitment phase.
// TODO(spapini): consider renaming this struct.
pub trait Air {
    fn components(&self) -> Vec<Box<dyn Component>>;
}

/// A component is a set of trace columns of various sizes along with a set of
/// constraints on them.
pub trait Component {
    /// Evaluates the constraint quotients of the component on canonic cosets.
    /// Accumulates quotients using `combination_coefficient` for each size.
    // Note: This will be computed using a MaterializedGraph.
    fn evaluate_constraint_quotients_on_domain(
        &self,
        trace: &ComponentTrace,
        combination_coefficient: QM31,
        evaluation_accumulator: &mut Vec<Option<CircleEvaluation<QM31>>>,
    );

    /// Evaluates the mask values for the constraints at a point.
    fn mask_values_at_point(
        &self,
        point: CirclePoint<QM31>,
        component_trace: &ComponentTrace,
    ) -> Vec<QM31>;

    /// Evaluates the constraint quotients combincation of the component, given the mask values.
    fn evaluate_quotients_by_mask(
        &self,
        point: CirclePoint<QM31>,
        combination_coefficient: QM31,
        mask: Vec<QM31>,
    ) -> Vec<QM31>;

    // TODO(spapini): Extra functions for FRI and decommitment.
}

pub struct ComponentTrace(pub Vec<CirclePoly<BaseField>>);

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
    pub fn get_evaluation<F: ExtensionOf<BaseField>>(
        &self,
        cosets: &[CanonicCoset],
        poly_oracles: &[impl PolyOracle<F>],
    ) -> PointMapping<F> {
        let mut res: BTreeMap<CirclePoint<F>, F> = BTreeMap::new();
        let mask_offsets = self.get_point_indices(cosets);
        for (mask_item, mask_offset) in self.items.iter().zip(mask_offsets) {
            res.insert(
                poly_oracles[mask_item.column_index].point() + mask_offset.to_point().into_ef(),
                poly_oracles[mask_item.column_index].get_at(mask_offset),
            );
        }
        PointMapping::new(res)
    }

    pub fn get_point_indices(&self, cosets: &[CanonicCoset]) -> Vec<CirclePointIndex> {
        let mut res = Vec::with_capacity(self.items.len());
        for item in &self.items {
            res.push(cosets[item.column_index].index_at(item.offset));
        }
        res
    }
}

impl Deref for Mask {
    type Target = Vec<MaskItem>;

    fn deref(&self) -> &Self::Target {
        &self.items
    }
}

#[cfg(test)]
mod tests {
    use crate::core::air::{Mask, MaskItem};
    use crate::core::constraints::EvalByPoly;
    use crate::core::fields::m31::BaseField;
    use crate::core::poly::circle::{CanonicCoset, CircleEvaluation, Evaluation};

    #[test]
    fn test_mask() {
        const N_TRACE_COLUMNS: u32 = 4;
        const COSET_SIZE: u32 = 16;
        let coset = CanonicCoset::new(4);
        let trace_domains = [coset; N_TRACE_COLUMNS as usize];
        let trace: Vec<CircleEvaluation<BaseField>> = (0..N_TRACE_COLUMNS)
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
        let mask_point_indices = mask.get_point_indices(&trace_domains);
        let oracle_point_index = coset.step_size * 3;
        let oracle_point = oracle_point_index.to_point();
        let poly_oracles = (0..N_TRACE_COLUMNS)
            .map(|i| EvalByPoly {
                point: oracle_point,
                poly: &trace_polys[i as usize],
            })
            .collect::<Vec<_>>();
        // Mask evaluations on the original trace coset.
        let mask_evaluation = mask.get_evaluation(&trace_domains, &poly_oracles);

        assert_eq!(mask_point_indices.len(), mask_evaluation.len());
        for (mask_item, mask_point_index) in mask.items.iter().zip(mask_point_indices) {
            let point_index = oracle_point_index + mask_point_index;
            let point = point_index.to_point();
            let value = mask_evaluation.get_at(point);
            assert_eq!(value, trace[mask_item.column_index].get_at(point_index));
        }
    }
}
