use itertools::Itertools;
use serde::Serialize;

use super::logup::LogupSums;
use super::{
    EvalAtRow, FrameworkEval, InfoEvaluator, Relation, RelationEntry, TraceLocationAllocator,
    INTERACTION_TRACE_IDX,
};
use crate::core::backend::simd::column::VeryPackedBaseColumn;
use crate::core::backend::simd::m31::{PackedM31, LOG_N_LANES, N_LANES};
use crate::core::backend::simd::qm31::PackedQM31;
use crate::core::backend::simd::very_packed_m31::{
    VeryPackedBaseField, VeryPackedSecureField, LOG_N_VERY_PACKED_ELEMS, N_VERY_PACKED_ELEMS,
};
use crate::core::backend::simd::SimdBackend;
use crate::core::backend::Column;
use crate::core::fields::m31::BaseField;
use crate::core::fields::secure_column::SECURE_EXTENSION_DEGREE;
use crate::core::lookups::utils::Fraction;
use crate::core::pcs::{TreeSubspan, TreeVec};
use crate::core::poly::circle::CircleEvaluation;
use crate::core::poly::BitReversedOrder;
use crate::core::utils::offset_bit_reversed_circle_domain_index;

#[derive(Serialize, Debug)]
pub struct RelationTrackerEntry {
    pub relation: String,
    pub mult: u32,
    pub values: Vec<u32>,
}

pub trait RelationTracker {
    fn entries(
        self,
        trace: &TreeVec<Vec<&CircleEvaluation<SimdBackend, BaseField, BitReversedOrder>>>,
    ) -> Vec<RelationTrackerEntry>;
}

pub struct RelationTrackerComponent<E: FrameworkEval> {
    eval: E,
    trace_locations: TreeVec<TreeSubspan>,
}
impl<E: FrameworkEval> RelationTrackerComponent<E> {
    pub fn new(location_allocator: &mut TraceLocationAllocator, eval: E) -> Self {
        let info = eval.evaluate(InfoEvaluator::new(
            eval.log_size(),
            vec![],
            LogupSums::default(),
        ));
        let mut mask_offsets = info.mask_offsets;
        mask_offsets.drain(INTERACTION_TRACE_IDX..);
        let trace_locations = location_allocator.next_for_structure(&mask_offsets);
        Self {
            eval,
            trace_locations,
        }
    }
}
impl<E: FrameworkEval> RelationTracker for RelationTrackerComponent<E> {
    fn entries(
        self,
        trace: &TreeVec<Vec<&CircleEvaluation<SimdBackend, BaseField, BitReversedOrder>>>,
    ) -> Vec<RelationTrackerEntry> {
        let log_size = self.eval.log_size();

        // Deref the sub-tree. Only copies the references.
        let sub_tree = trace
            .sub_tree(&self.trace_locations)
            .map(|vec| vec.into_iter().copied().collect_vec());
        let mut entries = vec![];

        for vec_row in 0..(1 << (log_size - LOG_N_LANES)) {
            let evaluator = RelationTrackerEvaluator::new(&sub_tree, vec_row, log_size);
            entries.extend(self.eval.evaluate(evaluator).summarize());
        }
        entries
    }
}

/// Aggregates relation entries.
/// TODO(Ohad): write a summarize function.
pub struct RelationTrackerEvaluator<'a> {
    entries: Vec<RelationTrackerEntry>,
    pub trace_eval:
        &'a TreeVec<Vec<&'a CircleEvaluation<SimdBackend, BaseField, BitReversedOrder>>>,
    pub column_index_per_interaction: Vec<usize>,
    /// The row index of the simd-vector row to evaluate the constraints at.
    pub vec_row: usize,
    pub domain_log_size: u32,
}
impl<'a> RelationTrackerEvaluator<'a> {
    pub fn new(
        trace_eval: &'a TreeVec<Vec<&CircleEvaluation<SimdBackend, BaseField, BitReversedOrder>>>,
        vec_row: usize,
        domain_log_size: u32,
    ) -> Self {
        Self {
            entries: vec![],
            trace_eval,
            column_index_per_interaction: vec![0; trace_eval.len()],
            vec_row,
            domain_log_size,
        }
    }

    pub fn summarize(self) -> Vec<RelationTrackerEntry> {
        self.entries
    }
}
impl<'a> EvalAtRow for RelationTrackerEvaluator<'a> {
    type F = VeryPackedBaseField;
    type EF = VeryPackedSecureField;

    // TODO(Ohad): Add debug boundary checks.
    fn next_interaction_mask<const N: usize>(
        &mut self,
        interaction: usize,
        offsets: [isize; N],
    ) -> [Self::F; N] {
        assert_ne!(interaction, INTERACTION_TRACE_IDX);
        let col_index = self.column_index_per_interaction[interaction];
        self.column_index_per_interaction[interaction] += 1;
        offsets.map(|off| {
            // If the offset is 0, we can just return the value directly from this row.
            if off == 0 {
                unsafe {
                    let col = &self
                        .trace_eval
                        .get_unchecked(interaction)
                        .get_unchecked(col_index)
                        .values;
                    let very_packed_col = VeryPackedBaseColumn::transform_under_ref(col);
                    return *very_packed_col.data.get_unchecked(self.vec_row);
                };
            }
            // Otherwise, we need to look up the value at the offset.
            // Since the domain is bit-reversed circle domain ordered, we need to look up the value
            // at the bit-reversed natural order index at an offset.
            VeryPackedBaseField::from_array(std::array::from_fn(|i| {
                let row_index = offset_bit_reversed_circle_domain_index(
                    (self.vec_row << (LOG_N_LANES + LOG_N_VERY_PACKED_ELEMS)) + i,
                    self.domain_log_size,
                    self.domain_log_size,
                    off,
                );
                self.trace_eval[interaction][col_index].at(row_index)
            }))
        })
    }
    fn add_constraint<G>(&mut self, _constraint: G) {}

    fn combine_ef(values: [Self::F; SECURE_EXTENSION_DEGREE]) -> Self::EF {
        VeryPackedSecureField::from_very_packed_m31s(values)
    }

    fn write_logup_frac(&mut self, _fraction: Fraction<Self::EF, Self::EF>) {}

    fn finalize_logup(&mut self) {}

    fn add_to_relation<R: Relation<Self::F, Self::EF>>(
        &mut self,
        entries: &[RelationEntry<'_, Self::F, Self::EF, R>],
    ) {
        for entry in entries {
            let relation = entry.relation.get_name().to_owned();
            // Unpack VeryPacked.
            let values: [Vec<PackedM31>; N_VERY_PACKED_ELEMS] = std::array::from_fn(|i| {
                entry
                    .values
                    .iter()
                    .map(|vectorized_value| vectorized_value.0[i])
                    .collect()
            });
            let mults: [PackedQM31; N_VERY_PACKED_ELEMS] =
                std::array::from_fn(|i| entry.multiplicity.0[i]);

            for i in 0..N_VERY_PACKED_ELEMS {
                let values = values
                    .iter()
                    .map(|v| v[i].into_simd().to_array())
                    .collect_vec();
                let mult = mults[i].to_array();
                // Unpack SIMD.
                for j in 0..N_LANES {
                    let values = values.iter().map(|v| v[j]).collect_vec();
                    let mult = mult[j].to_m31_array()[0].0;
                    self.entries.push(RelationTrackerEntry {
                        relation: relation.clone(),
                        mult,
                        values,
                    });
                }
            }
        }
    }
}
