use std::collections::HashMap;
use std::fmt::Debug;

use itertools::Itertools;
use num_traits::Zero;

use super::logup::LogupSums;
use super::{
    EvalAtRow, FrameworkEval, InfoEvaluator, Relation, RelationEntry, TraceLocationAllocator,
    INTERACTION_TRACE_IDX,
};
use crate::core::backend::simd::m31::{PackedBaseField, LOG_N_LANES, N_LANES};
use crate::core::backend::simd::qm31::PackedSecureField;
use crate::core::backend::simd::very_packed_m31::LOG_N_VERY_PACKED_ELEMS;
use crate::core::backend::simd::SimdBackend;
use crate::core::backend::Column;
use crate::core::fields::m31::{BaseField, M31};
use crate::core::fields::secure_column::SECURE_EXTENSION_DEGREE;
use crate::core::lookups::utils::Fraction;
use crate::core::pcs::{TreeSubspan, TreeVec};
use crate::core::poly::circle::CircleEvaluation;
use crate::core::poly::BitReversedOrder;
use crate::core::utils::{
    bit_reverse_index, coset_index_to_circle_domain_index, offset_bit_reversed_circle_domain_index,
};

#[derive(Debug)]
pub struct RelationTrackerEntry {
    pub relation: String,
    pub mult: M31,
    pub values: Vec<M31>,
}

pub struct RelationTrackerComponent<E: FrameworkEval> {
    eval: E,
    trace_locations: TreeVec<TreeSubspan>,
    n_rows: usize,
}
impl<E: FrameworkEval> RelationTrackerComponent<E> {
    pub fn new(location_allocator: &mut TraceLocationAllocator, eval: E, n_rows: usize) -> Self {
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
            n_rows,
        }
    }

    pub fn entries(
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
            let evaluator =
                RelationTrackerEvaluator::new(&sub_tree, vec_row, log_size, self.n_rows);
            entries.extend(self.eval.evaluate(evaluator).entries());
        }
        entries
    }
}

/// Aggregates relation entries.
pub struct RelationTrackerEvaluator<'a> {
    entries: Vec<RelationTrackerEntry>,
    pub trace_eval:
        &'a TreeVec<Vec<&'a CircleEvaluation<SimdBackend, BaseField, BitReversedOrder>>>,
    pub column_index_per_interaction: Vec<usize>,
    pub vec_row: usize,
    pub domain_log_size: u32,
    pub n_rows: usize,
}
impl<'a> RelationTrackerEvaluator<'a> {
    pub fn new(
        trace_eval: &'a TreeVec<Vec<&CircleEvaluation<SimdBackend, BaseField, BitReversedOrder>>>,
        vec_row: usize,
        domain_log_size: u32,
        n_rows: usize,
    ) -> Self {
        Self {
            entries: vec![],
            trace_eval,
            column_index_per_interaction: vec![0; trace_eval.len()],
            vec_row,
            domain_log_size,
            n_rows,
        }
    }

    pub fn entries(self) -> Vec<RelationTrackerEntry> {
        self.entries
    }
}
impl<'a> EvalAtRow for RelationTrackerEvaluator<'a> {
    type F = PackedBaseField;
    type EF = PackedSecureField;

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
                    return *col.data.get_unchecked(self.vec_row);
                };
            }
            // Otherwise, we need to look up the value at the offset.
            // Since the domain is bit-reversed circle domain ordered, we need to look up the value
            // at the bit-reversed natural order index at an offset.
            PackedBaseField::from_array(std::array::from_fn(|i| {
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

    fn combine_ef(_values: [Self::F; SECURE_EXTENSION_DEGREE]) -> Self::EF {
        PackedSecureField::zero()
    }

    fn write_logup_frac(&mut self, _fraction: Fraction<Self::EF, Self::EF>) {}

    fn finalize_logup(&mut self) {}

    fn add_to_relation<R: Relation<Self::F, Self::EF>>(
        &mut self,
        entries: &[RelationEntry<'_, Self::F, Self::EF, R>],
    ) {
        for entry in entries {
            let relation = entry.relation.get_name().to_owned();
            let values = entry.values.iter().map(|v| v.to_array()).collect_vec();
            let mult = entry.multiplicity.to_array();

            // Unpack SIMD.
            for j in 0..N_LANES {
                // Skip padded values.
                let cannonical_index = bit_reverse_index(
                    coset_index_to_circle_domain_index(
                        (self.vec_row << LOG_N_LANES) + j,
                        self.domain_log_size,
                    ),
                    self.domain_log_size,
                );
                if cannonical_index >= self.n_rows {
                    continue;
                }
                let values = values.iter().map(|v| v[j]).collect_vec();
                let mult = mult[j].to_m31_array()[0];
                self.entries.push(RelationTrackerEntry {
                    relation: relation.clone(),
                    mult,
                    values,
                });
            }
        }
    }
}

type RelationInfo = (String, Vec<(Vec<M31>, M31)>);
pub struct RelationSummary(Vec<RelationInfo>);
impl RelationSummary {
    /// Returns the sum of every entry's yields and uses.
    /// The result is a map from relation name to a list of values(M31 vectors) and their sum.
    pub fn summarize_relations(entries: &[RelationTrackerEntry]) -> Self {
        let mut summary = vec![];
        let relations = entries.iter().group_by(|entry| entry.relation.clone());
        for (relation, entries) in &relations {
            let mut relation_sums: HashMap<Vec<_>, M31> = HashMap::new();
            for entry in entries {
                let mult = relation_sums
                    .entry(entry.values.clone())
                    .or_insert(M31::zero());
                *mult += entry.mult;
            }
            let relation_sums = relation_sums.into_iter().collect_vec();
            summary.push((relation.clone(), relation_sums));
        }
        Self(summary)
    }

    pub fn get_relation_info(&self, relation: &str) -> Option<&[(Vec<M31>, M31)]> {
        self.0
            .iter()
            .find(|(name, _)| name == relation)
            .map(|(_, entries)| entries.as_slice())
    }
}
impl Debug for RelationSummary {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        for (relation, entries) in &self.0 {
            writeln!(f, "{}:", relation)?;
            for (vector, sum) in entries {
                let vector = vector.iter().map(|v| v.0).collect_vec();
                writeln!(f, "  {:?} -> {}", vector, sum)?;
            }
        }
        Ok(())
    }
}
