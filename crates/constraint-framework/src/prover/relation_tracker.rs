use std::collections::HashMap;
use std::fmt::Debug;

use itertools::Itertools;
use num_traits::Zero;
use stwo::core::fields::m31::{BaseField, M31};
use stwo::core::fields::qm31::{SecureField, SECURE_EXTENSION_DEGREE};
use stwo::core::pcs::TreeVec;
use stwo::core::utils::{
    bit_reverse_index, circle_domain_index_to_coset_index, coset_index_to_circle_domain_index,
};
use stwo::core::Fraction;
use stwo::prover::backend::Column;

use crate::{
    Batching, EvalAtRow, FrameworkComponent, FrameworkEval, Relation, RelationEntry,
    INTERACTION_TRACE_IDX, PREPROCESSED_TRACE_IDX,
};

#[derive(Debug)]
pub struct RelationTrackerEntry {
    pub relation: String,
    pub mult: M31,
    pub values: Vec<M31>,
}

pub fn add_to_relation_entries<E: FrameworkEval>(
    component: &FrameworkComponent<E>,
    trace: &TreeVec<Vec<&Vec<BaseField>>>,
) -> Vec<RelationTrackerEntry> {
    let log_size = component.eval.log_size();

    // Deref the sub-tree. Only copies the references.
    // Interaction trace is no needed for relation tracker.
    let mut sub_tree = trace
        .sub_tree(&component.trace_locations[..INTERACTION_TRACE_IDX])
        .map_cols(|col| *col);

    // Aggregating the preprocessed columns. This information does not propagate from
    // "next_interaction_mask", hence requires special treatment.
    sub_tree[PREPROCESSED_TRACE_IDX] = component
        .preprocessed_column_indices
        .iter()
        .map(|idx| trace[PREPROCESSED_TRACE_IDX][*idx])
        .collect();

    (0..1 << log_size)
        .flat_map(|row| {
            let evaluator = RelationTrackerEvaluator::new(&sub_tree, row, log_size);
            component.eval.evaluate(evaluator).entries()
        })
        .collect()
}

/// Aggregates relation entries.
pub struct RelationTrackerEvaluator<'a> {
    entries: Vec<RelationTrackerEntry>,
    trace: &'a TreeVec<Vec<&'a Vec<BaseField>>>,
    pub column_index_per_interaction: Vec<usize>,
    pub vec_row: usize,
    pub domain_log_size: u32,
}
impl<'a> RelationTrackerEvaluator<'a> {
    pub fn new(trace: &'a TreeVec<Vec<&Vec<BaseField>>>, row: usize, domain_log_size: u32) -> Self {
        Self {
            entries: vec![],
            trace,
            column_index_per_interaction: vec![0; trace.len()],
            vec_row: row,
            domain_log_size,
        }
    }

    pub fn entries(self) -> Vec<RelationTrackerEntry> {
        self.entries
    }
}
impl EvalAtRow for RelationTrackerEvaluator<'_> {
    type F = BaseField;
    type EF = SecureField;

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
                let col = &self.trace[interaction][col_index];
                return col[self.vec_row];
            }
            // Otherwise, we need to look up the value at the offset.
            // Since the domain is bit-reversed circle domain ordered, we need to look up the value
            // at the bit-reversed natural order index at an offset.
            let domain_size = 1 << self.domain_log_size;

            let coset_index = circle_domain_index_to_coset_index(
                bit_reverse_index(self.vec_row, self.domain_log_size),
                self.domain_log_size,
            );
            let next_coset_index = (coset_index as isize + off).rem_euclid(domain_size);
            let next_index = bit_reverse_index(
                coset_index_to_circle_domain_index(next_coset_index as usize, self.domain_log_size),
                self.domain_log_size,
            );
            self.trace[interaction][col_index].at(next_index)
        })
    }

    fn add_constraint<G>(&mut self, _constraint: G) {}

    fn combine_ef(_values: [Self::F; SECURE_EXTENSION_DEGREE]) -> Self::EF {
        0.into()
    }

    fn write_logup_frac(&mut self, _fraction: Fraction<Self::EF, Self::EF>) {}

    fn finalize_logup_batched(&mut self, _batching: &Batching) {}
    fn finalize_logup(&mut self) {}
    fn finalize_logup_in_pairs(&mut self) {}

    fn add_to_relation<R: Relation<Self::F, Self::EF>>(
        &mut self,
        entry: RelationEntry<'_, Self::F, Self::EF, R>,
    ) {
        let relation = entry.relation.get_name().to_owned();
        let values = entry.values.to_vec();
        let mult = entry.multiplicity.to_m31_array()[0];

        self.entries.push(RelationTrackerEntry {
            relation: relation.clone(),
            mult,
            values,
        });
    }
}

type RelationInfo = (String, Vec<(Vec<M31>, M31)>);
pub struct RelationSummary(Vec<RelationInfo>);
impl RelationSummary {
    /// Returns the sum of every entry's yields and uses.
    /// The result is a map from relation name to a list of values(M31 vectors) and their sum.
    pub fn summarize_relations(entries: &[RelationTrackerEntry]) -> Self {
        let mut entry_by_relation = HashMap::new();
        for entry in entries {
            entry_by_relation
                .entry(entry.relation.clone())
                .or_insert_with(Vec::new)
                .push(entry);
        }
        let mut summary = vec![];
        for (relation, entries) in entry_by_relation {
            let mut relation_sums: HashMap<Vec<_>, M31> = HashMap::new();
            for entry in entries {
                let mut values = entry.values.clone();

                // Trailing zeroes do not affect the sum, remove for correct aggregation.
                while values.last().is_some_and(|v| v.is_zero()) {
                    values.pop();
                }
                let mult = relation_sums.entry(values).or_insert(M31::zero());
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

    /// Cleans up the summary by removing zero-sum entries, only keeping the non-zero ones.
    /// Used for debugging.
    pub fn cleaned(self) -> Self {
        let mut cleaned = vec![];
        for (relation, entries) in self.0 {
            let mut cleaned_entries = vec![];
            for (vector, sum) in entries {
                if !sum.is_zero() {
                    cleaned_entries.push((vector, sum));
                }
            }
            if !cleaned_entries.is_empty() {
                cleaned.push((relation, cleaned_entries));
            }
        }
        Self(cleaned)
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
