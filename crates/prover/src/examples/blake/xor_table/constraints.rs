use itertools::Itertools;

use crate::constraint_framework::logup::{LogupAtRow, LookupElements};
use crate::constraint_framework::preprocessed_columns::PreprocessedColumn;
use crate::constraint_framework::{EvalAtRow, Relation, RelationEntry};
use crate::core::fields::m31::BaseField;
use crate::core::fields::qm31::SecureField;
use crate::core::lookups::utils::Fraction;
use crate::examples::blake::{
    XorElements12, XorElements4, XorElements7, XorElements8, XorElements9,
};

macro_rules! xor_table_eval {
    ($modname:tt, $elements:tt, $elem_bits:literal, $expand_bits:literal) => {
        pub mod $modname {
            use super::*;
            use crate::examples::blake::xor_table::$modname::limb_bits;

/// Constraints for the xor table.
pub struct XorTableEval<'a, E: EvalAtRow, const ELEM_BITS: u32, const EXPAND_BITS: u32> {
    pub eval: E,
    pub lookup_elements: &'a $elements,
    pub claimed_sum: SecureField,
    pub log_size: u32,
}
impl<'a, E: EvalAtRow, const ELEM_BITS: u32, const EXPAND_BITS: u32>
    XorTableEval<'a, E, ELEM_BITS, EXPAND_BITS>
{
    pub fn eval(mut self) -> E {
        // al, bl are the constant columns for the inputs: All pairs of elements in [0,
        // 2^LIMB_BITS).
        // cl is the constant column for the xor: al ^ bl.
        let al = self
            .eval
            .get_preprocessed_column(PreprocessedColumn::XorTable(ELEM_BITS, EXPAND_BITS, 0));

        let bl = self
            .eval
            .get_preprocessed_column(PreprocessedColumn::XorTable(ELEM_BITS, EXPAND_BITS, 1));

        let cl = self
            .eval
            .get_preprocessed_column(PreprocessedColumn::XorTable(ELEM_BITS, EXPAND_BITS, 2));

        let entry_chunks = (0..(1 << (2 * EXPAND_BITS)))
            .map(|i| {
                let (i, j) = ((i >> EXPAND_BITS) as u32, (i % (1 << EXPAND_BITS)) as u32);
                let multiplicity = self.eval.next_trace_mask();

                let a = al.clone()
                    + E::F::from(BaseField::from_u32_unchecked(
                        i << limb_bits::<ELEM_BITS, EXPAND_BITS>(),
                    ));
                let b = bl.clone()
                    + E::F::from(BaseField::from_u32_unchecked(
                        j << limb_bits::<ELEM_BITS, EXPAND_BITS>(),
                    ));
                let c = cl.clone()
                    + E::F::from(BaseField::from_u32_unchecked(
                        (i ^ j) << limb_bits::<ELEM_BITS, EXPAND_BITS>(),
                    ));

                (self.lookup_elements, -multiplicity, [a, b, c])
            })
            .collect_vec();

        for entry_chunk in entry_chunks.chunks(2) {
            self.eval.add_to_relation(&[
                RelationEntry::new(entry_chunk[0].0, entry_chunk[0].1.clone().into(), &entry_chunk[0].2),
                RelationEntry::new(entry_chunk[1].0, entry_chunk[1].1.clone().into(), &entry_chunk[1].2),
            ]);
        }
        self.eval.finalize();
        self.eval
    }
}
}}}

xor_table_eval!(xor12, XorElements12, 12, 4);
xor_table_eval!(xor9, XorElements9, 9, 2);
xor_table_eval!(xor8, XorElements8, 8, 2);
xor_table_eval!(xor7, XorElements7, 7, 2);
xor_table_eval!(xor4, XorElements4, 4, 0);
