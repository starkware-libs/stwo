use itertools::Itertools;

use super::{limb_bits, XorElements};
use crate::constraint_framework::logup::{LogupAtRow, LookupElements};
use crate::constraint_framework::preprocessed_columns::PreprocessedColumn;
use crate::constraint_framework::EvalAtRow;
use crate::core::fields::m31::BaseField;
use crate::core::lookups::utils::Fraction;

/// Constraints for the xor table.
pub struct XorTableEval<'a, E: EvalAtRow, const ELEM_BITS: u32, const EXPAND_BITS: u32> {
    pub eval: E,
    pub lookup_elements: &'a XorElements,
    pub logup: LogupAtRow<E>,
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

        let frac_chunks = (0..(1 << (2 * EXPAND_BITS)))
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

                Fraction::<E::EF, E::EF>::new(
                    (-multiplicity).into(),
                    self.lookup_elements.combine(&[a, b, c]),
                )
            })
            .collect_vec();

        for frac_chunk in frac_chunks.chunks(2) {
            let sum_frac: Fraction<E::EF, E::EF> = frac_chunk.iter().cloned().sum();
            self.logup.write_frac(&mut self.eval, sum_frac);
        }
        self.logup.finalize(&mut self.eval);
        self.eval
    }
}
