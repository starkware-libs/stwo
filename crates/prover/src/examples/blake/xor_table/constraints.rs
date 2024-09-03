use itertools::Itertools;

use super::{limb_bits, XorElements};
use crate::constraint_framework::logup::{LogupAtRow, LookupElements};
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
        let [al] = self.eval.next_interaction_mask(2, [0]);
        let [bl] = self.eval.next_interaction_mask(2, [0]);
        let [cl] = self.eval.next_interaction_mask(2, [0]);
        for [(i0, j0), (i1, j1)] in (0..1 << EXPAND_BITS)
            .cartesian_product(0..1 << EXPAND_BITS)
            .array_chunks::<2>()
        {
            let multiplicity0 = self.eval.next_trace_mask();
            let multiplicity1 = self.eval.next_trace_mask();

            let [a0, a1] = [i0, i1].map(|i| {
                al + E::F::from(BaseField::from_u32_unchecked(
                    i << limb_bits::<ELEM_BITS, EXPAND_BITS>(),
                ))
            });
            let [b0, b1] = [j0, j1].map(|j| {
                bl + E::F::from(BaseField::from_u32_unchecked(
                    j << limb_bits::<ELEM_BITS, EXPAND_BITS>(),
                ))
            });
            let [c0, c1] = [(i0, j0), (i1, j1)].map(|(i, j)| {
                cl + E::F::from(BaseField::from_u32_unchecked(
                    (i ^ j) << limb_bits::<ELEM_BITS, EXPAND_BITS>(),
                ))
            });

            let frac0 = Fraction::<E::EF, E::EF>::new(
                (-multiplicity0).into(),
                self.lookup_elements.combine(&[a0, b0, c0]),
            );
            let frac1 = Fraction::<E::EF, E::EF>::new(
                (-multiplicity1).into(),
                self.lookup_elements.combine(&[a1, b1, c1]),
            );

            // Add with negative multiplicity. Consumers should lookup with positive
            // multiplicity.
            self.logup.write_frac(&mut self.eval, frac0 + frac1);
        }
        self.logup.finalize(&mut self.eval);
        self.eval
    }
}
