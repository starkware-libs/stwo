use super::{limb_bits, XorElements};
use crate::constraint_framework::logup::{LogupAtRow, LookupElements};
use crate::constraint_framework::EvalAtRow;
use crate::core::fields::m31::BaseField;

/// Constraints for the xor table.
pub struct XorTableEval<'a, E: EvalAtRow, const ELEM_BITS: u32, const EXPAND_BITS: u32> {
    pub eval: E,
    pub lookup_elements: &'a XorElements,
    pub logup: LogupAtRow<2, E>,
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
        for i in 0..1 << EXPAND_BITS {
            for j in 0..1 << EXPAND_BITS {
                let multiplicity = self.eval.next_trace_mask();

                let a = al
                    + E::F::from(BaseField::from_u32_unchecked(
                        i << limb_bits::<ELEM_BITS, EXPAND_BITS>(),
                    ));
                let b = bl
                    + E::F::from(BaseField::from_u32_unchecked(
                        j << limb_bits::<ELEM_BITS, EXPAND_BITS>(),
                    ));
                let c = cl
                    + E::F::from(BaseField::from_u32_unchecked(
                        (i ^ j) << limb_bits::<ELEM_BITS, EXPAND_BITS>(),
                    ));

                // Add with negative multiplicity. Consumers should lookup with positive
                // multiplicity.
                self.logup.push_lookup(
                    &mut self.eval,
                    (-multiplicity).into(),
                    &[a, b, c],
                    self.lookup_elements,
                );
            }
        }
        self.logup.finalize(&mut self.eval);
        self.eval
    }
}
