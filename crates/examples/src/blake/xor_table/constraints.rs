#[macro_export]
macro_rules! xor_table_eval {
    ($modname:tt, $elements:tt, $elem_bits:literal, $expand_bits:literal) => {
        /// Constraints for the xor table.
        pub struct XorTableEvalAtRow<'a, E: EvalAtRow, const ELEM_BITS: u32, const EXPAND_BITS: u32>
        {
            pub eval: E,
            pub lookup_elements: &'a $elements,
            pub claimed_sum: SecureField,
            pub log_size: u32,
        }
        impl<'a, E: EvalAtRow, const ELEM_BITS: u32, const EXPAND_BITS: u32>
            XorTableEvalAtRow<'a, E, ELEM_BITS, EXPAND_BITS>
        {
            pub fn eval(mut self) -> E {
                // al, bl are the constant columns for the inputs: All pairs of elements in [0,
                // 2^LIMB_BITS).
                // cl is the constant column for the xor: al ^ bl.
                let al = self
                    .eval
                    .get_preprocessed_column(XorTable::new(ELEM_BITS, EXPAND_BITS, 0).id());

                let bl = self
                    .eval
                    .get_preprocessed_column(XorTable::new(ELEM_BITS, EXPAND_BITS, 1).id());

                let cl = self
                    .eval
                    .get_preprocessed_column(XorTable::new(ELEM_BITS, EXPAND_BITS, 2).id());

                for i in (0..(1 << (2 * EXPAND_BITS))) {
                    let (i, j) = ((i >> EXPAND_BITS) as u32, (i % (1 << EXPAND_BITS)) as u32);
                    let multiplicity = self.eval.next_trace_mask();

                    let a = al.clone()
                        + E::F::from(BaseField::from_u32_unchecked(
                            i << XorTable::new(ELEM_BITS, EXPAND_BITS, 0).limb_bits(),
                        ));
                    let b = bl.clone()
                        + E::F::from(BaseField::from_u32_unchecked(
                            j << XorTable::new(ELEM_BITS, EXPAND_BITS, 1).limb_bits(),
                        ));
                    let c = cl.clone()
                        + E::F::from(BaseField::from_u32_unchecked(
                            (i ^ j) << XorTable::new(ELEM_BITS, EXPAND_BITS, 2).limb_bits(),
                        ));

                    self.eval.add_to_relation(RelationEntry::new(
                        self.lookup_elements,
                        -E::EF::from(multiplicity),
                        &[a, b, c],
                    ));
                }

                self.eval.finalize_logup_in_pairs();
                self.eval
            }
        }
    };
}
pub(crate) use xor_table_eval;
