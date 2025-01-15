#![allow(unused)]
//! Xor table component.
//! Generic on `ELEM_BITS` and `EXPAND_BITS`.
//! The table has all triplets of (a, b, a^b), where a, b are in the range [0,2^ELEM_BITS).
//! a,b are split into high and low parts, of size `EXPAND_BITS` and `ELEM_BITS - EXPAND_BITS`
//! respectively.
//! The component itself will hold 2^(2*EXPAND_BITS) multiplicity columns, each of size
//! 2^(ELEM_BITS - EXPAND_BITS).
//! The constant columns correspond only to the smaller table of the lower `ELEM_BITS - EXPAND_BITS`
//! xors: (a_l, b_l, a_l^b_l).
//! The rest of the lookups are computed based on these constant columns.

mod constraints;
mod gen;

use std::simd::u32x16;

use itertools::Itertools;
use num_traits::Zero;
use tracing::{span, Level};

use super::preprocessed_columns::XorTable;
use crate::constraint_framework::logup::{LogupAtRow, LogupTraceGenerator};
use crate::constraint_framework::preprocessed_columns::IsFirst;
use crate::constraint_framework::{
    relation, EvalAtRow, FrameworkComponent, FrameworkEval, InfoEvaluator, Relation, RelationEntry,
    INTERACTION_TRACE_IDX, PREPROCESSED_TRACE_IDX,
};
use crate::core::backend::simd::column::BaseColumn;
use crate::core::backend::simd::m31::{PackedBaseField, LOG_N_LANES};
use crate::core::backend::simd::qm31::PackedSecureField;
use crate::core::backend::simd::SimdBackend;
use crate::core::backend::Column;
use crate::core::fields::m31::BaseField;
use crate::core::fields::qm31::SecureField;
use crate::core::lookups::utils::Fraction;
use crate::core::pcs::{TreeSubspan, TreeVec};
use crate::core::poly::circle::{CanonicCoset, CircleEvaluation};
use crate::core::poly::BitReversedOrder;
use crate::core::ColumnVec;
use crate::examples::blake::{
    XorElements12, XorElements4, XorElements7, XorElements8, XorElements9,
};
use crate::{xor_table_eval, xor_table_gen};

macro_rules! xor_table_component {
    ($modname:tt, $elements:tt, $elem_bits:literal, $expand_bits:literal) => {
        pub fn trace_sizes<const ELEM_BITS: u32, const EXPAND_BITS: u32>() -> TreeVec<Vec<u32>> {
            let component = XorTableEval::<ELEM_BITS, EXPAND_BITS> {
                lookup_elements: $elements::dummy(),
                claimed_sum: SecureField::zero(),
            };
            let info = component.evaluate(InfoEvaluator::empty());
            info.mask_offsets
                .as_cols_ref()
                .map_cols(|_| XorTable::new(ELEM_BITS, EXPAND_BITS, 0).column_bits())
        }

        /// Accumulator that keeps track of the number of times each input has been used.
        pub struct XorAccumulator<const ELEM_BITS: u32, const EXPAND_BITS: u32> {
            /// 2^(2*EXPAND_BITS) multiplicity columns. Index (al, bl) of column (ah, bh) is the
            /// number of times ah||al ^ bh||bl has been used.
            pub mults: Vec<BaseColumn>,
        }
        impl<const ELEM_BITS: u32, const EXPAND_BITS: u32> Default
            for XorAccumulator<ELEM_BITS, EXPAND_BITS>
        {
            fn default() -> Self {
                Self {
                    mults: (0..(1 << (2 * EXPAND_BITS)))
                        .map(|_| {
                            BaseColumn::zeros(
                                1 << XorTable::new(ELEM_BITS, EXPAND_BITS, 0).column_bits(),
                            )
                        })
                        .collect_vec(),
                }
            }
        }
        impl<const ELEM_BITS: u32, const EXPAND_BITS: u32> XorAccumulator<ELEM_BITS, EXPAND_BITS> {
            pub fn add_input(&mut self, a: u32x16, b: u32x16) {
                // Split a and b into high and low parts, according to ELEMENT_BITS and EXPAND_BITS.
                // The high part is the index of the multiplicity column.
                // The low part is the index of the element in that column.
                let al = a & u32x16::splat(
                    (1 << XorTable::new(ELEM_BITS, EXPAND_BITS, 0).limb_bits()) - 1,
                );
                let ah = a >> XorTable::new(ELEM_BITS, EXPAND_BITS, 0).limb_bits();
                let bl = b & u32x16::splat(
                    (1 << XorTable::new(ELEM_BITS, EXPAND_BITS, 0).limb_bits()) - 1,
                );
                let bh = b >> XorTable::new(ELEM_BITS, EXPAND_BITS, 0).limb_bits();
                let column_idx = (ah << EXPAND_BITS) + bh;
                let offset = (al << XorTable::new(ELEM_BITS, EXPAND_BITS, 0).limb_bits()) + bl;

                // Since the indices may collide, we cannot use scatter simd operations here.
                // Instead, loop over packed values.
                for (column_idx, offset) in
                    column_idx.as_array().iter().zip(offset.as_array().iter())
                {
                    self.mults[*column_idx as usize].as_mut_slice()[*offset as usize].0 += 1;
                }
            }
        }
        /// Component that evaluates the xor table.
        pub type XorTableComponent<const ELEM_BITS: u32, const EXPAND_BITS: u32> =
            FrameworkComponent<XorTableEval<ELEM_BITS, EXPAND_BITS>>;

        /// Evaluates the xor table.
        pub struct XorTableEval<const ELEM_BITS: u32, const EXPAND_BITS: u32> {
            pub lookup_elements: $elements,
            pub claimed_sum: SecureField,
        }

        impl<const ELEM_BITS: u32, const EXPAND_BITS: u32> FrameworkEval
            for XorTableEval<ELEM_BITS, EXPAND_BITS>
        {
            fn log_size(&self) -> u32 {
                XorTable::new($elem_bits, $expand_bits, 0).column_bits()
            }
            fn max_constraint_log_degree_bound(&self) -> u32 {
                XorTable::new($elem_bits, $expand_bits, 0).column_bits() + 1
            }
            fn evaluate<E: EvalAtRow>(&self, mut eval: E) -> E {
                let xor_eval = $modname::XorTableEvalAtRow::<'_, _, ELEM_BITS, EXPAND_BITS> {
                    eval,
                    lookup_elements: &self.lookup_elements,
                    claimed_sum: self.claimed_sum,
                    log_size: self.log_size(),
                };
                xor_eval.eval()
            }
        }
    };
}

macro_rules! define_xor_table {
    ($modname:tt, $elements:tt, $elem_bits:literal, $expand_bits:literal) => {
        pub mod $modname {
            use super::*;
            xor_table_component!($modname, $elements, $elem_bits, $expand_bits);
            xor_table_eval!($modname, $elements, $elem_bits, $expand_bits);
            xor_table_gen!($modname, $elements, $elem_bits, $expand_bits);
        }
    };
}

define_xor_table!(xor12, XorElements12, 12, 4);
define_xor_table!(xor9, XorElements9, 9, 2);
define_xor_table!(xor8, XorElements8, 8, 2);
define_xor_table!(xor7, XorElements7, 7, 2);
define_xor_table!(xor4, XorElements4, 4, 0);

#[cfg(test)]
mod tests {
    use std::simd::u32x16;

    use crate::constraint_framework::logup::LookupElements;
    use crate::constraint_framework::{assert_constraints, FrameworkEval};
    use crate::core::poly::circle::CanonicCoset;
    use crate::examples::blake::preprocessed_columns::XorTable;
    use crate::examples::blake::xor_table::xor12::{
        generate_interaction_trace, generate_trace, XorAccumulator, XorTableEval,
    };

    #[test]
    fn test_xor_table() {
        use crate::core::pcs::TreeVec;

        const ELEM_BITS: u32 = 9;
        const EXPAND_BITS: u32 = 2;

        let mut xor_accum = XorAccumulator::<ELEM_BITS, EXPAND_BITS>::default();
        xor_accum.add_input(u32x16::splat(1), u32x16::splat(2));

        let (trace, lookup_data) = generate_trace(xor_accum);
        let lookup_elements = crate::examples::blake::XorElements12::dummy();
        let (interaction_trace, claimed_sum) =
            generate_interaction_trace(lookup_data, &lookup_elements);
        let constant_trace = XorTable::new(ELEM_BITS, EXPAND_BITS, 0).generate_constant_trace();

        let trace = TreeVec::new(vec![constant_trace, trace, interaction_trace]);
        let trace_polys = TreeVec::<Vec<_>>::map_cols(trace, |c| c.interpolate());

        let component = XorTableEval::<ELEM_BITS, EXPAND_BITS> {
            lookup_elements,
            claimed_sum,
        };
        assert_constraints(
            &trace_polys,
            CanonicCoset::new(XorTable::new(ELEM_BITS, EXPAND_BITS, 0).column_bits()),
            |eval| {
                component.evaluate(eval);
            },
            claimed_sum,
        )
    }
}
