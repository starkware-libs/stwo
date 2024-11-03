use std::simd::u32x16;

use itertools::Itertools;
use tracing::{span, Level};

use super::{column_bits, limb_bits, XorAccumulator, XorElements};
use crate::constraint_framework::logup::{LogupTraceGenerator, LookupElements};
use crate::constraint_framework::preprocessed_columns::gen_is_first;
use crate::core::backend::simd::column::BaseColumn;
use crate::core::backend::simd::m31::{PackedBaseField, LOG_N_LANES};
use crate::core::backend::simd::qm31::PackedSecureField;
use crate::core::backend::simd::SimdBackend;
use crate::core::fields::m31::BaseField;
use crate::core::fields::qm31::SecureField;
use crate::core::poly::circle::{CanonicCoset, CircleEvaluation};
use crate::core::poly::BitReversedOrder;
use crate::core::ColumnVec;

pub struct XorTableLookupData<const ELEM_BITS: u32, const EXPAND_BITS: u32> {
    pub xor_accum: XorAccumulator<ELEM_BITS, EXPAND_BITS>,
}

pub fn generate_trace<const ELEM_BITS: u32, const EXPAND_BITS: u32>(
    xor_accum: XorAccumulator<ELEM_BITS, EXPAND_BITS>,
) -> (
    ColumnVec<CircleEvaluation<SimdBackend, BaseField, BitReversedOrder>>,
    XorTableLookupData<ELEM_BITS, EXPAND_BITS>,
) {
    (
        xor_accum
            .mults
            .iter()
            .map(|mult| {
                CircleEvaluation::new(
                    CanonicCoset::new(column_bits::<ELEM_BITS, EXPAND_BITS>()).circle_domain(),
                    mult.clone(),
                )
            })
            .collect_vec(),
        XorTableLookupData { xor_accum },
    )
}

/// Generates the interaction trace for the xor table.
/// Returns the interaction trace, the constant trace, and the claimed sum.
#[allow(clippy::type_complexity)]
pub fn generate_interaction_trace<const ELEM_BITS: u32, const EXPAND_BITS: u32>(
    lookup_data: XorTableLookupData<ELEM_BITS, EXPAND_BITS>,
    lookup_elements: &XorElements,
) -> (
    ColumnVec<CircleEvaluation<SimdBackend, BaseField, BitReversedOrder>>,
    SecureField,
) {
    let limb_bits = limb_bits::<ELEM_BITS, EXPAND_BITS>();
    let _span = span!(Level::INFO, "Xor interaction trace").entered();
    let offsets_vec = u32x16::from_array(std::array::from_fn(|i| i as u32));
    let mut logup_gen = LogupTraceGenerator::new(column_bits::<ELEM_BITS, EXPAND_BITS>());

    // Iterate each pair of columns, to batch their lookup together.
    // There are 2^(2*EXPAND_BITS) column, for each combination of ah, bh.
    let mut iter = lookup_data
        .xor_accum
        .mults
        .iter()
        .enumerate()
        .array_chunks::<2>();
    for [(i0, mults0), (i1, mults1)] in &mut iter {
        let mut col_gen = logup_gen.new_col();

        // Extract ah, bh from column index.
        let ah0 = i0 as u32 >> EXPAND_BITS;
        let bh0 = i0 as u32 & ((1 << EXPAND_BITS) - 1);
        let ah1 = i1 as u32 >> EXPAND_BITS;
        let bh1 = i1 as u32 & ((1 << EXPAND_BITS) - 1);

        // Each column has 2^(2*LIMB_BITS) rows, packed in N_LANES.
        #[allow(clippy::needless_range_loop)]
        for vec_row in 0..(1 << (column_bits::<ELEM_BITS, EXPAND_BITS>() - LOG_N_LANES)) {
            // vec_row is LIMB_BITS of al and LIMB_BITS - LOG_N_LANES of bl.
            // Extract al, blh from vec_row.
            let al = vec_row >> (limb_bits - LOG_N_LANES);
            let blh = vec_row & ((1 << (limb_bits - LOG_N_LANES)) - 1);

            // Construct the 3 vectors a, b, c.
            let a0 = u32x16::splat((ah0 << limb_bits) | al);
            let a1 = u32x16::splat((ah1 << limb_bits) | al);
            // bll is just the consecutive numbers 0 .. N_LANES-1.
            let b0 = u32x16::splat((bh0 << limb_bits) | (blh << LOG_N_LANES)) | offsets_vec;
            let b1 = u32x16::splat((bh1 << limb_bits) | (blh << LOG_N_LANES)) | offsets_vec;

            let c0 = a0 ^ b0;
            let c1 = a1 ^ b1;

            let p0: PackedSecureField = lookup_elements
                .combine(&[a0, b0, c0].map(|x| unsafe { PackedBaseField::from_simd_unchecked(x) }));
            let p1: PackedSecureField = lookup_elements
                .combine(&[a1, b1, c1].map(|x| unsafe { PackedBaseField::from_simd_unchecked(x) }));

            let num = p1 * mults0.data[vec_row as usize] + p0 * mults1.data[vec_row as usize];
            let denom = p0 * p1;
            col_gen.write_frac(vec_row as usize, -num, denom);
        }
        col_gen.finalize_col();
    }

    // If there is an odd number of lookup expressions, handle the last one.
    if let Some(rem) = iter.into_remainder() {
        if let Some((i, mults)) = rem.collect_vec().pop() {
            let mut col_gen = logup_gen.new_col();
            let ah = i as u32 >> EXPAND_BITS;
            let bh = i as u32 & ((1 << EXPAND_BITS) - 1);

            #[allow(clippy::needless_range_loop)]
            for vec_row in 0..(1 << (column_bits::<ELEM_BITS, EXPAND_BITS>() - LOG_N_LANES)) {
                // vec_row is LIMB_BITS of a, and LIMB_BITS - LOG_N_LANES of b.
                let al = vec_row >> (limb_bits - LOG_N_LANES);
                let a = u32x16::splat((ah << limb_bits) | al);
                let bm = vec_row & ((1 << (limb_bits - LOG_N_LANES)) - 1);
                let b = u32x16::splat((bh << limb_bits) | (bm << LOG_N_LANES)) | offsets_vec;

                let c = a ^ b;

                let p: PackedSecureField = lookup_elements.combine(
                    &[a, b, c].map(|x| unsafe { PackedBaseField::from_simd_unchecked(x) }),
                );

                let num = mults.data[vec_row as usize];
                let denom = p;
                col_gen.write_frac(vec_row as usize, PackedSecureField::from(-num), denom);
            }
            col_gen.finalize_col();
        }
    }

    logup_gen.finalize_last()
}

/// Generates the constant trace for the xor table.
/// Returns the constant trace, the constant trace, and the claimed sum.
#[allow(clippy::type_complexity)]
pub fn generate_constant_trace<const ELEM_BITS: u32, const EXPAND_BITS: u32>(
) -> ColumnVec<CircleEvaluation<SimdBackend, BaseField, BitReversedOrder>> {
    let limb_bits = limb_bits::<ELEM_BITS, EXPAND_BITS>();
    let _span = span!(Level::INFO, "Xor constant trace").entered();

    // Generate the constant columns. In reality, these should be generated before the proof
    // even began.
    let a_col: BaseColumn = (0..(1 << (column_bits::<ELEM_BITS, EXPAND_BITS>())))
        .map(|i| BaseField::from_u32_unchecked((i >> limb_bits) as u32))
        .collect();
    let b_col: BaseColumn = (0..(1 << (column_bits::<ELEM_BITS, EXPAND_BITS>())))
        .map(|i| BaseField::from_u32_unchecked((i & ((1 << limb_bits) - 1)) as u32))
        .collect();
    let c_col: BaseColumn = (0..(1 << (column_bits::<ELEM_BITS, EXPAND_BITS>())))
        .map(|i| {
            BaseField::from_u32_unchecked(((i >> limb_bits) ^ (i & ((1 << limb_bits) - 1))) as u32)
        })
        .collect();

    let mut constant_trace = [a_col, b_col, c_col]
        .map(|x| {
            CircleEvaluation::new(
                CanonicCoset::new(column_bits::<ELEM_BITS, EXPAND_BITS>()).circle_domain(),
                x,
            )
        })
        .to_vec();
    constant_trace.insert(0, gen_is_first(column_bits::<ELEM_BITS, EXPAND_BITS>()));
    constant_trace
}
