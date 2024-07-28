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
//! The restr of the lookups are computed based on these constant columns.

use std::simd::u32x16;

use itertools::Itertools;
use tracing::{span, Level};

use crate::constraint_framework::constant_columns::gen_is_first;
use crate::constraint_framework::logup::{LogupAtRow, LogupTraceGenerator, LookupElements};
use crate::constraint_framework::{EvalAtRow, FrameworkComponent};
use crate::core::backend::simd::column::BaseColumn;
use crate::core::backend::simd::m31::{PackedBaseField, LOG_N_LANES};
use crate::core::backend::simd::qm31::PackedSecureField;
use crate::core::backend::simd::SimdBackend;
use crate::core::backend::Column;
use crate::core::fields::m31::BaseField;
use crate::core::fields::qm31::SecureField;
use crate::core::poly::circle::{CanonicCoset, CircleEvaluation};
use crate::core::poly::BitReversedOrder;
use crate::core::ColumnVec;

const fn limb_bits<const ELEM_BITS: u32, const EXPAND_BITS: u32>() -> u32 {
    ELEM_BITS - EXPAND_BITS
}
pub const fn column_bits<const ELEM_BITS: u32, const EXPAND_BITS: u32>() -> u32 {
    2 * limb_bits::<ELEM_BITS, EXPAND_BITS>()
}

/// Accumulator that keeps track of the number of times each input has been used.
pub struct XorAccumulator<const ELEM_BITS: u32, const EXPAND_BITS: u32> {
    /// 2^EXPAND_BITS columns.
    pub mults: Vec<BaseColumn>,
}
impl<const ELEM_BITS: u32, const EXPAND_BITS: u32> Default
    for XorAccumulator<ELEM_BITS, EXPAND_BITS>
{
    fn default() -> Self {
        Self {
            mults: (0..(1 << (2 * EXPAND_BITS)))
                .map(|_| BaseColumn::zeros(1 << column_bits::<ELEM_BITS, EXPAND_BITS>()))
                .collect_vec(),
        }
    }
}
impl<const ELEM_BITS: u32, const EXPAND_BITS: u32> XorAccumulator<ELEM_BITS, EXPAND_BITS> {
    pub fn add_input(&mut self, a: u32x16, b: u32x16) {
        let al = a & u32x16::splat((1 << limb_bits::<ELEM_BITS, EXPAND_BITS>()) - 1);
        let ah = a >> limb_bits::<ELEM_BITS, EXPAND_BITS>();
        let bl = b & u32x16::splat((1 << limb_bits::<ELEM_BITS, EXPAND_BITS>()) - 1);
        let bh = b >> limb_bits::<ELEM_BITS, EXPAND_BITS>();
        let idxh = (ah << EXPAND_BITS) + bh;
        let idxl = (al << limb_bits::<ELEM_BITS, EXPAND_BITS>()) + bl;

        // Since the indices may collide, we cannot use scatter simd operations here.
        for (ih, il) in idxh.as_array().iter().zip(idxl.as_array().iter()) {
            self.mults[*ih as usize].as_mut_slice()[*il as usize].0 += 1;
        }
    }
}

/// Component that evaluates the xor table.
pub struct XorTableComponent<const ELEM_BITS: u32, const EXPAND_BITS: u32> {
    pub lookup_elements: LookupElements,
    pub claimed_sum: SecureField,
}
impl<const ELEM_BITS: u32, const EXPAND_BITS: u32> FrameworkComponent
    for XorTableComponent<ELEM_BITS, EXPAND_BITS>
{
    fn log_size(&self) -> u32 {
        column_bits::<ELEM_BITS, EXPAND_BITS>()
    }
    fn max_constraint_log_degree_bound(&self) -> u32 {
        column_bits::<ELEM_BITS, EXPAND_BITS>() + 1
    }
    fn evaluate<E: EvalAtRow>(&self, mut eval: E) -> E {
        let [is_first] = eval.next_interaction_mask(2, [0]);
        let blake_eval = XorTableEval::<'_, _, ELEM_BITS, EXPAND_BITS> {
            eval,
            lookup_elements: &self.lookup_elements,
            logup: LogupAtRow::new(1, self.claimed_sum, is_first),
        };
        blake_eval.eval()
    }
}

/// Constraints for the xor table.
pub struct XorTableEval<'a, E: EvalAtRow, const ELEM_BITS: u32, const EXPAND_BITS: u32> {
    pub eval: E,
    pub lookup_elements: &'a LookupElements,
    pub logup: LogupAtRow<2, E>,
}
impl<'a, E: EvalAtRow, const ELEM_BITS: u32, const EXPAND_BITS: u32>
    XorTableEval<'a, E, ELEM_BITS, EXPAND_BITS>
{
    pub fn eval(mut self) -> E {
        let [a] = self.eval.next_interaction_mask(2, [0]);
        let [b] = self.eval.next_interaction_mask(2, [0]);
        let [c] = self.eval.next_interaction_mask(2, [0]);
        for i in 0..1 << EXPAND_BITS {
            for j in 0..1 << EXPAND_BITS {
                let multiplicity = self.eval.next_trace_mask();

                let a = a + E::F::from(BaseField::from_u32_unchecked(
                    i << limb_bits::<ELEM_BITS, EXPAND_BITS>(),
                ));
                let b = b + E::F::from(BaseField::from_u32_unchecked(
                    j << limb_bits::<ELEM_BITS, EXPAND_BITS>(),
                ));
                let c = c + E::F::from(BaseField::from_u32_unchecked(
                    (i ^ j) << limb_bits::<ELEM_BITS, EXPAND_BITS>(),
                ));

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
pub fn gen_interaction_trace<const ELEM_BITS: u32, const EXPAND_BITS: u32>(
    lookup_data: XorTableLookupData<ELEM_BITS, EXPAND_BITS>,
    lookup_elements: &LookupElements,
) -> (
    ColumnVec<CircleEvaluation<SimdBackend, BaseField, BitReversedOrder>>,
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

        // Each column has 2^(2*LIMB_BITS) rows, in packs of LOG_LANES.
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

    // If there is an odd number of columns, handle the last one.
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
    let (interaction_trace, claimed_sum) = logup_gen.finalize();
    (interaction_trace, constant_trace, claimed_sum)
}

#[test]
fn test_xor_table() {
    use crate::core::pcs::TreeVec;

    const ELEM_BITS: u32 = 9;
    const EXPAND_BITS: u32 = 2;

    let mut xor_accum = XorAccumulator::<ELEM_BITS, EXPAND_BITS>::default();
    xor_accum.add_input(u32x16::splat(1), u32x16::splat(2));

    let (trace, lookup_data) = generate_trace(xor_accum);
    let lookup_elements = LookupElements::dummy();
    let (interaction_trace, constant_trace, claimed_sum) =
        gen_interaction_trace(lookup_data, &lookup_elements);

    let trace = TreeVec::new(vec![trace, interaction_trace, constant_trace]);
    let trace_polys = trace.map_cols(|c| c.interpolate());

    let component = XorTableComponent::<ELEM_BITS, EXPAND_BITS> {
        lookup_elements,
        claimed_sum,
    };
    crate::constraint_framework::assert_constraints(
        &trace_polys,
        CanonicCoset::new(column_bits::<ELEM_BITS, EXPAND_BITS>()),
        |eval| {
            component.evaluate(eval);
        },
    )
}
