use std::simd::u32x16;

use itertools::Itertools;
use num_traits::Zero;
use tracing::{span, Level};

use super::DomainEvalHelper;
use crate::constraint_framework::logup::{LogupAtRow, LogupTraceGenerator, LookupElements};
use crate::constraint_framework::{EvalAtRow, InfoEvaluator, PointEvaluator, SimdDomainEvaluator};
use crate::core::air::accumulation::{DomainEvaluationAccumulator, PointEvaluationAccumulator};
use crate::core::air::{Component, ComponentProver, ComponentTrace};
use crate::core::backend::simd::column::BaseFieldVec;
use crate::core::backend::simd::m31::{PackedBaseField, LOG_N_LANES};
use crate::core::backend::simd::qm31::PackedSecureField;
use crate::core::backend::simd::SimdBackend;
use crate::core::backend::Column;
use crate::core::circle::CirclePoint;
use crate::core::constraints::coset_vanishing;
use crate::core::fields::m31::BaseField;
use crate::core::fields::qm31::SecureField;
use crate::core::fields::FieldExpOps;
use crate::core::pcs::TreeVec;
use crate::core::poly::circle::{CanonicCoset, CircleEvaluation};
use crate::core::poly::BitReversedOrder;
use crate::core::{ColumnVec, InteractionElements, LookupValues};

// pub const LIMB_BITS: u32 = 10;
pub const LIMB_EXPAND_BITS: u32 = 2;
// pub const COLUMN_BITS: u32 = 2 * LIMB_BITS;
const fn limb_bits<const ELEM_BITS: u32>() -> u32 {
    ELEM_BITS - LIMB_EXPAND_BITS
}
const fn column_bits<const ELEM_BITS: u32>() -> u32 {
    2 * limb_bits::<ELEM_BITS>()
}

pub struct XorAccumulator<const ELEM_BITS: u32> {
    pub mults: [BaseFieldVec; 1 << (2 * (LIMB_EXPAND_BITS))],
}
impl<const ELEM_BITS: u32> Default for XorAccumulator<ELEM_BITS> {
    fn default() -> Self {
        Self {
            mults: std::array::from_fn(|_| BaseFieldVec::zeros(1 << column_bits::<ELEM_BITS>())),
        }
    }
}
impl<const ELEM_BITS: u32> XorAccumulator<ELEM_BITS> {
    pub fn add(&mut self, a: u32x16, b: u32x16) {
        let al = a & u32x16::splat((1 << limb_bits::<ELEM_BITS>()) - 1);
        let ah = a >> limb_bits::<ELEM_BITS>();
        let bl = b & u32x16::splat((1 << limb_bits::<ELEM_BITS>()) - 1);
        let bh = b >> limb_bits::<ELEM_BITS>();
        let idxh = (ah << LIMB_EXPAND_BITS) + bh;
        let idxl = (al << limb_bits::<ELEM_BITS>()) + bl;
        for (ih, il) in idxh.as_array().iter().zip(idxl.as_array().iter()) {
            self.mults[*ih as usize].as_mut_slice()[*il as usize].0 += 1;
        }
    }
}

pub struct XorTableComponent<const ELEM_BITS: u32> {
    pub lookup_elements: LookupElements,
    pub claimed_sum: SecureField,
}

fn xor_table_info<const ELEM_BITS: u32>() -> InfoEvaluator {
    let mut counter = XorTableEval::<'_, _, ELEM_BITS> {
        eval: InfoEvaluator::default(),
        lookup_elements: &LookupElements::dummy(3),
        logup: LogupAtRow::new(1, SecureField::zero(), BaseField::zero()),
    };
    // Constant is_first column.
    counter.eval.next_interaction_mask(2, [0]);
    counter.eval()
}

impl<const ELEM_BITS: u32> Component for XorTableComponent<ELEM_BITS> {
    fn n_constraints(&self) -> usize {
        xor_table_info::<ELEM_BITS>().n_constraints
    }

    fn max_constraint_log_degree_bound(&self) -> u32 {
        column_bits::<ELEM_BITS>() + 1
    }

    fn trace_log_degree_bounds(&self) -> TreeVec<ColumnVec<u32>> {
        TreeVec::new(
            xor_table_info::<ELEM_BITS>()
                .mask_offsets
                .iter()
                .map(|tree_masks| vec![column_bits::<ELEM_BITS>(); tree_masks.len()])
                .collect(),
        )
    }

    fn mask_points(
        &self,
        point: CirclePoint<SecureField>,
    ) -> TreeVec<ColumnVec<Vec<CirclePoint<SecureField>>>> {
        let info = xor_table_info::<ELEM_BITS>();
        let trace_step = CanonicCoset::new(column_bits::<ELEM_BITS>()).step();
        info.mask_offsets.map(|tree_mask| {
            tree_mask
                .iter()
                .map(|col_mask| {
                    col_mask
                        .iter()
                        .map(|off| point + trace_step.mul_signed(*off).into_ef())
                        .collect()
                })
                .collect()
        })
    }

    fn evaluate_constraint_quotients_at_point(
        &self,
        point: CirclePoint<SecureField>,
        mask: &TreeVec<ColumnVec<Vec<SecureField>>>,
        evaluation_accumulator: &mut PointEvaluationAccumulator,
        _interaction_elements: &InteractionElements,
        _lookup_values: &LookupValues,
    ) {
        let constraint_zero_domain = CanonicCoset::new(column_bits::<ELEM_BITS>()).coset;
        let denom = coset_vanishing(constraint_zero_domain, point);
        let denom_inverse = denom.inverse();
        let mut eval = PointEvaluator::new(mask.as_ref(), evaluation_accumulator, denom_inverse);
        let [is_first] = eval.next_interaction_mask(2, [0]);
        let blake_eval = XorTableEval::<'_, _, ELEM_BITS> {
            eval,
            lookup_elements: &self.lookup_elements,
            logup: LogupAtRow::new(1, self.claimed_sum, is_first),
        };
        blake_eval.eval();
    }
}

impl<const ELEM_BITS: u32> ComponentProver<SimdBackend> for XorTableComponent<ELEM_BITS> {
    fn evaluate_constraint_quotients_on_domain(
        &self,
        trace: &ComponentTrace<'_, SimdBackend>,
        evaluation_accumulator: &mut DomainEvaluationAccumulator<SimdBackend>,
        _interaction_elements: &InteractionElements,
        _lookup_values: &LookupValues,
    ) {
        let mut domain_eval = DomainEvalHelper::new(
            column_bits::<ELEM_BITS>(),
            column_bits::<ELEM_BITS>() + 1,
            trace,
            evaluation_accumulator,
            self.max_constraint_log_degree_bound(),
            self.n_constraints(),
        );

        // TODO:
        let _span = span!(Level::INFO, "Constraint pointwise eval").entered();
        for vec_row in 0..(1 << (domain_eval.eval_domain.log_size() - LOG_N_LANES)) {
            let mut eval = SimdDomainEvaluator::new(
                &domain_eval.trace.evals,
                vec_row,
                &domain_eval.accum.random_coeff_powers,
                domain_eval.trace_domain.log_size,
                domain_eval.eval_domain.log_size(),
            );
            // Constant column is_first.
            let [is_first] = eval.next_interaction_mask(2, [0]);
            let logup = LogupAtRow::new(1, self.claimed_sum, is_first);
            let table_eval = XorTableEval::<'_, _, ELEM_BITS> {
                eval,
                lookup_elements: &self.lookup_elements,
                logup,
            };
            let eval = table_eval.eval();
            domain_eval.finalize_row(vec_row, eval.row_res);
        }
    }

    fn lookup_values(
        &self,
        _trace: &crate::core::air::ComponentTrace<'_, SimdBackend>,
    ) -> LookupValues {
        LookupValues::default()
    }
}

pub struct XorTableEval<'a, E: EvalAtRow, const ELEM_BITS: u32> {
    pub eval: E,
    pub lookup_elements: &'a LookupElements,
    pub logup: LogupAtRow<2, E>,
}
impl<'a, E: EvalAtRow, const ELEM_BITS: u32> XorTableEval<'a, E, ELEM_BITS> {
    pub fn eval(mut self) -> E {
        let [a] = self.eval.next_interaction_mask(2, [0]);
        let [b] = self.eval.next_interaction_mask(2, [0]);
        let [c] = self.eval.next_interaction_mask(2, [0]);
        for i in 0..1 << LIMB_EXPAND_BITS {
            for j in 0..1 << LIMB_EXPAND_BITS {
                let multiplicity = self.eval.next_trace_mask();

                let a =
                    a + E::F::from(BaseField::from_u32_unchecked(i << limb_bits::<ELEM_BITS>()));
                let b =
                    b + E::F::from(BaseField::from_u32_unchecked(j << limb_bits::<ELEM_BITS>()));
                let c = c + E::F::from(BaseField::from_u32_unchecked(
                    (i ^ j) << limb_bits::<ELEM_BITS>(),
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

pub struct XorTableLookupData<const ELEM_BITS: u32> {
    pub xor_accum: XorAccumulator<ELEM_BITS>,
}

pub fn generate_trace<const ELEM_BITS: u32>(
    xor_accum: XorAccumulator<ELEM_BITS>,
) -> (
    ColumnVec<CircleEvaluation<SimdBackend, BaseField, BitReversedOrder>>,
    XorTableLookupData<ELEM_BITS>,
) {
    (
        xor_accum
            .mults
            .iter()
            .map(|mult| {
                CircleEvaluation::new(
                    CanonicCoset::new(column_bits::<ELEM_BITS>()).circle_domain(),
                    mult.clone(),
                )
            })
            .collect_vec(),
        XorTableLookupData { xor_accum },
    )
}

#[allow(clippy::type_complexity)]
pub fn gen_interaction_trace<const ELEM_BITS: u32>(
    lookup_data: XorTableLookupData<ELEM_BITS>,
    lookup_elements: &LookupElements,
) -> (
    ColumnVec<CircleEvaluation<SimdBackend, BaseField, BitReversedOrder>>,
    ColumnVec<CircleEvaluation<SimdBackend, BaseField, BitReversedOrder>>,
    SecureField,
) {
    let limb_bits = limb_bits::<ELEM_BITS>();
    let _span = span!(Level::INFO, "Generate table trace").entered();
    let vec_off = u32x16::from_array(std::array::from_fn(|i| i as u32));
    let mut logup_gen = LogupTraceGenerator::new(column_bits::<ELEM_BITS>());
    for [(i0, mults0), (i1, mults1)] in lookup_data
        .xor_accum
        .mults
        .iter()
        .enumerate()
        .array_chunks::<2>()
    {
        let mut col_gen = logup_gen.new_col();
        let ah0 = i0 as u32 >> LIMB_EXPAND_BITS;
        let bh0 = i0 as u32 & ((1 << LIMB_EXPAND_BITS) - 1);
        let ah1 = i1 as u32 >> LIMB_EXPAND_BITS;
        let bh1 = i1 as u32 & ((1 << LIMB_EXPAND_BITS) - 1);

        #[allow(clippy::needless_range_loop)]
        for vec_row in 0..(1 << (column_bits::<ELEM_BITS>() - LOG_N_LANES)) {
            // vec_row is LIMB_BITS of a, and LIMB_BITS - LOG_N_LANES of b.
            let al = vec_row >> (limb_bits - LOG_N_LANES);
            let a0 = u32x16::splat((ah0 << limb_bits) | al);
            let a1 = u32x16::splat((ah1 << limb_bits) | al);
            let bm = vec_row & ((1 << (limb_bits - LOG_N_LANES)) - 1);
            let b0 = u32x16::splat((bh0 << limb_bits) | (bm << LOG_N_LANES)) | vec_off;
            let b1 = u32x16::splat((bh1 << limb_bits) | (bm << LOG_N_LANES)) | vec_off;

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

    let a_col: BaseFieldVec = (0..(1 << (column_bits::<ELEM_BITS>())))
        .map(|i| BaseField::from_u32_unchecked((i >> limb_bits) as u32))
        .collect();
    let b_col: BaseFieldVec = (0..(1 << (column_bits::<ELEM_BITS>())))
        .map(|i| BaseField::from_u32_unchecked((i & ((1 << limb_bits) - 1)) as u32))
        .collect();
    let c_col: BaseFieldVec = (0..(1 << (column_bits::<ELEM_BITS>())))
        .map(|i| {
            BaseField::from_u32_unchecked(((i >> limb_bits) ^ (i & ((1 << limb_bits) - 1))) as u32)
        })
        .collect();
    let constant_trace = [a_col, b_col, c_col]
        .map(|x| {
            CircleEvaluation::new(
                CanonicCoset::new(column_bits::<ELEM_BITS>()).circle_domain(),
                x,
            )
        })
        .to_vec();
    let (interaction_trace, claimed_sum) = logup_gen.finalize();
    (interaction_trace, constant_trace, claimed_sum)
}

#[test]
fn test_xor_table() {
    const ELEM_BITS: u32 = 10;
    let mut xor_accum = XorAccumulator::<ELEM_BITS>::default();
    xor_accum.add(u32x16::splat(1), u32x16::splat(2));
    let (trace, lookup_data) = generate_trace(xor_accum);
    let lookup_elements = LookupElements::dummy(3);
    let (interaction_trace, mut constant_trace, claimed_sum) =
        gen_interaction_trace(lookup_data, &lookup_elements);
    constant_trace.insert(
        0,
        crate::constraint_framework::constant_columns::gen_is_first(column_bits::<ELEM_BITS>()),
    );
    let trace = TreeVec::new(vec![trace, interaction_trace, constant_trace]);
    let trace_polys = trace.map_cols(|c| c.interpolate());
    crate::constraint_framework::assert_constraints(
        &trace_polys,
        CanonicCoset::new(column_bits::<ELEM_BITS>()),
        |mut eval| {
            let [is_first] = eval.next_interaction_mask(2, [0]);
            let logup = LogupAtRow::new(1, claimed_sum, is_first);
            let table_eval = XorTableEval::<'_, _, ELEM_BITS> {
                eval,
                lookup_elements: &lookup_elements,
                logup,
            };
            table_eval.eval();
        },
    )
}
