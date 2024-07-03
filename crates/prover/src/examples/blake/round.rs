use std::ops::{Add, AddAssign, Mul, Sub};
use std::simd::{u32x16, Simd};

use itertools::Itertools;
use num_traits::{One, Zero};
use tracing::{span, Level};

use super::Fu32;
use crate::core::air::accumulation::{DomainEvaluationAccumulator, PointEvaluationAccumulator};
use crate::core::air::mask::fixed_mask_points;
use crate::core::air::{Component, ComponentProver, ComponentTrace};
use crate::core::backend::simd::column::BaseFieldVec;
use crate::core::backend::simd::m31::{PackedBaseField, LOG_N_LANES};
use crate::core::backend::simd::qm31::PackedSecureField;
use crate::core::backend::simd::SimdBackend;
use crate::core::backend::{Col, Column, ColumnOps};
use crate::core::circle::CirclePoint;
use crate::core::constraints::coset_vanishing;
use crate::core::fields::m31::BaseField;
use crate::core::fields::qm31::SecureField;
use crate::core::fields::{FieldExpOps, FieldOps};
use crate::core::pcs::TreeVec;
use crate::core::poly::circle::{CanonicCoset, CircleEvaluation};
use crate::core::poly::BitReversedOrder;
use crate::core::{ColumnVec, InteractionElements};

// TODO: Fix.
pub struct BlakeRoundComponent {
    pub log_size: u32,
}

fn n_columns() -> usize {
    let mut counter = BlakeConstraintCounter::default();
    counter.eval();
    counter.n_cols
}

impl BlakeRoundComponent {
    pub fn new(log_size: u32) -> Self {
        Self { log_size }
    }
}

impl Component for BlakeRoundComponent {
    fn n_constraints(&self) -> usize {
        let mut counter = BlakeConstraintCounter::default();
        counter.eval();
        counter.n_constraints
    }

    fn max_constraint_log_degree_bound(&self) -> u32 {
        self.log_size + 1
    }

    fn n_interaction_phases(&self) -> u32 {
        // TODO(spapini): Add interaction.
        1
    }

    fn trace_log_degree_bounds(&self) -> TreeVec<ColumnVec<u32>> {
        TreeVec::new(vec![vec![self.log_size; n_columns()], vec![]])
    }

    fn mask_points(
        &self,
        point: CirclePoint<SecureField>,
    ) -> TreeVec<ColumnVec<Vec<CirclePoint<SecureField>>>> {
        let mut counter = BlakeConstraintCounter::default();
        counter.eval();
        TreeVec::new(vec![
            fixed_mask_points(&vec![vec![0_usize]; counter.n_cols], point),
            vec![],
        ])
    }

    fn interaction_element_ids(&self) -> Vec<String> {
        vec![]
    }

    fn evaluate_constraint_quotients_at_point(
        &self,
        point: CirclePoint<SecureField>,
        mask: &ColumnVec<Vec<SecureField>>,
        evaluation_accumulator: &mut PointEvaluationAccumulator,
        _interaction_elements: &InteractionElements,
    ) {
        let constraint_zero_domain = CanonicCoset::new(self.log_size).coset;
        let denom = coset_vanishing(constraint_zero_domain, point);
        let denom_inverse = denom.inverse();
        let two = SecureField::one() + SecureField::one();
        let mut eval = BlakeEvalAtPoint {
            mask,
            evaluation_accumulator,
            col_index: 0,
            denom_inverse,
            pow2: &(0..32).map(|i| two.pow(i)).collect::<Vec<_>>(),
        };
        eval.eval();
        assert_eq!(eval.col_index, n_columns());
    }
}

#[derive(Default)]
struct BlakeConstraintCounter {
    n_cols: usize,
    n_constraints: usize,
}
impl BlakeRoundEval for BlakeConstraintCounter {
    type F = BaseField;
    fn next_mask(&mut self) -> Self::F {
        self.n_cols += 1;
        BaseField::one()
    }
    fn add_constraint(&mut self, _constraint: Self::F) {
        self.n_constraints += 1;
    }
    fn pow2(&self, _i: u32) -> Self::F {
        BaseField::one()
    }
}

struct BlakeEvalAtPoint<'a> {
    mask: &'a ColumnVec<Vec<SecureField>>,
    evaluation_accumulator: &'a mut PointEvaluationAccumulator,
    col_index: usize,
    denom_inverse: SecureField,
    pow2: &'a [SecureField],
}
impl<'a> BlakeRoundEval for BlakeEvalAtPoint<'a> {
    type F = SecureField;

    fn next_mask(&mut self) -> Self::F {
        let res = self.mask[self.col_index][0];
        self.col_index += 1;
        res
    }
    fn add_constraint(&mut self, constraint: Self::F) {
        self.evaluation_accumulator
            .accumulate(constraint * self.denom_inverse);
    }
    fn pow2(&self, i: u32) -> Self::F {
        self.pow2[i as usize]
    }
}

// TODO: turn all divisions into muls.
trait BlakeRoundEval {
    type F: FieldExpOps
        + Copy
        + AddAssign<Self::F>
        + Add<Self::F, Output = Self::F>
        + Sub<Self::F, Output = Self::F>
        + Mul<BaseField, Output = Self::F>;

    fn next_mask(&mut self) -> Self::F;
    fn add_constraint(&mut self, constraint: Self::F);
    fn pow2(&self, i: u32) -> Self::F;

    fn next_u32(&mut self) -> Fu32<Self::F> {
        let l = self.next_mask();
        let h = self.next_mask();
        Fu32 { l, h }
    }
    fn eval(&mut self) {
        let mut v: [Fu32<Self::F>; 16] = std::array::from_fn(|_| self.next_u32());
        // let _input_v = v.clone();
        self.g(0, v.get_many_mut([0, 4, 8, 12]).unwrap());
        self.g(1, v.get_many_mut([1, 5, 9, 13]).unwrap());
        self.g(2, v.get_many_mut([2, 6, 10, 14]).unwrap());
        self.g(3, v.get_many_mut([3, 7, 11, 15]).unwrap());
        self.g(4, v.get_many_mut([0, 5, 10, 15]).unwrap());
        self.g(5, v.get_many_mut([1, 6, 11, 12]).unwrap());
        self.g(6, v.get_many_mut([2, 7, 8, 13]).unwrap());
        self.g(7, v.get_many_mut([3, 4, 9, 14]).unwrap());

        // TODO: yield BlakeRound(round, input_v, v);
    }
    fn g(&mut self, _round: u32, v: [&mut Fu32<Self::F>; 4]) {
        let [a, b, c, d] = v;
        // TODO: lookup m0, m1.
        let m0 = self.next_u32();
        let m1 = self.next_u32();

        *a = self.add3_u32_unchecked(*a, *b, m0);
        *d = self.xor_rotr16_u32(*a, *d);
        *c = self.add2_u32_unchecked(*c, *d);
        *b = self.xor_rotr_u32(*b, *c, 12);
        *a = self.add3_u32_unchecked(*a, *b, m1);
        *d = self.xor_rotr_u32(*a, *d, 8);
        *c = self.add2_u32_unchecked(*c, *d);
        *b = self.xor_rotr_u32(*b, *c, 7);
    }

    /// Adds two u32s, returning the sum.
    /// Assumes a, b are properly range checked.
    /// The caller is responsible for checking:
    /// res.{l,h} not in [2^16, 2^17) or in [-2^16,0)
    fn add2_u32_unchecked(&mut self, a: Fu32<Self::F>, b: Fu32<Self::F>) -> Fu32<Self::F> {
        let carry_l = self.next_mask();
        self.add_constraint(carry_l * carry_l - carry_l);

        let carry_h = self.next_mask();
        self.add_constraint(carry_h * carry_h - carry_h);

        Fu32 {
            l: a.l + b.l - carry_l * self.pow2(16),
            h: a.h + b.h + carry_l - carry_h * self.pow2(16),
        }
    }

    /// Adds three u32s, returning the sum.
    /// Assumes a, b, c are properly range checked.
    /// Caller is responsible for checking:
    /// res.{l,h} not in [2^16, 3*2^16) or in [-2^17,0)
    fn add3_u32_unchecked(
        &mut self,
        a: Fu32<Self::F>,
        b: Fu32<Self::F>,
        c: Fu32<Self::F>,
    ) -> Fu32<Self::F> {
        let carry_l = self.next_mask();
        self.add_constraint(carry_l * (carry_l - self.pow2(0)) * (carry_l - self.pow2(1)));

        let carry_h = self.next_mask();
        self.add_constraint(carry_h * (carry_h - self.pow2(0)) * (carry_h - self.pow2(1)));

        Fu32 {
            l: a.l + b.l + c.l - carry_l * self.pow2(16),
            h: a.h + b.h + c.h + carry_l - carry_h * self.pow2(16),
        }
    }

    /// Splits a felt at r.
    /// Caller is responsible for checking that the ranges of h * 2^r and l don't overlap.
    fn split_unchecked(&mut self, a: Self::F, r: u32) -> (Self::F, Self::F) {
        let h = self.next_mask();
        let l = a - h * self.pow2(r);
        (l, h)
    }

    /// Checks that a, b are in range, and computes their xor rotate.
    fn xor_rotr_u32(&mut self, a: Fu32<Self::F>, b: Fu32<Self::F>, r: u32) -> Fu32<Self::F> {
        let (all, alh) = self.split_unchecked(a.l, r);
        let (ahl, ahh) = self.split_unchecked(a.h, r);
        let (bll, blh) = self.split_unchecked(b.l, r);
        let (bhl, bhh) = self.split_unchecked(b.h, r);

        // These also guarantee that all elements are in range.
        let xorll = self.xor(r, all, bll);
        let xorlh = self.xor(16 - r, alh, blh);
        let xorhl = self.xor(r, ahl, bhl);
        let xorhh = self.xor(16 - r, ahh, bhh);

        Fu32 {
            l: xorhl * self.pow2(16 - r) + xorlh,
            h: xorll * self.pow2(16 - r) + xorhh,
        }
    }

    /// Checks that a, b are in range, and computes their xor 16 rotate.
    fn xor_rotr16_u32(&mut self, a: Fu32<Self::F>, b: Fu32<Self::F>) -> Fu32<Self::F> {
        let (all, alh) = self.split_unchecked(a.l, 8);
        let (ahl, ahh) = self.split_unchecked(a.h, 8);
        let (bll, blh) = self.split_unchecked(b.l, 8);
        let (bhl, bhh) = self.split_unchecked(b.h, 8);

        // These also guarantee that all elements are in range.
        let xorll = self.xor(8, all, bll);
        let xorlh = self.xor(8, alh, blh);
        let xorhl = self.xor(8, ahl, bhl);
        let xorhh = self.xor(8, ahh, bhh);

        Fu32 {
            l: xorhh * self.pow2(8) + xorhl,
            h: xorlh * self.pow2(8) + xorll,
        }
    }

    /// Checks that a,b in in [0,2^w) and computes their xor.
    fn xor(&mut self, _w: u32, _a: Self::F, _b: Self::F) -> Self::F {
        // TODO: use Xor(w, a, b, res);
        self.next_mask()
    }
}

struct BlakeRoundEvalAtDomain<'a> {
    trace_eval: &'a TreeVec<Vec<&'a CircleEvaluation<SimdBackend, BaseField, BitReversedOrder>>>,
    vec_row: usize,
    random_coeff_powers: &'a [SecureField],
    row_res: PackedSecureField,
    col_index: usize,
    constraint_index: usize,
}
impl<'a> BlakeRoundEval for BlakeRoundEvalAtDomain<'a> {
    type F = PackedBaseField;

    fn next_mask(&mut self) -> Self::F {
        let res = unsafe {
            *self.trace_eval[0]
                .get_unchecked(self.col_index)
                .data
                .get_unchecked(self.vec_row)
        };
        self.col_index += 1;
        res
    }
    fn add_constraint(&mut self, constraint: Self::F) {
        self.row_res +=
            PackedSecureField::broadcast(self.random_coeff_powers[self.constraint_index])
                * constraint;
        self.constraint_index += 1;
    }

    fn pow2(&self, i: u32) -> Self::F {
        PackedBaseField::broadcast(BaseField::from_u32_unchecked(1 << i))
    }
}

impl ComponentProver<SimdBackend> for BlakeRoundComponent {
    fn evaluate_constraint_quotients_on_domain(
        &self,
        trace: &ComponentTrace<'_, SimdBackend>,
        evaluation_accumulator: &mut DomainEvaluationAccumulator<SimdBackend>,
        _interaction_elements: &InteractionElements,
    ) {
        let eval_domain = CanonicCoset::new(self.log_size + 1).circle_domain();

        // Create a new evaluation.
        let trace_eval = &trace.evals;

        // Denoms.
        let span = span!(Level::INFO, "Constraint eval denominators").entered();
        // TODO(spapini): Make this prettier.
        let zero_domain = CanonicCoset::new(self.log_size).coset;
        let mut denoms =
            BaseFieldVec::from_iter(eval_domain.iter().map(|p| coset_vanishing(zero_domain, p)));
        <SimdBackend as ColumnOps<BaseField>>::bit_reverse_column(&mut denoms);
        let mut denom_inverses = BaseFieldVec::zeros(denoms.len());
        <SimdBackend as FieldOps<BaseField>>::batch_inverse(&denoms, &mut denom_inverses);
        span.exit();

        let _span = span!(Level::INFO, "Constraint pointwise eval").entered();

        let constraint_log_degree_bound = self.max_constraint_log_degree_bound();
        let n_constraints = self.n_constraints();
        let [accum] =
            evaluation_accumulator.columns([(constraint_log_degree_bound, n_constraints)]);
        let mut pows = accum.random_coeff_powers.clone();
        pows.reverse();

        for vec_row in 0..(1 << (eval_domain.log_size() - LOG_N_LANES)) {
            let mut evaluator = BlakeRoundEvalAtDomain {
                trace_eval,
                vec_row,
                random_coeff_powers: &pows,
                row_res: PackedSecureField::zero(),
                col_index: 0,
                constraint_index: 0,
            };
            evaluator.eval();

            let row_res = evaluator.row_res;

            unsafe {
                accum.col.set_packed(
                    vec_row,
                    accum.col.packed_at(vec_row) + row_res * denom_inverses.data[vec_row],
                )
            }
            assert_eq!(evaluator.constraint_index, n_constraints);
        }
    }
}

pub struct BlakeTraceGenerator {
    trace: Vec<BaseFieldVec>,
    vec_row: usize,
    col_index: usize,
}
impl BlakeTraceGenerator {
    pub fn gen_trace(
        log_size: u32,
    ) -> ColumnVec<CircleEvaluation<SimdBackend, BaseField, BitReversedOrder>> {
        assert!(log_size >= LOG_N_LANES);
        let trace = (0..n_columns())
            .map(|_| Col::<SimdBackend, BaseField>::zeros(1 << log_size))
            .collect_vec();

        let mut gen = Self {
            trace,
            vec_row: 0,
            col_index: 0,
        };
        for vec_row in 0..(1 << (log_size - LOG_N_LANES)) {
            gen.vec_row = vec_row;
            gen.col_index = 0;

            let v: [u32x16; 16] = std::array::from_fn(|state_i| {
                Simd::from_array(std::array::from_fn(|i| (vec_row * 16 + i + state_i) as u32))
            });
            let m: [u32x16; 16] = std::array::from_fn(|state_i| {
                Simd::from_array(std::array::from_fn(|i| {
                    (vec_row * 16 + i + state_i + 100) as u32
                }))
            });

            gen.gen_row(v, m);
        }
        let domain = CanonicCoset::new(log_size).circle_domain();
        gen.trace
            .into_iter()
            .map(|eval| CircleEvaluation::<SimdBackend, _, BitReversedOrder>::new(domain, eval))
            .collect_vec()
    }

    fn append_felt(&mut self, val: u32x16) {
        self.trace[self.col_index].data[self.vec_row] =
            unsafe { PackedBaseField::from_simd_unchecked(val) };
        self.col_index += 1;
    }

    fn append_u32(&mut self, val: [u32x16; 2]) {
        self.append_felt(val[0]);
        self.append_felt(val[1]);
    }

    fn gen_row(&mut self, v: [u32x16; 16], m: [u32x16; 16]) {
        let mut v = v.map(|state| [state & u32x16::splat(0xffff), state >> 16]);
        let m = m.map(|state| [state & u32x16::splat(0xffff), state >> 16]);
        v.iter().for_each(|s| {
            self.append_u32(*s);
        });

        self.g(0, v.get_many_mut([0, 4, 8, 12]).unwrap(), m[0], m[1]);
        self.g(1, v.get_many_mut([1, 5, 9, 13]).unwrap(), m[2], m[3]);
        self.g(2, v.get_many_mut([2, 6, 10, 14]).unwrap(), m[4], m[5]);
        self.g(3, v.get_many_mut([3, 7, 11, 15]).unwrap(), m[6], m[7]);
        self.g(4, v.get_many_mut([0, 5, 10, 15]).unwrap(), m[8], m[9]);
        self.g(5, v.get_many_mut([1, 6, 11, 12]).unwrap(), m[10], m[11]);
        self.g(6, v.get_many_mut([2, 7, 8, 13]).unwrap(), m[12], m[13]);
        self.g(7, v.get_many_mut([3, 4, 9, 14]).unwrap(), m[14], m[15]);
    }

    fn g(&mut self, _round: u32, v: [&mut [u32x16; 2]; 4], m0: [u32x16; 2], m1: [u32x16; 2]) {
        let [a, b, c, d] = v;
        self.append_u32(m0);
        self.append_u32(m1);

        *a = self.add3_u32s(*a, *b, m0);
        *d = self.xor_rotr16_u32(*a, *d);
        *c = self.add2_u32s(*c, *d);
        *b = self.xor_rotr_u32(*b, *c, 12);
        *a = self.add3_u32s(*a, *b, m1);
        *d = self.xor_rotr_u32(*a, *d, 8);
        *c = self.add2_u32s(*c, *d);
        *b = self.xor_rotr_u32(*b, *c, 7);
    }

    fn add2_u32s(&mut self, a: [u32x16; 2], b: [u32x16; 2]) -> [u32x16; 2] {
        let sl = a[0] + b[0];
        let carryl = sl >> 16;
        self.append_felt(carryl);

        let sh = a[1] + b[1] + carryl;
        let carryh = sh >> 16;
        self.append_felt(carryh);

        [sl & u32x16::splat(0xffff), sh & u32x16::splat(0xffff)]
    }

    fn add3_u32s(&mut self, a: [u32x16; 2], b: [u32x16; 2], c: [u32x16; 2]) -> [u32x16; 2] {
        let sl = a[0] + b[0] + c[0];
        let carryl = sl >> 16;
        self.append_felt(carryl);

        let sh = a[1] + b[1] + c[1] + carryl;
        let carryh = sh >> 16;
        self.append_felt(carryh);

        [sl & u32x16::splat(0xffff), sh & u32x16::splat(0xffff)]
    }

    fn xor_rotr_u32(&mut self, a: [u32x16; 2], b: [u32x16; 2], r: u32) -> [u32x16; 2] {
        let (all, alh) = self.split(a[0], r);
        let (ahl, ahh) = self.split(a[1], r);
        let (bll, blh) = self.split(b[0], r);
        let (bhl, bhh) = self.split(b[1], r);

        // These also guarantee that all elements are in range.
        let xorll = self.xor(r, all, bll);
        let xorlh = self.xor(16 - r, alh, blh);
        let xorhl = self.xor(r, ahl, bhl);
        let xorhh = self.xor(16 - r, ahh, bhh);

        [(xorhl << (16 - r)) + xorlh, (xorll << (16 - r)) + xorhh]
    }

    fn xor_rotr16_u32(&mut self, a: [u32x16; 2], b: [u32x16; 2]) -> [u32x16; 2] {
        let (all, alh) = self.split(a[0], 8);
        let (ahl, ahh) = self.split(a[1], 8);
        let (bll, blh) = self.split(b[0], 8);
        let (bhl, bhh) = self.split(b[1], 8);

        // These also guarantee that all elements are in range.
        let xorll = self.xor(8, all, bll);
        let xorlh = self.xor(8, alh, blh);
        let xorhl = self.xor(8, ahl, bhl);
        let xorhh = self.xor(8, ahh, bhh);

        [(xorhh << 8) + xorhl, (xorlh << 8) + xorll]
    }

    fn split(&mut self, a: u32x16, r: u32) -> (u32x16, u32x16) {
        let h = a >> r;
        let l = a & u32x16::splat((1 << r) - 1);
        self.append_felt(h);
        (l, h)
    }

    fn xor(&mut self, _w: u32, a: u32x16, b: u32x16) -> u32x16 {
        let res = a ^ b;
        self.append_felt(res);
        res
    }
}
