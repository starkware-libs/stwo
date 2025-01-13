use itertools::{chain, Itertools};
use num_traits::One;

use super::{BlakeXorElements, RoundElements};
use crate::constraint_framework::{EvalAtRow, RelationEntry};
use crate::core::fields::m31::BaseField;
use crate::core::fields::qm31::SecureField;
use crate::examples::blake::{Fu32, STATE_SIZE};

const INV16: BaseField = BaseField::from_u32_unchecked(1 << 15);
const TWO: BaseField = BaseField::from_u32_unchecked(2);

pub struct BlakeRoundEval<'a, E: EvalAtRow> {
    pub eval: E,
    pub xor_lookup_elements: &'a BlakeXorElements,
    pub round_lookup_elements: &'a RoundElements,
    // TODO(first): validate logup.
    pub _claimed_sum: SecureField,
    pub _log_size: u32,
}
impl<E: EvalAtRow> BlakeRoundEval<'_, E> {
    pub fn eval(mut self) -> E {
        let mut v: [Fu32<E::F>; STATE_SIZE] = std::array::from_fn(|_| self.next_u32());
        let input_v = v.clone();
        let m: [Fu32<E::F>; STATE_SIZE] = std::array::from_fn(|_| self.next_u32());

        self.g(
            v.get_many_mut([0, 4, 8, 12]).unwrap(),
            m[0].clone(),
            m[1].clone(),
        );
        self.g(
            v.get_many_mut([1, 5, 9, 13]).unwrap(),
            m[2].clone(),
            m[3].clone(),
        );
        self.g(
            v.get_many_mut([2, 6, 10, 14]).unwrap(),
            m[4].clone(),
            m[5].clone(),
        );
        self.g(
            v.get_many_mut([3, 7, 11, 15]).unwrap(),
            m[6].clone(),
            m[7].clone(),
        );
        self.g(
            v.get_many_mut([0, 5, 10, 15]).unwrap(),
            m[8].clone(),
            m[9].clone(),
        );
        self.g(
            v.get_many_mut([1, 6, 11, 12]).unwrap(),
            m[10].clone(),
            m[11].clone(),
        );
        self.g(
            v.get_many_mut([2, 7, 8, 13]).unwrap(),
            m[12].clone(),
            m[13].clone(),
        );
        self.g(
            v.get_many_mut([3, 4, 9, 14]).unwrap(),
            m[14].clone(),
            m[15].clone(),
        );

        // Yield `Round(input_v, output_v, message)`.
        self.eval.add_to_relation(RelationEntry::new(
            self.round_lookup_elements,
            -E::EF::one(),
            &chain![
                input_v.iter().cloned().flat_map(Fu32::into_felts),
                v.iter().cloned().flat_map(Fu32::into_felts),
                m.iter().cloned().flat_map(Fu32::into_felts)
            ]
            .collect_vec(),
        ));

        self.eval.finalize_logup_in_pairs();
        self.eval
    }
    fn next_u32(&mut self) -> Fu32<E::F> {
        let l = self.eval.next_trace_mask();
        let h = self.eval.next_trace_mask();
        Fu32 { l, h }
    }
    fn g(&mut self, v: [&mut Fu32<E::F>; 4], m0: Fu32<E::F>, m1: Fu32<E::F>) {
        let [a, b, c, d] = v;

        *a = self.add3_u32_unchecked(a.clone(), b.clone(), m0);
        *d = self.xor_rotr16_u32(a.clone(), d.clone());
        *c = self.add2_u32_unchecked(c.clone(), d.clone());
        *b = self.xor_rotr_u32(b.clone(), c.clone(), 12);
        *a = self.add3_u32_unchecked(a.clone(), b.clone(), m1);
        *d = self.xor_rotr_u32(a.clone(), d.clone(), 8);
        *c = self.add2_u32_unchecked(c.clone(), d.clone());
        *b = self.xor_rotr_u32(b.clone(), c.clone(), 7);
    }

    /// Adds two u32s, returning the sum.
    /// Assumes a, b are properly range checked.
    /// The caller is responsible for checking:
    /// res.{l,h} not in [2^16, 2^17) or in [-2^16,0)
    fn add2_u32_unchecked(&mut self, a: Fu32<E::F>, b: Fu32<E::F>) -> Fu32<E::F> {
        let sl = self.eval.next_trace_mask();
        let sh = self.eval.next_trace_mask();

        let carry_l = (a.l + b.l - sl.clone()) * E::F::from(INV16);
        self.eval
            .add_constraint(carry_l.clone() * carry_l.clone() - carry_l.clone());

        let carry_h = (a.h + b.h + carry_l - sh.clone()) * E::F::from(INV16);
        self.eval
            .add_constraint(carry_h.clone() * carry_h.clone() - carry_h.clone());

        Fu32 { l: sl, h: sh }
    }

    /// Adds three u32s, returning the sum.
    /// Assumes a, b, c are properly range checked.
    /// Caller is responsible for checking:
    /// res.{l,h} not in [2^16, 3*2^16) or in [-2^17,0)
    fn add3_u32_unchecked(&mut self, a: Fu32<E::F>, b: Fu32<E::F>, c: Fu32<E::F>) -> Fu32<E::F> {
        let sl = self.eval.next_trace_mask();
        let sh = self.eval.next_trace_mask();

        let carry_l = (a.l + b.l + c.l - sl.clone()) * E::F::from(INV16);
        self.eval.add_constraint(
            carry_l.clone() * (carry_l.clone() - E::F::one()) * (carry_l.clone() - E::F::from(TWO)),
        );

        let carry_h = (a.h + b.h + c.h + carry_l - sh.clone()) * E::F::from(INV16);
        self.eval.add_constraint(
            carry_h.clone() * (carry_h.clone() - E::F::one()) * (carry_h.clone() - E::F::from(TWO)),
        );

        Fu32 {
            l: sl,
            h: sh.clone(),
        }
    }

    /// Splits a felt at r.
    /// Caller is responsible for checking that the ranges of h * 2^r and l don't overlap.
    fn split_unchecked(&mut self, a: E::F, r: u32) -> (E::F, E::F) {
        let h = self.eval.next_trace_mask();
        let l = a - h.clone() * E::F::from(BaseField::from_u32_unchecked(1 << r));
        (l, h)
    }

    /// Checks that a, b are in range, and computes their xor rotated right by `r` bits.
    /// Guarantees that all elements are in range.
    fn xor_rotr_u32(&mut self, a: Fu32<E::F>, b: Fu32<E::F>, r: u32) -> Fu32<E::F> {
        let (all, alh) = self.split_unchecked(a.l, r);
        let (ahl, ahh) = self.split_unchecked(a.h, r);
        let (bll, blh) = self.split_unchecked(b.l, r);
        let (bhl, bhh) = self.split_unchecked(b.h, r);

        // These also guarantee that all elements are in range.
        let [xorll, xorhl] = self.xor2(r, [all, ahl], [bll, bhl]);
        let [xorlh, xorhh] = self.xor2(16 - r, [alh, ahh], [blh, bhh]);

        Fu32 {
            l: xorhl * E::F::from(BaseField::from_u32_unchecked(1 << (16 - r))) + xorlh,
            h: xorll * E::F::from(BaseField::from_u32_unchecked(1 << (16 - r))) + xorhh,
        }
    }

    /// Checks that a, b are in range, and computes their xor rotated right by 16 bits.
    /// Guarantees that all elements are in range.
    fn xor_rotr16_u32(&mut self, a: Fu32<E::F>, b: Fu32<E::F>) -> Fu32<E::F> {
        let (all, alh) = self.split_unchecked(a.l, 8);
        let (ahl, ahh) = self.split_unchecked(a.h, 8);
        let (bll, blh) = self.split_unchecked(b.l, 8);
        let (bhl, bhh) = self.split_unchecked(b.h, 8);

        // These also guarantee that all elements are in range.
        let [xorll, xorhl] = self.xor2(8, [all, ahl], [bll, bhl]);
        let [xorlh, xorhh] = self.xor2(8, [alh, ahh], [blh, bhh]);

        Fu32 {
            l: xorhh * E::F::from(BaseField::from_u32_unchecked(1 << 8)) + xorhl,
            h: xorlh * E::F::from(BaseField::from_u32_unchecked(1 << 8)) + xorll,
        }
    }

    /// Checks that a, b are in [0, 2^w) and computes their xor.
    fn xor2(&mut self, w: u32, a: [E::F; 2], b: [E::F; 2]) -> [E::F; 2] {
        // TODO: Separate lookups by w.
        let c = [self.eval.next_trace_mask(), self.eval.next_trace_mask()];

        let xor_lookup_elements = self.xor_lookup_elements;

        xor_lookup_elements.use_relation(
            &mut self.eval,
            w,
            [
                &[a[0].clone(), b[0].clone(), c[0].clone()],
                &[a[1].clone(), b[1].clone(), c[1].clone()],
            ],
        );

        c
    }
}
