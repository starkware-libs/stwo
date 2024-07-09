use itertools::{chain, Itertools};
use num_traits::One;

use super::Fu32;
use crate::constraint_framework::logup::{LogupAtRow, LookupElements};
use crate::constraint_framework::EvalAtRow;
use crate::core::fields::m31::BaseField;

pub struct BlakeRoundEval<'a, E: EvalAtRow> {
    pub eval: E,
    pub xor_lookup_elements: &'a LookupElements,
    pub round_lookup_elements: &'a LookupElements,
    pub logup: LogupAtRow<2, E>,
}
impl<'a, E: EvalAtRow> BlakeRoundEval<'a, E> {
    pub fn eval(mut self) -> E {
        let mut v: [Fu32<E::F>; 16] = std::array::from_fn(|_| self.next_u32());
        let input_v = v;
        let m: [Fu32<E::F>; 16] = std::array::from_fn(|_| self.next_u32());

        self.g(0, v.get_many_mut([0, 4, 8, 12]).unwrap(), m[0], m[1]);
        self.g(1, v.get_many_mut([1, 5, 9, 13]).unwrap(), m[2], m[3]);
        self.g(2, v.get_many_mut([2, 6, 10, 14]).unwrap(), m[4], m[5]);
        self.g(3, v.get_many_mut([3, 7, 11, 15]).unwrap(), m[6], m[7]);
        self.g(4, v.get_many_mut([0, 5, 10, 15]).unwrap(), m[8], m[9]);
        self.g(5, v.get_many_mut([1, 6, 11, 12]).unwrap(), m[10], m[11]);
        self.g(6, v.get_many_mut([2, 7, 8, 13]).unwrap(), m[12], m[13]);
        self.g(7, v.get_many_mut([3, 4, 9, 14]).unwrap(), m[14], m[15]);

        self.logup.push_lookup(
            &mut self.eval,
            -E::EF::one(),
            &chain![
                input_v.iter().copied().flat_map(Fu32::to_felts),
                v.iter().copied().flat_map(Fu32::to_felts),
                m.iter().copied().flat_map(Fu32::to_felts)
            ]
            .collect_vec(),
            self.round_lookup_elements,
        );

        self.logup.finalize(&mut self.eval);
        self.eval
    }
    fn next_u32(&mut self) -> Fu32<E::F> {
        let l = self.eval.next_mask();
        let h = self.eval.next_mask();
        Fu32 { l, h }
    }
    fn g(&mut self, _round: u32, v: [&mut Fu32<E::F>; 4], m0: Fu32<E::F>, m1: Fu32<E::F>) {
        let [a, b, c, d] = v;

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
    fn add2_u32_unchecked(&mut self, a: Fu32<E::F>, b: Fu32<E::F>) -> Fu32<E::F> {
        let carry_l = self.eval.next_mask();
        self.eval.add_constraint(carry_l * carry_l - carry_l);

        let carry_h = self.eval.next_mask();
        self.eval.add_constraint(carry_h * carry_h - carry_h);

        Fu32 {
            l: a.l + b.l - carry_l * E::F::from(BaseField::from_u32_unchecked(1 << 16)),
            h: a.h + b.h + carry_l - carry_h * E::F::from(BaseField::from_u32_unchecked(1 << 16)),
        }
    }

    /// Adds three u32s, returning the sum.
    /// Assumes a, b, c are properly range checked.
    /// Caller is responsible for checking:
    /// res.{l,h} not in [2^16, 3*2^16) or in [-2^17,0)
    fn add3_u32_unchecked(&mut self, a: Fu32<E::F>, b: Fu32<E::F>, c: Fu32<E::F>) -> Fu32<E::F> {
        let carry_l = self.eval.next_mask();
        self.eval.add_constraint(
            carry_l
                * (carry_l - E::F::from(BaseField::from_u32_unchecked(1 << 0)))
                * (carry_l - E::F::from(BaseField::from_u32_unchecked(1 << 1))),
        );

        let carry_h = self.eval.next_mask();
        self.eval.add_constraint(
            carry_h
                * (carry_h - E::F::from(BaseField::from_u32_unchecked(1 << 0)))
                * (carry_h - E::F::from(BaseField::from_u32_unchecked(1 << 1))),
        );

        Fu32 {
            l: a.l + b.l + c.l - carry_l * E::F::from(BaseField::from_u32_unchecked(1 << 16)),
            h: a.h + b.h + c.h + carry_l
                - carry_h * E::F::from(BaseField::from_u32_unchecked(1 << 16)),
        }
    }

    /// Splits a felt at r.
    /// Caller is responsible for checking that the ranges of h * 2^r and l don't overlap.
    fn split_unchecked(&mut self, a: E::F, r: u32) -> (E::F, E::F) {
        let h = self.eval.next_mask();
        let l = a - h * E::F::from(BaseField::from_u32_unchecked(1 << r));
        (l, h)
    }

    /// Checks that a, b are in range, and computes their xor rotate.
    fn xor_rotr_u32(&mut self, a: Fu32<E::F>, b: Fu32<E::F>, r: u32) -> Fu32<E::F> {
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
            l: xorhl * E::F::from(BaseField::from_u32_unchecked(1 << (16 - r))) + xorlh,
            h: xorll * E::F::from(BaseField::from_u32_unchecked(1 << (16 - r))) + xorhh,
        }
    }

    /// Checks that a, b are in range, and computes their xor 16 rotate.
    fn xor_rotr16_u32(&mut self, a: Fu32<E::F>, b: Fu32<E::F>) -> Fu32<E::F> {
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
            l: xorhh * E::F::from(BaseField::from_u32_unchecked(1 << 8)) + xorhl,
            h: xorlh * E::F::from(BaseField::from_u32_unchecked(1 << 8)) + xorll,
        }
    }

    /// Checks that a,b in in [0,2^w) and computes their xor.
    fn xor(&mut self, _w: u32, a: E::F, b: E::F) -> E::F {
        // TODO: Separate lookups by w.
        let c = self.eval.next_mask();
        self.logup.push_lookup(
            &mut self.eval,
            E::EF::one(),
            &[a, b, c],
            self.xor_lookup_elements,
        );
        c
    }
}
