use num_traits::{One, Zero};

use super::eval::EvalAtRow;
use super::lookup::LogupAtRow;
use super::{Fu32, LookupElements};
use crate::core::utils::shifted_secure_combination;

pub struct BlakeEvalAtRow<'a, E: EvalAtRow> {
    pub eval: E,
    pub lookup_elements: LookupElements,
    pub xor_logup: LogupAtRow<'a, E>,
}
impl<'a, E: EvalAtRow> BlakeEvalAtRow<'a, E> {
    pub fn eval(&mut self) {
        let mut v: [Fu32<E::F>; 16] = std::array::from_fn(|_| self.next_u32());
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
    fn next_u32(&mut self) -> Fu32<E::F> {
        let l = self.eval.next_mask();
        let h = self.eval.next_mask();
        Fu32 { l, h }
    }
    fn g(&mut self, _round: u32, v: [&mut Fu32<E::F>; 4]) {
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
    fn add2_u32_unchecked(&mut self, a: Fu32<E::F>, b: Fu32<E::F>) -> Fu32<E::F> {
        let carry_l = self.eval.next_mask();
        self.eval.add_constraint(carry_l * carry_l - carry_l);

        let carry_h = self.eval.next_mask();
        self.eval.add_constraint(carry_h * carry_h - carry_h);

        Fu32 {
            l: a.l + b.l - carry_l * self.eval.pow2(16),
            h: a.h + b.h + carry_l - carry_h * self.eval.pow2(16),
        }
    }

    /// Adds three u32s, returning the sum.
    /// Assumes a, b, c are properly range checked.
    /// Caller is responsible for checking:
    /// res.{l,h} not in [2^16, 3*2^16) or in [-2^17,0)
    fn add3_u32_unchecked(&mut self, a: Fu32<E::F>, b: Fu32<E::F>, c: Fu32<E::F>) -> Fu32<E::F> {
        let carry_l = self.eval.next_mask();
        self.eval.add_constraint(
            carry_l * (carry_l - self.eval.pow2(0)) * (carry_l - self.eval.pow2(1)),
        );

        let carry_h = self.eval.next_mask();
        self.eval.add_constraint(
            carry_h * (carry_h - self.eval.pow2(0)) * (carry_h - self.eval.pow2(1)),
        );

        Fu32 {
            l: a.l + b.l + c.l - carry_l * self.eval.pow2(16),
            h: a.h + b.h + c.h + carry_l - carry_h * self.eval.pow2(16),
        }
    }

    /// Splits a felt at r.
    /// Caller is responsible for checking that the ranges of h * 2^r and l don't overlap.
    fn split_unchecked(&mut self, a: E::F, r: u32) -> (E::F, E::F) {
        let h = self.eval.next_mask();
        let l = a - h * self.eval.pow2(r);
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
            l: xorhl * self.eval.pow2(16 - r) + xorlh,
            h: xorll * self.eval.pow2(16 - r) + xorhh,
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
            l: xorhh * self.eval.pow2(8) + xorhl,
            h: xorlh * self.eval.pow2(8) + xorll,
        }
    }

    /// Checks that a,b in in [0,2^w) and computes their xor.
    fn xor(&mut self, _w: u32, a: E::F, b: E::F) -> E::F {
        // TODO: Separate lookups by w.
        let c = self.eval.next_mask();
        let LookupElements { z, alpha } = self.lookup_elements;
        let shifted_value =
            shifted_secure_combination(&[a, b, c], E::EF::zero() + alpha, E::EF::zero() + z);
        self.xor_logup
            .push(&mut self.eval, E::EF::one(), shifted_value);
        c
    }
}
