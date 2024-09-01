use itertools::{chain, Itertools};
use num_traits::One;

use super::{BlakeXorElements, RoundElements};
use crate::constraint_framework::logup::LogupAtRow;
use crate::constraint_framework::EvalAtRow;
use crate::core::fields::m31::BaseField;
use crate::core::lookups::utils::Fraction;
use crate::examples::blake::{Fu32, STATE_SIZE};

const INV16: BaseField = BaseField::from_u32_unchecked(1 << 15);
const TWO: BaseField = BaseField::from_u32_unchecked(2);

pub struct BlakeRoundEval<'a, E: EvalAtRow> {
    pub eval: E,
    pub xor_lookup_elements: &'a BlakeXorElements,
    pub round_lookup_elements: &'a RoundElements,
    pub logup: LogupAtRow<2, E>,
}
impl<'a, E: EvalAtRow> BlakeRoundEval<'a, E> {
    pub fn eval(mut self) -> E {
        let mut v: [Fu32<E::F>; STATE_SIZE] = std::array::from_fn(|_| self.next_u32());
        let input_v = v;
        let m: [Fu32<E::F>; STATE_SIZE] = std::array::from_fn(|_| self.next_u32());

        self.g(v.get_many_mut([0, 4, 8, 12]).unwrap(), m[0], m[1]);
        self.g(v.get_many_mut([1, 5, 9, 13]).unwrap(), m[2], m[3]);
        self.g(v.get_many_mut([2, 6, 10, 14]).unwrap(), m[4], m[5]);
        self.g(v.get_many_mut([3, 7, 11, 15]).unwrap(), m[6], m[7]);
        self.g(v.get_many_mut([0, 5, 10, 15]).unwrap(), m[8], m[9]);
        self.g(v.get_many_mut([1, 6, 11, 12]).unwrap(), m[10], m[11]);
        self.g(v.get_many_mut([2, 7, 8, 13]).unwrap(), m[12], m[13]);
        self.g(v.get_many_mut([3, 4, 9, 14]).unwrap(), m[14], m[15]);

        // Yield `Round(input_v, output_v, message)`.
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
        let l = self.eval.next_trace_mask();
        let h = self.eval.next_trace_mask();
        Fu32 { l, h }
    }
    fn g(&mut self, v: [&mut Fu32<E::F>; 4], m0: Fu32<E::F>, m1: Fu32<E::F>) {
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
        let sl = self.eval.next_trace_mask();
        let sh = self.eval.next_trace_mask();

        let carry_l = (a.l + b.l - sl) * E::F::from(INV16);
        self.eval.add_constraint(carry_l * carry_l - carry_l);

        let carry_h = (a.h + b.h + carry_l - sh) * E::F::from(INV16);
        self.eval.add_constraint(carry_h * carry_h - carry_h);

        Fu32 { l: sl, h: sh }
    }

    /// Adds three u32s, returning the sum.
    /// Assumes a, b, c are properly range checked.
    /// Caller is responsible for checking:
    /// res.{l,h} not in [2^16, 3*2^16) or in [-2^17,0)
    fn add3_u32_unchecked(&mut self, a: Fu32<E::F>, b: Fu32<E::F>, c: Fu32<E::F>) -> Fu32<E::F> {
        let sl = self.eval.next_trace_mask();
        let sh = self.eval.next_trace_mask();

        let carry_l = (a.l + b.l + c.l - sl) * E::F::from(INV16);
        self.eval
            .add_constraint(carry_l * (carry_l - E::F::one()) * (carry_l - E::F::from(TWO)));

        let carry_h = (a.h + b.h + c.h + carry_l - sh) * E::F::from(INV16);
        self.eval
            .add_constraint(carry_h * (carry_h - E::F::one()) * (carry_h - E::F::from(TWO)));

        Fu32 { l: sl, h: sh }
    }

    /// Splits a felt at r.
    /// Caller is responsible for checking that the ranges of h * 2^r and l don't overlap.
    fn split_unchecked(&mut self, a: E::F, r: u32) -> (E::F, E::F) {
        let h = self.eval.next_trace_mask();
        let l = a - h * E::F::from(BaseField::from_u32_unchecked(1 << r));
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
        let lookup_elements = self.xor_lookup_elements.get(w);
        let comb0 = lookup_elements.combine::<E::F, E::EF>(&[a[0], b[0], c[0]]);
        let comb1 = lookup_elements.combine::<E::F, E::EF>(&[a[1], b[1], c[1]]);
        let frac = Fraction {
            numerator: comb0 + comb1,
            denominator: comb0 * comb1,
        };

        self.logup.add_frac(&mut self.eval, frac);
        c
    }
}
