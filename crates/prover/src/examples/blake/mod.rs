use std::fmt::Debug;
use std::ops::{Add, AddAssign, Mul, Sub};
use std::simd::u32x16;

use xor_table::XorAccumulator;

use crate::constraint_framework::logup::LookupElements;
use crate::core::channel::Blake2sChannel;
use crate::core::fields::m31::BaseField;
use crate::core::fields::FieldExpOps;

mod round;
mod xor_table;

#[derive(Default)]
struct XorAccums {
    xor12: XorAccumulator<12, 4>,
    xor9: XorAccumulator<9, 2>,
    xor8: XorAccumulator<8, 2>,
    xor7: XorAccumulator<7, 2>,
    xor4: XorAccumulator<4, 0>,
}
impl XorAccums {
    fn add_input(&mut self, w: u32, a: u32x16, b: u32x16) {
        match w {
            12 => self.xor12.add_input(a, b),
            9 => self.xor9.add_input(a, b),
            8 => self.xor8.add_input(a, b),
            7 => self.xor7.add_input(a, b),
            4 => self.xor4.add_input(a, b),
            _ => panic!("Invalid w"),
        }
    }
}

#[derive(Clone)]
pub struct XorLookupElements {
    xor12: LookupElements,
    xor9: LookupElements,
    xor8: LookupElements,
    xor7: LookupElements,
    xor4: LookupElements,
}
impl XorLookupElements {
    fn draw(channel: &mut Blake2sChannel) -> Self {
        Self {
            xor12: LookupElements::draw(channel, 3),
            xor9: LookupElements::draw(channel, 3),
            xor8: LookupElements::draw(channel, 3),
            xor7: LookupElements::draw(channel, 3),
            xor4: LookupElements::draw(channel, 3),
        }
    }

    fn get(&self, w: u32) -> &LookupElements {
        match w {
            12 => &self.xor12,
            9 => &self.xor9,
            8 => &self.xor8,
            7 => &self.xor7,
            4 => &self.xor4,
            _ => panic!("Invalid w"),
        }
    }
}

#[derive(Clone, Copy, Debug)]
struct Fu32<F>
where
    F: FieldExpOps
        + Copy
        + Debug
        + AddAssign<F>
        + Add<F, Output = F>
        + Sub<F, Output = F>
        + Mul<BaseField, Output = F>,
{
    l: F,
    h: F,
}
impl<F> Fu32<F>
where
    F: FieldExpOps
        + Copy
        + Debug
        + AddAssign<F>
        + Add<F, Output = F>
        + Sub<F, Output = F>
        + Mul<BaseField, Output = F>,
{
    fn to_felts(self) -> [F; 2] {
        [self.l, self.h]
    }
}
