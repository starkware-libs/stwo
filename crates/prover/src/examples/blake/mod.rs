//! AIR for blake2s and blake3.
//! See <https://en.wikipedia.org/wiki/BLAKE_(hash_function)>

use std::fmt::Debug;
use std::ops::{Add, AddAssign, Mul, Sub};
use std::simd::u32x16;

use xor_table::{XorAccumulator, XorElements};

use crate::core::backend::simd::m31::PackedBaseField;
use crate::core::channel::Channel;
use crate::core::fields::m31::BaseField;
use crate::core::fields::FieldExpOps;

mod air;
mod round;
mod scheduler;
mod xor_table;

const STATE_SIZE: usize = 16;
const MESSAGE_SIZE: usize = 16;
const N_FELTS_IN_U32: usize = 2;
const N_ROUND_INPUT_FELTS: usize = (STATE_SIZE + STATE_SIZE + MESSAGE_SIZE) * N_FELTS_IN_U32;

// Parameters for Blake2s. Change these for blake3.
const N_ROUNDS: usize = 10;
/// A splitting N_ROUNDS into several powers of 2.
const ROUND_LOG_SPLIT: [u32; 2] = [3, 1];

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
pub struct BlakeXorElements {
    xor12: XorElements,
    xor9: XorElements,
    xor8: XorElements,
    xor7: XorElements,
    xor4: XorElements,
}
impl BlakeXorElements {
    fn draw(channel: &mut impl Channel) -> Self {
        Self {
            xor12: XorElements::draw(channel),
            xor9: XorElements::draw(channel),
            xor8: XorElements::draw(channel),
            xor7: XorElements::draw(channel),
            xor4: XorElements::draw(channel),
        }
    }
    fn dummy() -> Self {
        Self {
            xor12: XorElements::dummy(),
            xor9: XorElements::dummy(),
            xor8: XorElements::dummy(),
            xor7: XorElements::dummy(),
            xor4: XorElements::dummy(),
        }
    }
    fn get(&self, w: u32) -> &XorElements {
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

/// Utility for representing a u32 as two field elements, for constraint evaluation.
#[derive(Clone, Debug)]
struct Fu32<F>
where
    F: FieldExpOps
        + Clone
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
        + Clone
        + Debug
        + AddAssign<F>
        + Add<F, Output = F>
        + Sub<F, Output = F>
        + Mul<BaseField, Output = F>,
{
    fn into_felts(self) -> [F; 2] {
        [self.l, self.h]
    }
}

/// Utility for splitting a u32 into 2 field elements in trace generation.
fn to_felts(x: &u32x16) -> [PackedBaseField; 2] {
    [
        unsafe { PackedBaseField::from_simd_unchecked(x & u32x16::splat(0xffff)) },
        unsafe { PackedBaseField::from_simd_unchecked(x >> 16) },
    ]
}
