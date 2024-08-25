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

pub mod air;
pub mod round;
pub mod scheduler;
pub mod xor_table;

pub const STATE_SIZE: usize = 16;
pub const MESSAGE_SIZE: usize = 16;
pub const N_FELTS_IN_U32: usize = 2;
pub const N_ROUND_INPUT_FELTS: usize = (STATE_SIZE + STATE_SIZE + MESSAGE_SIZE) * N_FELTS_IN_U32;

// Parameters for Blake2s. Change these for blake3.
pub const N_ROUNDS: usize = 10;
/// A splitting N_ROUNDS into several powers of 2.
pub const ROUND_LOG_SPLIT: [u32; 2] = [3, 1];

#[derive(Default)]
pub struct XorAccums {
    pub xor12: XorAccumulator<12, 4>,
    pub xor9: XorAccumulator<9, 2>,
    pub xor8: XorAccumulator<8, 2>,
    pub xor7: XorAccumulator<7, 2>,
    pub xor4: XorAccumulator<4, 0>,
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
    pub xor12: XorElements,
    pub xor9: XorElements,
    pub xor8: XorElements,
    pub xor7: XorElements,
    pub xor4: XorElements,
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
    pub fn get(&self, w: u32) -> &XorElements {
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
#[derive(Clone, Copy, Debug)]
pub struct Fu32<F>
where
    F: FieldExpOps
        + Copy
        + Debug
        + AddAssign<F>
        + Add<F, Output = F>
        + Sub<F, Output = F>
        + Mul<BaseField, Output = F>,
{
    pub l: F,
    pub h: F,
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
    pub fn to_felts(self) -> [F; 2] {
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
