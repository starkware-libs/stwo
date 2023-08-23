use super::field::M31;
use std::ops::{Add, Neg, Sub};

/// A point on the complex circle.
#[derive(Copy, Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct CirclePoint {
    pub x: M31,
    pub y: M31,
}
impl CirclePoint {
    pub fn unit() -> Self {
        Self {
            x: M31::one(),
            y: M31::zero(),
        }
    }
    pub fn double(&self) -> Self {
        *self + *self
    }
    pub fn order_bits(&self) -> usize {
        let mut res = 0;
        let mut cur = *self;
        while cur != Self::unit() {
            cur = cur.double();
            res += 1;
        }
        res
    }
    pub fn mul(&self, mut scalar: u64) -> CirclePoint {
        let mut res = Self::unit();
        let mut cur = *self;
        while scalar > 0 {
            if scalar & 1 == 1 {
                res = res + cur;
            }
            cur = cur.double();
            scalar >>= 1;
        }
        res
    }
    pub fn repeated_double(&self, n: usize) -> Self {
        let mut res = *self;
        for _ in 0..n {
            res = res.double();
        }
        res
    }
}
impl Add for CirclePoint {
    type Output = Self;

    fn add(self, rhs: Self) -> Self::Output {
        let x = self.x * rhs.x - self.y * rhs.y;
        let y = self.x * rhs.y + self.y * rhs.x;
        Self { x, y }
    }
}
impl Neg for CirclePoint {
    type Output = Self;

    fn neg(self) -> Self::Output {
        Self {
            x: self.x,
            y: -self.y,
        }
    }
}
impl Sub for CirclePoint {
    type Output = Self;

    fn sub(self, rhs: Self) -> Self::Output {
        self + (-rhs)
    }
}

pub const CIRCLE_GEN: CirclePoint = CirclePoint {
    x: M31::from_u32_unchecked(2),
    y: M31::from_u32_unchecked(1268011823),
};
pub const CIRCLE_ORDER_BITS: usize = 31;

pub struct Coset {
    pub initial: CirclePoint,
    pub n_bits: usize,
}

#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub struct CanonicCoset {
    pub n_bits: usize,
}
impl CanonicCoset {
    pub fn new(n_bits: usize) -> Self {
        assert!(n_bits < CIRCLE_ORDER_BITS);
        Self { n_bits }
    }
    pub fn size(&self) -> usize {
        1 << self.n_bits
    }
    pub fn iter(&self) -> CanonicCosetIterator {
        let gen = CIRCLE_GEN.repeated_double(CIRCLE_ORDER_BITS - self.n_bits - 1);
        CanonicCosetIterator {
            cur: gen,
            step: gen.double(),
            remaining: self.size(),
        }
    }

    pub fn double(&self) -> Self {
        Self {
            n_bits: self.n_bits - 1,
        }
    }

    pub fn initial(&self) -> CirclePoint {
        CIRCLE_GEN.repeated_double(CIRCLE_ORDER_BITS - self.n_bits - 1)
    }
    pub fn step(&self) -> CirclePoint {
        CIRCLE_GEN.repeated_double(CIRCLE_ORDER_BITS - self.n_bits)
    }

    pub fn at(&self, index: usize) -> CirclePoint {
        self.initial() + self.step().mul(index as u64)
    }
}

pub struct CanonicCosetIterator {
    cur: CirclePoint,
    step: CirclePoint,
    remaining: usize,
}
impl Iterator for CanonicCosetIterator {
    type Item = CirclePoint;

    fn next(&mut self) -> Option<Self::Item> {
        if self.remaining == 0 {
            return None;
        }
        self.remaining -= 1;
        let res = self.cur;
        self.cur = self.cur + self.step;
        Some(res)
    }
}
