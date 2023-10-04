use super::fields::m31::M31;
use std::ops::{Add, Div, Mul, Neg, Sub};

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

    pub fn conjugate(&self) -> CirclePoint {
        Self {
            x: self.x,
            y: -self.y,
        }
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

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct CircleIndex(pub usize);
impl CircleIndex {
    pub fn zero() -> Self {
        Self(0)
    }
    pub fn generator() -> Self {
        Self(1)
    }
    pub fn root(n_bits: usize) -> Self {
        assert!(n_bits <= CIRCLE_ORDER_BITS);
        Self(1 << (CIRCLE_ORDER_BITS - n_bits))
    }
    pub fn to_point(self) -> CirclePoint {
        CIRCLE_GEN.mul(self.0 as u64)
    }
    pub fn half(self) -> Self {
        assert!(self.0 & 1 == 0);
        Self(self.0 >> 1)
    }
    pub fn try_div(&self, rhs: CircleIndex) -> Option<usize> {
        // Find x s.t. x * rhs.0 = self.0 (mod CIRCLE_ORDER).
        let (s, _t, g) = egcd(rhs.0 as i64, 1 << CIRCLE_ORDER_BITS);
        if (self.0 as i64) % g != 0 {
            return None;
        }
        let res = s * (self.0 as i64) / g;
        let cap = (1 << CIRCLE_ORDER_BITS) / g;
        let res = ((res % cap) + cap) % cap;
        Some(res as usize)
    }
}
impl Add for CircleIndex {
    type Output = Self;

    fn add(self, rhs: Self) -> Self::Output {
        Self((self.0 + rhs.0) & ((1 << CIRCLE_ORDER_BITS) - 1))
    }
}
impl Sub for CircleIndex {
    type Output = Self;

    fn sub(self, rhs: Self) -> Self::Output {
        Self((self.0 + (1 << CIRCLE_ORDER_BITS) - rhs.0) & ((1 << CIRCLE_ORDER_BITS) - 1))
    }
}
impl Mul<usize> for CircleIndex {
    type Output = Self;

    fn mul(self, rhs: usize) -> Self::Output {
        Self((self.0 * rhs) & ((1 << CIRCLE_ORDER_BITS) - 1))
    }
}
fn egcd(x: i64, y: i64) -> (i64, i64, i64) {
    if x == 0 {
        return (0, 1, y);
    }
    let k = y / x;
    let (s, t, g) = egcd(y % x, x);
    (t - s * k, s, g)
}
impl Div for CircleIndex {
    type Output = usize;

    fn div(self, rhs: Self) -> Self::Output {
        self.try_div(rhs).unwrap()
    }
}
impl Neg for CircleIndex {
    type Output = Self;

    fn neg(self) -> Self::Output {
        Self((1 << CIRCLE_ORDER_BITS) - self.0)
    }
}

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct Coset {
    pub initial_index: CircleIndex,
    pub initial: CirclePoint,
    pub step_size: CircleIndex,
    pub step: CirclePoint,
    pub n_bits: usize,
}
impl Coset {
    pub fn new(initial_index: CircleIndex, n_bits: usize) -> Self {
        assert!(n_bits <= CIRCLE_ORDER_BITS);
        let step_size = CircleIndex::root(n_bits);
        Self {
            initial_index,
            initial: initial_index.to_point(),
            step: step_size.to_point(),
            step_size,
            n_bits,
        }
    }
    // 4j+1.
    pub fn twisted(n_bits: usize) -> Self {
        Self::new(CircleIndex::root(n_bits + 2), n_bits)
    }
    // 2j+1.
    pub fn odds(n_bits: usize) -> Self {
        Self::new(CircleIndex::root(n_bits + 1), n_bits)
    }
    pub fn len(&self) -> usize {
        1 << self.n_bits
    }
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
    pub fn iter(&self) -> CosetIterator<CirclePoint> {
        CosetIterator {
            cur: self.initial,
            step: self.step,
            remaining: self.len(),
        }
    }
    pub fn iter_indices(&self) -> CosetIterator<CircleIndex> {
        CosetIterator {
            cur: self.initial_index,
            step: self.step_size,
            remaining: self.len(),
        }
    }
    pub fn double(&self) -> Self {
        assert!(self.n_bits > 0);
        Self {
            initial_index: self.initial_index * 2,
            initial: self.initial.double(),
            step: self.step.double(),
            step_size: self.step_size * 2,
            n_bits: self.n_bits - 1,
        }
    }
    pub fn initial(&self) -> CirclePoint {
        CIRCLE_GEN.repeated_double(CIRCLE_ORDER_BITS - self.n_bits - 1)
    }
    pub fn index_at(&self, index: usize) -> CircleIndex {
        self.initial_index + self.step_size.mul(index)
    }
    pub fn at(&self, index: usize) -> CirclePoint {
        self.index_at(index).to_point()
    }
    pub fn shift(&self, shift_size: CircleIndex) -> Self {
        let initial_index = self.initial_index + shift_size;
        Self {
            initial_index,
            initial: initial_index.to_point(),
            ..*self
        }
    }
    pub fn conjugate(&self) -> Self {
        let initial_index = -self.initial_index;
        let step_size = -self.step_size;
        Self {
            initial_index,
            initial: initial_index.to_point(),
            step_size,
            step: step_size.to_point(),
            n_bits: self.n_bits,
        }
    }
}

#[derive(Clone)]
pub struct CosetIterator<T: Add> {
    pub cur: T,
    pub step: T,
    pub remaining: usize,
}
impl<T: Add<Output = T> + Copy> Iterator for CosetIterator<T> {
    type Item = T;

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
