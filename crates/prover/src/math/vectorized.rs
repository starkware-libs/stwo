use std::ops::{Add, AddAssign, Mul, MulAssign, Neg, Sub};

use bytemuck::{Pod, Zeroable};
use num_traits::{One, Zero};

use crate::core::fields::FieldExpOps;

#[derive(Copy, Clone, Debug)]
#[repr(transparent)]
pub struct Vectorized<A, const N: usize>(pub [A; N]);

impl<A, const N: usize> Vectorized<A, N> {
    pub fn from_fn<F>(cb: F) -> Self
    where
        F: FnMut(usize) -> A,
    {
        Vectorized(std::array::from_fn(cb))
    }
}

unsafe impl<A, const N: usize> Zeroable for Vectorized<A, N> {
    fn zeroed() -> Self {
        unsafe { core::mem::zeroed() }
    }
}
unsafe impl<A: Pod, const N: usize> Pod for Vectorized<A, N> {}

pub trait Scalar {}

impl<A: Add<B> + Copy, B: Copy, const N: usize> Add<Vectorized<B, N>> for Vectorized<A, N> {
    type Output = Vectorized<A::Output, N>;

    fn add(self, other: Vectorized<B, N>) -> Self::Output {
        Vectorized::from_fn(|i| self.0[i] + other.0[i])
    }
}

impl<A: Add<B> + Copy, B: Scalar + Copy, const N: usize> Add<B> for Vectorized<A, N> {
    type Output = Vectorized<A::Output, N>;

    fn add(self, other: B) -> Self::Output {
        Vectorized::from_fn(|i| self.0[i] + other)
    }
}

impl<A: Sub<B> + Copy, B: Copy, const N: usize> Sub<Vectorized<B, N>> for Vectorized<A, N> {
    type Output = Vectorized<A::Output, N>;

    fn sub(self, other: Vectorized<B, N>) -> Self::Output {
        Vectorized::from_fn(|i| self.0[i] - other.0[i])
    }
}

impl<A: Sub<B> + Copy, B: Scalar + Copy, const N: usize> Sub<B> for Vectorized<A, N> {
    type Output = Vectorized<A::Output, N>;

    fn sub(self, other: B) -> Self::Output {
        Vectorized::from_fn(|i| self.0[i] - other)
    }
}

impl<A: Mul<B> + Copy, B: Copy, const N: usize> Mul<Vectorized<B, N>> for Vectorized<A, N> {
    type Output = Vectorized<A::Output, N>;

    fn mul(self, other: Vectorized<B, N>) -> Self::Output {
        Vectorized::from_fn(|i| self.0[i] * other.0[i])
    }
}

impl<A: Mul<B> + Copy, B: Scalar + Copy, const N: usize> Mul<B> for Vectorized<A, N> {
    type Output = Vectorized<A::Output, N>;

    fn mul(self, other: B) -> Self::Output {
        Vectorized::from_fn(|i| self.0[i] * other)
    }
}

impl<A: AddAssign<B> + Copy, B: Copy, const N: usize> AddAssign<Vectorized<B, N>>
    for Vectorized<A, N>
{
    fn add_assign(&mut self, other: Vectorized<B, N>) {
        for i in 0..N {
            self.0[i] += other.0[i];
        }
    }
}

impl<A: AddAssign<B> + Copy, B: Scalar + Copy, const N: usize> AddAssign<B> for Vectorized<A, N> {
    fn add_assign(&mut self, other: B) {
        for i in 0..N {
            self.0[i] += other;
        }
    }
}

impl<A: MulAssign<B> + Copy, B: Copy, const N: usize> MulAssign<Vectorized<B, N>>
    for Vectorized<A, N>
{
    fn mul_assign(&mut self, other: Vectorized<B, N>) {
        for i in 0..N {
            self.0[i] *= other.0[i];
        }
    }
}

impl<A: Neg + Copy, const N: usize> Neg for Vectorized<A, N> {
    type Output = Vectorized<A::Output, N>;

    #[inline(always)]
    fn neg(self) -> Self::Output {
        Vectorized::from_fn(|i| self.0[i].neg())
    }
}

impl<A: Zero + Copy, const N: usize> Zero for Vectorized<A, N> {
    fn zero() -> Self {
        Vectorized::from_fn(|_| A::zero())
    }

    fn is_zero(&self) -> bool {
        self.0.iter().all(A::is_zero)
    }
}

impl<A: One + Copy, const N: usize> One for Vectorized<A, N> {
    fn one() -> Self {
        Vectorized::from_fn(|_| A::one())
    }
}

impl<A: FieldExpOps + Zero, const N: usize> FieldExpOps for Vectorized<A, N> {
    fn inverse(&self) -> Self {
        Vectorized::from_fn(|i| self.0[i].inverse())
    }
}
