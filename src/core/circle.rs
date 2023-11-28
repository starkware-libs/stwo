use std::ops::{Add, Div, Mul, Neg, Sub};

use super::fields::m31::M31;
use super::fields::qm31::QM31;
use super::fields::Field;
use crate::math::egcd;

// TODO(AlonH): Consider also generalizing structs using this struct.
/// A point on the complex circle. Treaed as an additive group.
#[derive(Copy, Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct CirclePoint<F: Field> {
    pub x: F,
    pub y: F,
}

impl<F: Field> CirclePoint<F> {
    pub fn zero() -> Self {
        Self {
            x: F::one(),
            y: F::zero(),
        }
    }

    pub fn double(&self) -> Self {
        *self + *self
    }

    /// Applies the circle's x-coordinate doubling map i.e. `psi_x(x)`
    pub fn double_x(x: F) -> F {
        x.square().double() - F::one()
    }

    /// Returns the order of a point
    ///
    /// # Examples
    ///
    /// ```
    /// use prover_research::core::circle::{CirclePoint, M31_CIRCLE_GEN, M31_CIRCLE_ORDER_BITS};
    /// use prover_research::core::fields::m31::M31;
    /// assert_eq!(M31_CIRCLE_GEN.order_bits(), M31_CIRCLE_ORDER_BITS);
    /// ```
    pub fn order_bits(&self) -> usize {
        // we only need the x-coordinate to check order since the only point
        // with x=1 is the circle's identity
        let mut res = 0;
        let mut cur = self.x;
        while !cur.is_one() {
            cur = Self::double_x(cur);
            res += 1;
        }
        res
    }

    pub fn mul(&self, mut scalar: u128) -> CirclePoint<F> {
        let mut res = Self::zero();
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

    pub fn conjugate(&self) -> CirclePoint<F> {
        Self {
            x: self.x,
            y: -self.y,
        }
    }

    pub fn antipode(&self) -> CirclePoint<F> {
        Self {
            x: -self.x,
            y: -self.y,
        }
    }
}

impl<F: Field> Add for CirclePoint<F> {
    type Output = Self;

    fn add(self, rhs: Self) -> Self::Output {
        let x = self.x * rhs.x - self.y * rhs.y;
        let y = self.x * rhs.y + self.y * rhs.x;
        Self { x, y }
    }
}

impl<F: Field> Neg for CirclePoint<F> {
    type Output = Self;

    fn neg(self) -> Self::Output {
        self.conjugate()
    }
}

impl<F: Field> Sub for CirclePoint<F> {
    type Output = Self;

    fn sub(self, rhs: Self) -> Self::Output {
        self + (-rhs)
    }
}

pub const M31_CIRCLE_GEN: CirclePoint<M31> = CirclePoint {
    x: M31::from_u32_unchecked(2),
    y: M31::from_u32_unchecked(1268011823),
};

pub const M31_CIRCLE_ORDER_BITS: usize = 31;

pub const QM31_CIRCLE_GEN: CirclePoint<QM31> = CirclePoint {
    x: QM31::from_u32_unchecked(1, 0, 478637715, 513582961),
    y: QM31::from_u32_unchecked(568722919, 616616927, 0, 74382916),
};

/// Integer i that represent the circle point i * CIRCLE_GEN. Treated as an
/// additive ring modulo 1 << CURVE_ORDER_BITS.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Ord, PartialOrd)]
pub struct CirclePointIndex(pub usize);

impl CirclePointIndex {
    pub fn zero() -> Self {
        Self(0)
    }

    pub fn generator() -> Self {
        Self(1)
    }

    pub fn reduce(self) -> Self {
        Self(self.0 & ((1 << M31_CIRCLE_ORDER_BITS) - 1))
    }

    pub fn subgroup_gen(n_bits: usize) -> Self {
        assert!(n_bits <= M31_CIRCLE_ORDER_BITS);
        Self(1 << (M31_CIRCLE_ORDER_BITS - n_bits))
    }

    pub fn to_point(self) -> CirclePoint<M31> {
        M31_CIRCLE_GEN.mul(self.0 as u128)
    }

    pub fn half(self) -> Self {
        assert!(self.0 & 1 == 0);
        Self(self.0 >> 1)
    }

    pub fn try_div(&self, rhs: CirclePointIndex) -> Option<usize> {
        // Find x s.t. x * rhs.0 = self.0 (mod CIRCLE_ORDER).
        let (s, _t, g) = egcd(rhs.0 as isize, 1 << M31_CIRCLE_ORDER_BITS);
        if self.0 as isize % g != 0 {
            return None;
        }
        let res = s * self.0 as isize / g;
        let cap = (1 << M31_CIRCLE_ORDER_BITS) / g;
        let res = ((res % cap) + cap) % cap;
        Some(res as usize)
    }
}

impl Add for CirclePointIndex {
    type Output = Self;

    fn add(self, rhs: Self) -> Self::Output {
        Self(self.0 + rhs.0).reduce()
    }
}

impl Sub for CirclePointIndex {
    type Output = Self;

    fn sub(self, rhs: Self) -> Self::Output {
        Self(self.0 + (1 << M31_CIRCLE_ORDER_BITS) - rhs.0).reduce()
    }
}

impl Mul<usize> for CirclePointIndex {
    type Output = Self;

    fn mul(self, rhs: usize) -> Self::Output {
        Self(self.0 * rhs).reduce()
    }
}

impl Div for CirclePointIndex {
    type Output = usize;

    fn div(self, rhs: Self) -> Self::Output {
        self.try_div(rhs).unwrap()
    }
}

impl Neg for CirclePointIndex {
    type Output = Self;

    fn neg(self) -> Self::Output {
        Self((1 << M31_CIRCLE_ORDER_BITS) - self.0).reduce()
    }
}

/// Represents the coset initial + \<step\>.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct Coset {
    pub initial_index: CirclePointIndex,
    pub initial: CirclePoint<M31>,
    pub step_size: CirclePointIndex,
    pub step: CirclePoint<M31>,
    pub n_bits: usize,
}

impl Coset {
    pub fn new(initial_index: CirclePointIndex, n_bits: usize) -> Self {
        assert!(n_bits <= M31_CIRCLE_ORDER_BITS);
        let step_size = CirclePointIndex::subgroup_gen(n_bits);
        Self {
            initial_index,
            initial: initial_index.to_point(),
            step: step_size.to_point(),
            step_size,
            n_bits,
        }
    }

    /// Creates a coset of the form <G_n>.
    /// For example, for n=8, we get the point indices \[0,1,2,3,4,5,6,7\].
    pub fn subgroup(n_bits: usize) -> Self {
        Self::new(CirclePointIndex::zero(), n_bits)
    }

    /// Creates a coset of the form G_2n + \<G_n\>.
    /// For example, for n=8, we get the point indices \[1,3,5,7,9,11,13,15\].
    pub fn odds(n_bits: usize) -> Self {
        Self::new(CirclePointIndex::subgroup_gen(n_bits + 1), n_bits)
    }

    /// Creates a coset of the form G_4n + <G_n>.
    /// For example, for n=8, we get the point indices \[1,5,9,13,17,21,25,29\].
    /// Its conjugate will be \[3,7,11,15,19,23,27,31\].
    pub fn half_odds(n_bits: usize) -> Self {
        Self::new(CirclePointIndex::subgroup_gen(n_bits + 2), n_bits)
    }

    pub fn len(&self) -> usize {
        1 << self.n_bits
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    pub fn iter(&self) -> CosetIterator<CirclePoint<M31>> {
        CosetIterator {
            cur: self.initial,
            step: self.step,
            remaining: self.len(),
        }
    }

    pub fn iter_indices(&self) -> CosetIterator<CirclePointIndex> {
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

    pub fn initial(&self) -> CirclePoint<M31> {
        M31_CIRCLE_GEN.repeated_double(M31_CIRCLE_ORDER_BITS - self.n_bits - 1)
    }

    pub fn index_at(&self, index: usize) -> CirclePointIndex {
        self.initial_index + self.step_size.mul(index)
    }

    pub fn at(&self, index: usize) -> CirclePoint<M31> {
        self.index_at(index).to_point()
    }

    pub fn shift(&self, shift_size: CirclePointIndex) -> Self {
        let initial_index = self.initial_index + shift_size;
        Self {
            initial_index,
            initial: initial_index.to_point(),
            ..*self
        }
    }

    /// Creates the conjugate coset: -initial -\<step\>.
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

    pub fn find(&self, i: CirclePointIndex) -> Option<usize> {
        (i - self.initial_index).try_div(self.step_size)
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

#[cfg(test)]
mod tests {
    use std::collections::BTreeSet;

    use super::{CirclePointIndex, Coset};
    use crate::core::poly::circle::{CanonicCoset, CircleDomain};

    #[test]
    fn test_domains() {
        let eval_n_bits = 4;
        let canonic_cosets_extensions = [
            CanonicCoset::new(2).evaluation_domain(eval_n_bits + 1),
            CanonicCoset::new(2).evaluation_domain(eval_n_bits + 2),
            CanonicCoset::new(eval_n_bits - 1).evaluation_domain(eval_n_bits),
        ];

        let subgroup_gen = CirclePointIndex::subgroup_gen(eval_n_bits);
        let constraint_evaluation_domain =
            CircleDomain::constraint_evaluation_domain(eval_n_bits - 1);

        for point_index in constraint_evaluation_domain.iter_indices() {
            for eval in &canonic_cosets_extensions {
                assert!(eval.find(point_index - subgroup_gen).is_some());
            }
        }
    }

    #[test]
    fn test_iterator() {
        let coset = Coset::new(CirclePointIndex(1), 3);
        let actual_indices: Vec<_> = coset.iter_indices().collect();
        let expected_indices = vec![
            CirclePointIndex(1),
            CirclePointIndex(1) + CirclePointIndex::subgroup_gen(3) * 1,
            CirclePointIndex(1) + CirclePointIndex::subgroup_gen(3) * 2,
            CirclePointIndex(1) + CirclePointIndex::subgroup_gen(3) * 3,
            CirclePointIndex(1) + CirclePointIndex::subgroup_gen(3) * 4,
            CirclePointIndex(1) + CirclePointIndex::subgroup_gen(3) * 5,
            CirclePointIndex(1) + CirclePointIndex::subgroup_gen(3) * 6,
            CirclePointIndex(1) + CirclePointIndex::subgroup_gen(3) * 7,
        ];
        assert_eq!(actual_indices, expected_indices);

        let actual_points = coset.iter().collect::<Vec<_>>();
        let expected_points: Vec<_> = expected_indices.iter().map(|i| i.to_point()).collect();
        assert_eq!(actual_points, expected_points);
    }

    #[test]
    fn test_coset_is_half_coset_with_conjugate() {
        let canonic_coset = CanonicCoset::new(8);
        let coset_points = BTreeSet::from_iter(canonic_coset.coset().iter());

        let half_coset_points = BTreeSet::from_iter(canonic_coset.half_coset().iter());
        let half_coset_conjugate_points =
            BTreeSet::from_iter(canonic_coset.half_coset().conjugate().iter());

        assert!((&half_coset_points & &half_coset_conjugate_points).is_empty());
        assert_eq!(
            coset_points,
            &half_coset_points | &half_coset_conjugate_points
        )
    }
}
