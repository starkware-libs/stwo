use std::simd::{simd_swizzle, u32x2, Simd};

use super::m31::{PackedM31, LOG_N_LANES};
use crate::core::circle::{CirclePoint, M31_CIRCLE_LOG_ORDER};
use crate::core::fields::m31::M31;
use crate::core::poly::circle::CircleDomain;
use crate::core::utils::bit_reverse_index;

pub struct CircleDomainBitRevIterator {
    domain: CircleDomain,
    i: usize,
    current: CirclePoint<PackedM31>,
    flips: [CirclePoint<M31>; (M31_CIRCLE_LOG_ORDER - LOG_N_LANES) as usize],
}
impl CircleDomainBitRevIterator {
    pub fn new(domain: CircleDomain) -> Self {
        let log_size = domain.log_size();
        assert!(log_size >= LOG_N_LANES);

        let initial_points = std::array::from_fn(|i| domain.at(bit_reverse_index(i, log_size)));
        let current = CirclePoint {
            x: PackedM31::from_array(initial_points.each_ref().map(|p| p.x)),
            y: PackedM31::from_array(initial_points.each_ref().map(|p| p.y)),
        };

        let mut flips = [CirclePoint::zero(); (M31_CIRCLE_LOG_ORDER - LOG_N_LANES) as usize];
        for i in 0..(log_size - LOG_N_LANES) {
            //  L   i
            // 000111000000 ->
            // 000000100000
            let prev_mul = bit_reverse_index((1 << i) - 1, log_size - LOG_N_LANES);
            let new_mul = bit_reverse_index(1 << i, log_size - LOG_N_LANES);
            let flip = domain.half_coset.step.mul(new_mul as u128)
                - domain.half_coset.step.mul(prev_mul as u128);
            flips[i as usize] = flip;
        }
        Self {
            domain,
            i: 0,
            current,
            flips,
        }
    }
}
impl Iterator for CircleDomainBitRevIterator {
    type Item = CirclePoint<PackedM31>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.i << LOG_N_LANES >= self.domain.size() {
            return None;
        }
        let res = self.current;
        let flip = self.flips[self.i.trailing_ones() as usize];
        let flipx = Simd::splat(flip.x.0);
        let flipy = u32x2::from_array([flip.y.0, (-flip.y).0]);
        let flipy = simd_swizzle!(flipy, [0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1]);
        let flip = unsafe {
            CirclePoint {
                x: PackedM31::from_simd_unchecked(flipx),
                y: PackedM31::from_simd_unchecked(flipy),
            }
        };
        self.current = self.current + flip;
        self.i += 1;
        Some(res)
    }
}

#[test]
fn test_circle_domain_bit_rev_iterator() {
    let domain = CircleDomain::new(crate::core::circle::Coset::new(
        crate::core::circle::CirclePointIndex::generator(),
        5,
    ));
    let mut expected = domain.iter().collect::<Vec<_>>();
    crate::core::backend::cpu::bit_reverse(&mut expected);
    let actual = CircleDomainBitRevIterator::new(domain)
        .flat_map(|c| -> [_; 16] {
            std::array::from_fn(|i| CirclePoint {
                x: c.x.to_array()[i],
                y: c.y.to_array()[i],
            })
        })
        .collect::<Vec<_>>();
    assert_eq!(actual, expected);
}
