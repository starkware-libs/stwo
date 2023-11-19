use num_traits::One;

use super::circle::{CirclePoint, CirclePointIndex, Coset};
use super::fft::psi_x;
use super::fields::m31::BaseField;
use super::poly::circle::{CircleDomain, CirclePoly, Evaluation};

// Evaluates a vanishing polynomial of the coset at a point.
pub fn coset_vanishing(coset: Coset, mut p: CirclePoint) -> BaseField {
    // Doubling a point `n_bits / 2` times and taking the x coordinate is
    // essentially evaluating a polynomial in x of degree `2**(n_bits-1)`. If
    // the entire `2**n_bits` points of the coset are roots (i.e. yield 0), then
    // this is a vanishing polynomial of these points.

    // Rotating the coset -coset.initial + step / 2 yields a canonic coset:
    // `step/2 + <step>.`
    // Doubling this coset n_bits - 1 times yields the coset +-G_4.
    // th polynomial x vanishes on these points.
    //   X
    // . . X
    p = p - coset.initial + coset.step_size.half().to_point();
    let mut x = p.x;

    // The formula for the x coordinate of the double of a point.
    for _ in 0..(coset.n_bits - 1) {
        x = psi_x(x);
    }
    x
}

pub fn circle_domain_vanishing(domain: CircleDomain, p: CirclePoint) -> BaseField {
    coset_vanishing(domain.half_coset, p) * coset_vanishing(domain.half_coset.conjugate(), p)
}

// Evaluates the polynmial that is used to exclude the excluded point at point
// p. Note that this polynomial has a zero of multiplicity 2 at the excluded
// point.
pub fn point_excluder(excluded: CirclePoint, p: CirclePoint) -> BaseField {
    (p - excluded).x - BaseField::one()
}

// Evaluates a vanishing polynomial of the vanish_point at a point.
// Note that this function has a pole on the antipode of the vanish_point.
pub fn point_vanishing(vanish_point: CirclePoint, p: CirclePoint) -> BaseField {
    let h = p - vanish_point;
    h.y / (BaseField::one() + h.x)
}

// Utils for computing constraints.
// Oracle to a polynomial constrained to a coset.
pub trait PolyOracle: Copy {
    fn get_at(&self, index: CirclePointIndex) -> BaseField;
    fn point(&self) -> CirclePoint;
}

#[derive(Copy, Clone)]
pub struct EvalByPoly<'a> {
    pub point: CirclePoint,
    pub poly: &'a CirclePoly,
}

impl<'a> PolyOracle for EvalByPoly<'a> {
    fn point(&self) -> CirclePoint {
        self.point
    }

    fn get_at(&self, index: CirclePointIndex) -> BaseField {
        self.poly.eval_at_point(self.point + index.to_point())
    }
}

// TODO(spapini): make an iterator instead, so we do all computations beforehand.
#[derive(Clone)]
pub struct EvalByEvaluation<'a, T: Evaluation> {
    pub offset: CirclePointIndex,
    pub eval: &'a T,
}

impl<'a, T: Evaluation> PolyOracle for EvalByEvaluation<'a, T> {
    fn point(&self) -> CirclePoint {
        self.offset.to_point()
    }

    fn get_at(&self, index: CirclePointIndex) -> BaseField {
        self.eval.get_at(index + self.offset)
    }
}

impl<'a, T: Evaluation> Copy for EvalByEvaluation<'a, T> {}

#[test]
fn test_coset_vanishing() {
    use num_traits::Zero;
    let cosets = [
        Coset::half_odds(5),
        Coset::odds(5),
        Coset::new(CirclePointIndex::zero(), 5),
        Coset::half_odds(5).conjugate(),
    ];
    for c0 in cosets.iter() {
        for el in c0.iter() {
            assert_eq!(coset_vanishing(*c0, el), BaseField::zero());
            for c1 in cosets.iter() {
                if c0 == c1 {
                    continue;
                }
                assert_ne!(coset_vanishing(*c1, el), BaseField::zero());
            }
        }
    }
}

#[test]
fn test_point_excluder() {
    use crate::core::fields::Field;
    let excluded = Coset::half_odds(5).at(10);
    let point = (CirclePointIndex::generator() * 4).to_point();

    let num = point_excluder(excluded, point) * point_excluder(excluded.conjugate(), point);
    let denom = (point.x - excluded.x).pow(2);

    assert_eq!(num, denom);
}

#[test]
fn test_point_vanishing_success() {
    use num_traits::Zero;
    let coset = Coset::odds(5);
    let vanish_point = coset.at(2);
    for el in coset.iter() {
        if el == vanish_point {
            assert_eq!(point_vanishing(vanish_point, el), BaseField::zero());
            continue;
        }
        if el == vanish_point.antipode() {
            continue;
        }
        assert_ne!(point_vanishing(vanish_point, el), BaseField::zero());
    }
}

#[test]
#[should_panic(expected = "0 has no inverse")]
fn test_point_vanishing_failure() {
    let coset = Coset::half_odds(6);
    let point = coset.at(4);
    point_vanishing(point, point.antipode());
}
