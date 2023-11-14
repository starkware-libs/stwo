use super::{
    circle::{CirclePoint, CirclePointIndex, Coset},
    fft::psi_x,
    fields::m31::Field,
    poly::circle::{CircleDomain, CircleEvaluation, CirclePoly},
};
use num_traits::One;

// Evaluates a vanishing polynomial of the coset at a point.
pub fn coset_vanishing(coset: Coset, mut p: CirclePoint) -> Field {
    // Doubling a point `n_bits / 2` times and taking the x coordinate is essentially evaluating
    // a polynomial in x of degree `2**(n_bits-1)`. If the entire `2**n_bits` points of the coset
    // are roots (i.e. yield 0), then this is a vanishing polynomial of these points.

    // Rotating the coset -coset.initial + step / 2 yields a canonic coset:
    // `step/2 + <step>.`
    // Doubling this coset n_bits - 1 times yields the coset +-G_4.
    // th polynomial x vanishes on these points.
    //   X
    // .   .
    //   X
    p = p - coset.initial + coset.step_size.half().to_point();
    let mut x = p.x;

    // The formula for the x coordinate of the double of a point.
    for _ in 0..(coset.n_bits - 1) {
        x = psi_x(x);
    }
    x
}

pub fn circle_domain_vanishing(domain: CircleDomain, p: CirclePoint) -> Field {
    coset_vanishing(domain.half_coset, p) * coset_vanishing(domain.half_coset.conjugate(), p)
}

// Evaluates the polynmial that is used to exclude the excluded point at point p.
// Note that this polynomial has a zero of multiplicity 2 at the excluded point.
pub fn point_excluder(excluded: CirclePoint, p: CirclePoint) -> Field {
    (p - excluded).x - Field::one()
}

// Evaluates a vanishing polynomial of the vanish_point at a point.
// Note that this function has a pole on the antipode of the vanish_point.
pub fn point_vanishing(vanish_point: CirclePoint, p: CirclePoint) -> Field {
    let h = p - vanish_point;
    h.y / (Field::one() + h.x)
}

// Utils for computing constraints.
// Oracle to a polynomial constrained to a coset.
pub trait PolyOracle {
    fn get_at(&self, i: CirclePointIndex, point_index: CirclePointIndex) -> Field;
}

impl PolyOracle for CirclePoly {
    fn get_at(&self, i: CirclePointIndex, point_index: CirclePointIndex) -> Field {
        self.eval_at_point((point_index + i).to_point())
    }
}

// TODO(spapini): make an iterator instead, so we do all computations beforehand.
impl PolyOracle for CircleEvaluation {
    fn get_at(&self, i: CirclePointIndex, point_index: CirclePointIndex) -> Field {
        let d = self.domain.find(i + point_index).expect("Not in domain");
        self.values[d]
    }
}

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
            assert_eq!(coset_vanishing(*c0, el), Field::zero());
            for c1 in cosets.iter() {
                if c0 == c1 {
                    continue;
                }
                assert_ne!(coset_vanishing(*c1, el), Field::zero());
            }
        }
    }
}

#[test]
fn test_point_excluder() {
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
            assert_eq!(point_vanishing(vanish_point, el), Field::zero());
            continue;
        }
        if el == vanish_point.antipode() {
            continue;
        }
        assert_ne!(point_vanishing(vanish_point, el), Field::zero());
    }
}

#[test]
#[should_panic(expected = "0 has no inverse")]
fn test_point_vanishing_failure() {
    let coset = Coset::half_odds(6);
    let point = coset.at(4);
    point_vanishing(point, point.antipode());
}
