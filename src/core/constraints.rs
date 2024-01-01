use num_traits::One;

use super::circle::{CirclePoint, CirclePointIndex, Coset};
use super::fields::m31::BaseField;
use super::fields::ExtensionOf;
use super::poly::circle::{CircleDomain, CirclePoly, Evaluation, PointMapping};

/// Evaluates a vanishing polynomial of the coset at a point.
pub fn coset_vanishing<F: ExtensionOf<BaseField>>(coset: Coset, mut p: CirclePoint<F>) -> F {
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
    p = p - coset.initial.into_ef() + coset.step_size.half().to_point().into_ef();
    let mut x = p.x;

    // The formula for the x coordinate of the double of a point.
    for _ in 0..(coset.n_bits - 1) {
        x = CirclePoint::double_x(x);
    }
    x
}

pub fn circle_domain_vanishing(domain: CircleDomain, p: CirclePoint<BaseField>) -> BaseField {
    coset_vanishing(domain.half_coset, p) * coset_vanishing(domain.half_coset.conjugate(), p)
}

/// Evaluates the polynomial that is used to exclude the excluded point at point
/// p. Note that this polynomial has a zero of multiplicity 2 at the excluded
/// point.
pub fn point_excluder<F: ExtensionOf<BaseField>>(
    excluded: CirclePoint<BaseField>,
    p: CirclePoint<F>,
) -> F {
    (p - excluded.into_ef()).x - BaseField::one()
}

/// Evaluates a vanishing polynomial of the vanish_point at a point.
/// Note that this function has a pole on the antipode of the vanish_point.
pub fn point_vanishing<F: ExtensionOf<BaseField>, EF: ExtensionOf<F>>(
    vanish_point: CirclePoint<F>,
    p: CirclePoint<EF>,
) -> EF {
    let h = p - vanish_point.into_ef();
    h.y / (EF::one() + h.x)
}

/// Utils for computing constraints.
/// Oracle to a polynomial constrained to a coset.
pub trait PolyOracle<F: ExtensionOf<BaseField>>: Copy {
    fn get_at(&self, index: CirclePointIndex) -> F;
    fn point(&self) -> CirclePoint<F>;
}

#[derive(Copy, Clone)]
pub struct EvalByPoly<'a, F: ExtensionOf<BaseField>> {
    pub point: CirclePoint<F>,
    pub poly: &'a CirclePoly<BaseField>,
}

impl<'a, F: ExtensionOf<BaseField>> PolyOracle<F> for EvalByPoly<'a, F> {
    fn point(&self) -> CirclePoint<F> {
        self.point
    }

    fn get_at(&self, index: CirclePointIndex) -> F {
        let eval_point = self.point + index.to_point().into_ef();
        self.poly.eval_at_point(eval_point)
    }
}

// TODO(spapini): make an iterator instead, so we do all computations beforehand.
/// Polynomial evaluation over the base circle domain. The evaluation could be over an extension of
/// the base field.
#[derive(Clone)]
pub struct EvalByEvaluation<'a, F: ExtensionOf<BaseField>, T: Evaluation<F>> {
    pub offset: CirclePointIndex,
    pub eval: &'a T,
    pub _eval_field: std::marker::PhantomData<F>,
}

impl<'a, F: ExtensionOf<BaseField>, T: Evaluation<F>> EvalByEvaluation<'a, F, T> {
    pub fn new(offset: CirclePointIndex, eval: &'a T) -> Self {
        Self {
            offset,
            eval,
            _eval_field: std::marker::PhantomData,
        }
    }
}

impl<'a, F: ExtensionOf<BaseField>, T: Evaluation<F>> PolyOracle<F> for EvalByEvaluation<'a, F, T> {
    fn point(&self) -> CirclePoint<F> {
        // TODO(AlonH): Separate the point field generality from the evaluation field generality and
        // remove the `into_ef` here.
        self.offset.to_point().into_ef()
    }

    fn get_at(&self, index: CirclePointIndex) -> F {
        self.eval.get_at(index + self.offset)
    }
}

impl<'a, F: ExtensionOf<BaseField>, T: Evaluation<F>> Copy for EvalByEvaluation<'a, F, T> {}

#[derive(Clone)]
pub struct EvalByPointMapping<'a, F: ExtensionOf<BaseField>> {
    pub point: CirclePoint<F>,
    pub point_mapping: &'a PointMapping<F>,
}

impl<'a, F: ExtensionOf<BaseField>> PolyOracle<F> for EvalByPointMapping<'a, F> {
    fn point(&self) -> CirclePoint<F> {
        // TODO(AlonH): Separate the point field generality from the evaluation field generality and
        // remove the `into_ef` here.
        self.point
    }

    fn get_at(&self, index: CirclePointIndex) -> F {
        self.point_mapping
            .get_at(self.point + index.to_point().into_ef())
    }
}

impl<'a, F: ExtensionOf<BaseField>> Copy for EvalByPointMapping<'a, F> {}

#[cfg(test)]
mod tests {
    use num_traits::Zero;

    use super::{coset_vanishing, point_excluder, point_vanishing};
    use crate::core::circle::{CirclePointIndex, Coset};
    use crate::core::fields::m31::BaseField;
    use crate::core::fields::Field;

    #[test]
    fn test_coset_vanishing() {
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
        let excluded = Coset::half_odds(5).at(10);
        let point = (CirclePointIndex::generator() * 4).to_point();

        let num = point_excluder(excluded, point) * point_excluder(excluded.conjugate(), point);
        let denom = (point.x - excluded.x).pow(2);

        assert_eq!(num, denom);
    }

    #[test]
    fn test_point_vanishing_success() {
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
}
