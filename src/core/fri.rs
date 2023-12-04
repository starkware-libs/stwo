use std::iter::zip;

use super::fields::m31::BaseField;
use super::fields::{ExtensionOf, Field};
use crate::core::circle::Coset;
use crate::core::fft::ibutterfly;
use crate::core::poly::line::LineDomain;

/// Performs a degree respecting projection (DRP) on a polynomial.
///
/// Exmaple: Our evaluation domain is the x-coordinates of `E = c + <G>, |E| = 8`, `alpha`
/// is a random field element and `pi(x) = 2x^2 - 1` is the circle's x-coordinate doubling map. We
/// have evaluations of a polynomial `f` (i.e `evals`) and we can compute the evaluations of `f' =
/// 2 * (fe + alpha * fo)` such that `f(x) = fe(pi(x)) + x * fo(pi(x))`.
///
/// `evals` should be polynomial evaluations over a [LineDomain] stored in natural order. The return
/// evaluations are evaluations over a [LineDomain] of half the size stored in natural order.
///
/// # Panics
///
/// Panics if the number of evaluations is not a power of two greater than or equal to two.
pub fn apply_drp<F: ExtensionOf<BaseField>>(evals: &[F], alpha: F) -> Vec<F> {
    let n = evals.len();
    assert!(n.is_power_of_two());
    assert!(n >= 2);

    let (l, r) = evals.split_at(n / 2);

    let domain = LineDomain::new(Coset::half_odds(n.ilog2() as usize));

    zip(zip(l, r), domain.iter())
        .map(|((&f_x, &f_neg_x), x)| {
            let (mut f_e, mut f_o) = (f_x, f_neg_x);
            ibutterfly(&mut f_e, &mut f_o, x.inverse());
            f_e + alpha * f_o
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use std::iter::zip;

    use crate::core::circle::Coset;
    use crate::core::fields::m31::BaseField;
    use crate::core::fri::apply_drp;
    use crate::core::poly::line::{LineDomain, LinePoly};

    #[test]
    fn drp_works() {
        const DEGREE: usize = 8;
        // Coefficients are bit-reversed.
        let even_coeffs: [BaseField; DEGREE / 2] = [1, 2, 1, 3].map(BaseField::from_u32_unchecked);
        let odd_coeffs: [BaseField; DEGREE / 2] = [3, 5, 4, 1].map(BaseField::from_u32_unchecked);
        let poly = LinePoly::new([even_coeffs, odd_coeffs].concat());
        let even_poly = LinePoly::new(even_coeffs.to_vec());
        let odd_poly = LinePoly::new(odd_coeffs.to_vec());
        let alpha = BaseField::from_u32_unchecked(19283);
        let domain = LineDomain::new(Coset::half_odds(DEGREE.ilog2() as usize));
        let drp_domain = domain.double();
        let evals = poly.evaluate(domain);
        let two = BaseField::from_u32_unchecked(2);

        let drp_evals = apply_drp(&evals, alpha);

        assert_eq!(drp_evals.len(), DEGREE / 2);
        for (i, (drp_eval, x)) in zip(drp_evals, drp_domain.iter()).enumerate() {
            let f_e = even_poly.eval_at_point(x);
            let f_o = odd_poly.eval_at_point(x);
            assert_eq!(drp_eval, two * (f_e + alpha * f_o), "mismatch at {i}");
        }
    }
}
