use std::iter::zip;

use super::fields::m31::BaseField;
use super::fields::{ExtensionOf, Field};
use crate::core::circle::Coset;
use crate::core::fft::ibutterfly;
use crate::core::poly::line::LineDomain;

/// Performs a degree respecting projection (DRP) on a polynomial.
///
/// For example, when our evaluation domain is the x-coordinates of `E = c + <G>, |E| = 8` and
/// `pi(x) = 2x^2 - 1` is the circle's x-coordinate doubling map:
///
/// 1. Interpolate evals over the domain to obtain coefficients of f(x):
///
/// ```text
///    +---------+---+---+---+---+---+---+---+---+
///    | i       | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 |
///    +---------+---+---+---+---+---+---+---+---+
///    | eval[i] | 9 | 2 | 3 | 5 | 9 | 2 | 3 | 5 |
///    +---------+---+---+---+---+---+---+---+---+
///    +--------+-------+-------+-------+-------+-------+-------+-------+-------+
///    | p      | c+0*G | c+1*G | c+2*G | c+3*G | c+4*G | c+5*G | c+6*G | c+7*G |
///    +--------+-------+-------+-------+-------+-------+-------+-------+-------+
///    | f(p.x) | 9     | 9     | 3     | 2     | 2     | 5     | 3     | 5     |
///    +--------+-------+-------+-------+-------+-------+-------+-------+-------+
///      f(x) = c0 +
///             c1 * x +
///             c2 * pi(x) +
///             c3 * pi(x)*x +
///             c4 * pi(pi(x)) +
///             c5 * pi(pi(x))*x +
///             c6 * pi(pi(x))*pi(x) +
///             c7 * pi(pi(x))*pi(x)*x
/// ```
///
/// 2. Perform a random linear combination of odd and even coefficients of f(x):
///
/// ```text
///    f_e(x)   = c0 + c2 * x + c4 * pi(x) + c6 * pi(x)*x
///    f_o(x)   = c1 + c3 * x + c5 * pi(x) + c7 * pi(x)*x
///    f(x)     = f_e(pi(x)) + x * f_o(pi(x))
///    f'(x)    = 2 * f_e(x) + α * 2 * f_o(x)
///    deg(f') <= deg(f) / 2
///    α        = <random field element sent from verifier>
/// ```
///
/// 4. Obtain the DRP by evaluating f'(x) over a new domain of half the size:
///
/// ```text
///    +---------+-----------+-----------+-----------+-----------+
///    | p       | 2*(c+0*G) | 2*(c+1*G) | 2*(c+2*G) | 2*(c+3*G) |
///    +---------+-----------+-----------+-----------+-----------+
///    | f'(p.x) | 82        | 12        | 57        | 34        |
///    +---------+-----------+-----------+-----------+-----------+
///      E' = 2E = 2c + <2G>
///    +--------+----+----+----+----+
///    | i      | 0  | 1  | 2  | 3  |
///    +--------+----+----+----+----+
///    | drp[i] | 82 | 12 | 57 | 34 |
///    +--------+----+----+----+----+
/// ```
///
/// `evals` should be polynomial evaluations over a [LineDomain] stored in bit-reversed order. The
/// return evaluations are also stored in bit-reversed order.
pub fn apply_drp<F: ExtensionOf<BaseField>>(evals: &[F], alpha: F) -> Vec<F> {
    let n = evals.len();
    assert!(n.is_power_of_two());

    // Note `f(x) = f_e(pi(x)) + x * f_o(pi(x))` so `2 * f_e(pi(x)) = f(x) + f(-x)` and
    // `2 * f_o(pi(x)) = (f(x) - f(-x)) / x` therefore we only need `f(x)` and `f(-x)` to compute
    // `f'(pi(x))`. Since all `evals` are bit-reversed `f(x)` and `f(-x)` neighbor each other.
    let eval_pairs = evals.array_chunks();

    let domain = LineDomain::new(Coset::half_odds(n.ilog2() as usize));
    let domain_elements = bit_reversed_domain_elements(domain);
    let domains_elements_inv = batch_inverse(domain_elements);

    zip(eval_pairs, domains_elements_inv)
        .map(|(&[f_x, f_neg_x], x_inv)| {
            let (mut f_e, mut f_o) = (f_x, f_neg_x);
            ibutterfly(&mut f_e, &mut f_o, x_inv);
            f_e + alpha * f_o
        })
        .collect()
}

/// Returns the first half of the domain elements in bit reversed order.
///
/// This is used by [apply_drp] to obtain the domain elements in the same order as the evaluations
/// (bit-reversed). This algorithm is more efficient than generating the domain elements in their
/// natural order and then doing a bit-reversal since the algorithm has almost no additional
/// overhead to generating the domain elements in their natural order.
fn bit_reversed_domain_elements(domain: LineDomain) -> Vec<BaseField> {
    let n = domain.size() / 2;
    let log_n = n.ilog2() as usize;

    // Compute all the required doublings of our generator point.
    let mut doublings = Vec::with_capacity(log_n);
    let mut g = domain.coset().step;
    for _ in 0..log_n {
        doublings.push(g);
        g = g.double();
    }

    // Incrementally produce bit-reversed elements.
    let mut elements = Vec::with_capacity(n);
    elements.push(domain.coset().initial);
    let mut segment_index = 0;
    while let Some(doubling) = doublings.pop() {
        for i in 0..1 << segment_index {
            let element = doubling + elements[i];
            elements.push(element);
        }
        segment_index += 1;
    }

    // We only need the x-coordinates for a [LineDomain].
    elements.into_iter().map(|v| v.x).collect()
}

/// Computes the inverse of all items in `v`.
///
/// Inversions are expensive but their cost can be amortized by batching inversions together.
// TODO(andrew): move to utils
fn batch_inverse<F: Field, U: AsMut<[F]>>(mut v: U) -> U {
    // 1. `[1, a, ab, abc]`
    let mut acc = F::one();
    let n = v.as_mut().len();
    let mut prods = Vec::with_capacity(n);
    for (v, prod) in zip(v.as_mut(), prods.spare_capacity_mut()) {
        prod.write(acc);
        acc *= *v;
    }

    // SAFETY: all values have been initialized
    unsafe { prods.set_len(n) }

    // 2. `1/abcd`
    let mut acc_inv = acc.inverse();

    // 3. `[1/a, a/ab, ab/abc, abc/abcd] = [1/a, 1/b, 1/c, 1/d]`
    for (v, prod) in zip(v.as_mut().iter_mut().rev(), prods.into_iter().rev()) {
        let acc_inv_next = *v * acc_inv;
        *v = acc_inv * prod;
        acc_inv = acc_inv_next;
    }

    v
}

#[cfg(test)]
mod tests {
    use std::iter::zip;

    use super::batch_inverse;
    use crate::core::circle::Coset;
    use crate::core::fields::m31::BaseField;
    use crate::core::fields::Field;
    use crate::core::fri::{apply_drp, bit_reversed_domain_elements};
    use crate::core::poly::line::{LineDomain, LinePoly};

    #[test]
    fn batch_inverse_works() {
        let vals = [7, 3, 2, 129].map(BaseField::from_u32_unchecked);

        let vals_inv = batch_inverse(vals);

        for (i, (val, val_inv)) in zip(vals, vals_inv).enumerate() {
            assert_eq!(val_inv, val.inverse(), "mismatch at {i}");
        }
    }

    #[test]
    fn bit_reversed_domain_elements_works() {
        const N_BITS: usize = 8;
        let domain = LineDomain::new(Coset::half_odds(N_BITS));
        let n = domain.size() / 2;
        let expected_elements = bit_reverse(domain.iter().take(n).collect());

        let bit_reversed_elements = bit_reversed_domain_elements(domain);

        assert_eq!(bit_reversed_elements.len(), n);
        for i in 0..n {
            assert_eq!(
                bit_reversed_elements[i], expected_elements[i],
                "mismatch at {i}"
            );
        }
    }

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
        let bit_reversed_evals = bit_reverse(evals.to_vec());
        let two = BaseField::from_u32_unchecked(2);

        let drp_evals = apply_drp(&bit_reversed_evals, alpha);

        assert_eq!(drp_evals.len(), DEGREE / 2);
        for (i, (drp_eval, x)) in zip(bit_reverse(drp_evals), drp_domain.iter()).enumerate() {
            let f_e = even_poly.eval_at_point(x);
            let f_o = odd_poly.eval_at_point(x);
            assert_eq!(drp_eval, two * (f_e + alpha * f_o), "mismatch at {i}");
        }
    }

    pub fn bit_reverse<T>(mut v: Vec<T>) -> Vec<T> {
        let n = v.len();
        assert!(n.is_power_of_two());
        let n_bits = n.ilog2();
        for i in 0..n {
            let j = i.reverse_bits() >> (usize::BITS - n_bits);
            if j > i {
                v.swap(i, j);
            }
        }
        v
    }
}
