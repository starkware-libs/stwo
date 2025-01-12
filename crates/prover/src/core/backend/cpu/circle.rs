use num_traits::Zero;

use super::CpuBackend;
use crate::core::backend::cpu::bit_reverse;
use crate::core::backend::{Col, ColumnOps};
use crate::core::circle::{CirclePoint, Coset};
use crate::core::fft::{butterfly, ibutterfly};
use crate::core::fields::m31::BaseField;
use crate::core::fields::qm31::SecureField;
use crate::core::fields::{batch_inverse_in_place, ExtensionOf};
use crate::core::poly::circle::{
    CanonicCoset, CircleDomain, CircleEvaluation, CirclePoly, PolyOps,
};
use crate::core::poly::twiddles::TwiddleTree;
use crate::core::poly::utils::{domain_line_twiddles_from_tree, fold};
use crate::core::poly::BitReversedOrder;
use crate::core::utils::coset_order_to_circle_domain_order;

impl PolyOps for CpuBackend {
    type Twiddles = Vec<BaseField>;

    fn new_canonical_ordered(
        coset: CanonicCoset,
        values: Col<Self, BaseField>,
    ) -> CircleEvaluation<Self, BaseField, BitReversedOrder> {
        let domain = coset.circle_domain();
        assert_eq!(values.len(), domain.size());
        let mut new_values = coset_order_to_circle_domain_order(&values);
        CpuBackend::bit_reverse_column(&mut new_values);
        CircleEvaluation::new(domain, new_values)
    }

    fn interpolate(
        eval: CircleEvaluation<Self, BaseField, BitReversedOrder>,
        twiddles: &TwiddleTree<Self>,
    ) -> CirclePoly<Self> {
        assert!(eval.domain.half_coset.is_doubling_of(twiddles.root_coset));

        let mut values = eval.values;

        if eval.domain.log_size() == 1 {
            let y = eval.domain.half_coset.initial.y;
            let n = BaseField::from(2);
            let yn_inv = (y * n).inverse();
            let y_inv = yn_inv * n;
            let n_inv = yn_inv * y;
            let (mut v0, mut v1) = (values[0], values[1]);
            ibutterfly(&mut v0, &mut v1, y_inv);
            return CirclePoly::new(vec![v0 * n_inv, v1 * n_inv]);
        }

        if eval.domain.log_size() == 2 {
            let CirclePoint { x, y } = eval.domain.half_coset.initial;
            let n = BaseField::from(4);
            let xyn_inv = (x * y * n).inverse();
            let x_inv = xyn_inv * y * n;
            let y_inv = xyn_inv * x * n;
            let n_inv = xyn_inv * x * y;
            let (mut v0, mut v1, mut v2, mut v3) = (values[0], values[1], values[2], values[3]);
            ibutterfly(&mut v0, &mut v1, y_inv);
            ibutterfly(&mut v2, &mut v3, -y_inv);
            ibutterfly(&mut v0, &mut v2, x_inv);
            ibutterfly(&mut v1, &mut v3, x_inv);
            return CirclePoly::new(vec![v0 * n_inv, v1 * n_inv, v2 * n_inv, v3 * n_inv]);
        }

        let line_twiddles = domain_line_twiddles_from_tree(eval.domain, &twiddles.itwiddles);
        let circle_twiddles = circle_twiddles_from_line_twiddles(line_twiddles[0]);

        for (h, t) in circle_twiddles.enumerate() {
            fft_layer_loop(&mut values, 0, h, t, ibutterfly);
        }
        for (layer, layer_twiddles) in line_twiddles.into_iter().enumerate() {
            for (h, &t) in layer_twiddles.iter().enumerate() {
                fft_layer_loop(&mut values, layer + 1, h, t, ibutterfly);
            }
        }

        // Divide all values by 2^log_size.
        let inv = BaseField::from_u32_unchecked(eval.domain.size() as u32).inverse();
        for val in &mut values {
            *val *= inv;
        }

        CirclePoly::new(values)
    }

    fn eval_at_point(poly: &CirclePoly<Self>, point: CirclePoint<SecureField>) -> SecureField {
        if poly.log_size() == 0 {
            return poly.coeffs[0].into();
        }

        let mut mappings = vec![point.y];
        let mut x = point.x;
        for _ in 1..poly.log_size() {
            mappings.push(x);
            x = CirclePoint::double_x(x);
        }
        mappings.reverse();

        fold(&poly.coeffs, &mappings)
    }

    fn extend(poly: &CirclePoly<Self>, log_size: u32) -> CirclePoly<Self> {
        assert!(log_size >= poly.log_size());
        let mut coeffs = Vec::with_capacity(1 << log_size);
        coeffs.extend_from_slice(&poly.coeffs);
        coeffs.resize(1 << log_size, BaseField::zero());
        CirclePoly::new(coeffs)
    }

    fn evaluate(
        poly: &CirclePoly<Self>,
        domain: CircleDomain,
        twiddles: &TwiddleTree<Self>,
    ) -> CircleEvaluation<Self, BaseField, BitReversedOrder> {
        assert!(domain.half_coset.is_doubling_of(twiddles.root_coset));

        let mut values = poly.extend(domain.log_size()).coeffs;

        if domain.log_size() == 1 {
            let (mut v0, mut v1) = (values[0], values[1]);
            butterfly(&mut v0, &mut v1, domain.half_coset.initial.y);
            return CircleEvaluation::new(domain, vec![v0, v1]);
        }

        if domain.log_size() == 2 {
            let (mut v0, mut v1, mut v2, mut v3) = (values[0], values[1], values[2], values[3]);
            let CirclePoint { x, y } = domain.half_coset.initial;
            butterfly(&mut v0, &mut v2, x);
            butterfly(&mut v1, &mut v3, x);
            butterfly(&mut v0, &mut v1, y);
            butterfly(&mut v2, &mut v3, -y);
            return CircleEvaluation::new(domain, vec![v0, v1, v2, v3]);
        }

        let line_twiddles = domain_line_twiddles_from_tree(domain, &twiddles.twiddles);
        let circle_twiddles = circle_twiddles_from_line_twiddles(line_twiddles[0]);

        for (layer, layer_twiddles) in line_twiddles.iter().enumerate().rev() {
            for (h, &t) in layer_twiddles.iter().enumerate() {
                fft_layer_loop(&mut values, layer + 1, h, t, butterfly);
            }
        }
        for (h, t) in circle_twiddles.enumerate() {
            fft_layer_loop(&mut values, 0, h, t, butterfly);
        }

        CircleEvaluation::new(domain, values)
    }

    fn precompute_twiddles(coset: Coset) -> TwiddleTree<Self> {
        const CHUNK_LOG_SIZE: usize = 12;
        const CHUNK_SIZE: usize = 1 << CHUNK_LOG_SIZE;

        let root_coset = coset;
        let twiddles = slow_precompute_twiddles(coset);

        // Inverse twiddles.
        // Fallback to the non-chunked version if the domain is not big enough.
        if CHUNK_SIZE > root_coset.size() {
            let itwiddles = twiddles.iter().map(|&t| t.inverse()).collect();
            return TwiddleTree {
                root_coset,
                twiddles,
                itwiddles,
            };
        }

        let mut itwiddles = vec![BaseField::zero(); twiddles.len()];
        twiddles
            .array_chunks::<CHUNK_SIZE>()
            .zip(itwiddles.array_chunks_mut::<CHUNK_SIZE>())
            .for_each(|(src, dst)| {
                batch_inverse_in_place(src, dst);
            });

        TwiddleTree {
            root_coset,
            twiddles,
            itwiddles,
        }
    }
}

pub fn slow_precompute_twiddles(mut coset: Coset) -> Vec<BaseField> {
    let mut twiddles = Vec::with_capacity(coset.size());
    for _ in 0..coset.log_size() {
        let i0 = twiddles.len();
        twiddles.extend(
            coset
                .iter()
                .take(coset.size() / 2)
                .map(|p| p.x)
                .collect::<Vec<_>>(),
        );
        bit_reverse(&mut twiddles[i0..]);
        coset = coset.double();
    }
    // Pad with an arbitrary value to make the length a power of 2.
    twiddles.push(1.into());
    twiddles
}

fn fft_layer_loop(
    values: &mut [BaseField],
    i: usize,
    h: usize,
    t: BaseField,
    butterfly_fn: impl Fn(&mut BaseField, &mut BaseField, BaseField),
) {
    for l in 0..(1 << i) {
        let idx0 = (h << (i + 1)) + l;
        let idx1 = idx0 + (1 << i);
        let (mut val0, mut val1) = (values[idx0], values[idx1]);
        butterfly_fn(&mut val0, &mut val1, t);
        (values[idx0], values[idx1]) = (val0, val1);
    }
}

/// Computes the circle twiddles layer (layer 0) from the first line twiddles layer (layer 1).
///
/// Only works for line twiddles generated from a domain with size `>4`.
fn circle_twiddles_from_line_twiddles(
    first_line_twiddles: &[BaseField],
) -> impl Iterator<Item = BaseField> + '_ {
    // The twiddles for layer 0 can be computed from the twiddles for layer 1.
    // Since the twiddles are bit reversed, we consider the circle domain in bit reversed order.
    // Each consecutive 4 points in the bit reversed order of a coset form a circle coset of size 4.
    // A circle coset of size 4 in bit reversed order looks like this:
    //   [(x, y), (-x, -y), (y, -x), (-y, x)]
    // Note: This relation is derived from the fact that `M31_CIRCLE_GEN`.repeated_double(ORDER / 4)
    //   == (-1,0), and not (0,1). (0,1) would yield another relation.
    // The twiddles for layer 0 are the y coordinates:
    //   [y, -y, -x, x]
    // The twiddles for layer 1 in bit reversed order are the x coordinates of the even indices
    // points:
    //   [x, y]
    // Works also for inverse of the twiddles.
    first_line_twiddles
        .iter()
        .array_chunks()
        .flat_map(|[&x, &y]| [y, -y, -x, x])
}

impl<F: ExtensionOf<BaseField>, EvalOrder> IntoIterator
    for CircleEvaluation<CpuBackend, F, EvalOrder>
{
    type Item = F;
    type IntoIter = std::vec::IntoIter<F>;

    /// Creates a consuming iterator over the evaluations.
    ///
    /// Evaluations are returned in the same order as elements of the domain.
    fn into_iter(self) -> Self::IntoIter {
        self.values.into_iter()
    }
}

#[cfg(test)]
mod tests {
    use std::iter::zip;

    use num_traits::One;

    use crate::core::backend::cpu::CpuCirclePoly;
    use crate::core::circle::CirclePoint;
    use crate::core::fields::m31::BaseField;
    use crate::core::fields::qm31::SecureField;
    use crate::core::poly::circle::CanonicCoset;

    #[test]
    fn test_eval_at_point_with_4_coeffs() {
        // Represents the polynomial `1 + 2y + 3x + 4xy`.
        // Note coefficients are passed in bit reversed order.
        let poly = CpuCirclePoly::new([1, 3, 2, 4].map(BaseField::from).to_vec());
        let x = BaseField::from(5).into();
        let y = BaseField::from(8).into();

        let eval = poly.eval_at_point(CirclePoint { x, y });

        assert_eq!(
            eval,
            poly.coeffs[0] + poly.coeffs[1] * y + poly.coeffs[2] * x + poly.coeffs[3] * x * y
        );
    }

    #[test]
    fn test_eval_at_point_with_2_coeffs() {
        // Represents the polynomial `1 + 2y`.
        let poly = CpuCirclePoly::new(vec![BaseField::from(1), BaseField::from(2)]);
        let x = BaseField::from(5).into();
        let y = BaseField::from(8).into();

        let eval = poly.eval_at_point(CirclePoint { x, y });

        assert_eq!(eval, poly.coeffs[0] + poly.coeffs[1] * y);
    }

    #[test]
    fn test_eval_at_point_with_1_coeff() {
        // Represents the polynomial `1`.
        let poly = CpuCirclePoly::new(vec![BaseField::one()]);
        let x = BaseField::from(5).into();
        let y = BaseField::from(8).into();

        let eval = poly.eval_at_point(CirclePoint { x, y });

        assert_eq!(eval, SecureField::one());
    }

    #[test]
    fn test_evaluate_2_coeffs() {
        let domain = CanonicCoset::new(1).circle_domain();
        let poly = CpuCirclePoly::new((1..=2).map(BaseField::from).collect());

        let evaluation = poly.clone().evaluate(domain).bit_reverse();

        for (i, (p, eval)) in zip(domain, evaluation).enumerate() {
            let eval: SecureField = eval.into();
            assert_eq!(eval, poly.eval_at_point(p.into_ef()), "mismatch at i={i}");
        }
    }

    #[test]
    fn test_evaluate_4_coeffs() {
        let domain = CanonicCoset::new(2).circle_domain();
        let poly = CpuCirclePoly::new((1..=4).map(BaseField::from).collect());

        let evaluation = poly.clone().evaluate(domain).bit_reverse();

        for (i, (x, eval)) in zip(domain, evaluation).enumerate() {
            let eval: SecureField = eval.into();
            assert_eq!(eval, poly.eval_at_point(x.into_ef()), "mismatch at i={i}");
        }
    }

    #[test]
    fn test_evaluate_8_coeffs() {
        let domain = CanonicCoset::new(3).circle_domain();
        let poly = CpuCirclePoly::new((1..=8).map(BaseField::from).collect());

        let evaluation = poly.clone().evaluate(domain).bit_reverse();

        for (i, (x, eval)) in zip(domain, evaluation).enumerate() {
            let eval: SecureField = eval.into();
            assert_eq!(eval, poly.eval_at_point(x.into_ef()), "mismatch at i={i}");
        }
    }

    #[test]
    fn test_interpolate_2_evals() {
        let poly = CpuCirclePoly::new(vec![BaseField::one(), BaseField::from(2)]);
        let domain = CanonicCoset::new(1).circle_domain();
        let evals = poly.clone().evaluate(domain);

        let interpolated_poly = evals.interpolate();

        assert_eq!(interpolated_poly.coeffs, poly.coeffs);
    }

    #[test]
    fn test_interpolate_4_evals() {
        let poly = CpuCirclePoly::new((1..=4).map(BaseField::from).collect());
        let domain = CanonicCoset::new(2).circle_domain();
        let evals = poly.clone().evaluate(domain);

        let interpolated_poly = evals.interpolate();

        assert_eq!(interpolated_poly.coeffs, poly.coeffs);
    }

    #[test]
    fn test_interpolate_8_evals() {
        let poly = CpuCirclePoly::new((1..=8).map(BaseField::from).collect());
        let domain = CanonicCoset::new(3).circle_domain();
        let evals = poly.clone().evaluate(domain);

        let interpolated_poly = evals.interpolate();

        assert_eq!(interpolated_poly.coeffs, poly.coeffs);
    }
}
