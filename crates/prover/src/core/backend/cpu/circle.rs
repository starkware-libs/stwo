use num_traits::Zero;
use stwo_verifier::core::fields::m31::BaseField;
use stwo_verifier::core::fields::qm31::SecureField;
use stwo_verifier::core::fields::{ExtensionOf, MulGroup};

use super::CPUBackend;
use crate::core::backend::{Col, ColumnOps};
use crate::core::circle::{CirclePoint, Coset};
use crate::core::fft::{butterfly, ibutterfly};
use crate::core::poly::circle::{
    CanonicCoset, CircleDomain, CircleEvaluation, CirclePoly, PolyOps,
};
use crate::core::poly::twiddles::TwiddleTree;
use crate::core::poly::utils::{domain_line_twiddles_from_tree, fold};
use crate::core::poly::BitReversedOrder;
use crate::core::utils::bit_reverse;

impl PolyOps for CPUBackend {
    type Twiddles = Vec<BaseField>;

    fn new_canonical_ordered(
        coset: CanonicCoset,
        values: Col<Self, BaseField>,
    ) -> CircleEvaluation<Self, BaseField, BitReversedOrder> {
        let domain = coset.circle_domain();
        assert_eq!(values.len(), domain.size());
        let mut new_values = Vec::with_capacity(values.len());
        let half_len = 1 << (coset.log_size() - 1);
        for i in 0..half_len {
            new_values.push(values[i << 1]);
        }
        for i in 0..half_len {
            new_values.push(values[domain.size() - 1 - (i << 1)]);
        }
        CPUBackend::bit_reverse_column(&mut new_values);
        CircleEvaluation::new(domain, new_values)
    }

    fn interpolate(
        eval: CircleEvaluation<Self, BaseField, BitReversedOrder>,
        twiddles: &TwiddleTree<Self>,
    ) -> CirclePoly<Self> {
        let mut values = eval.values;

        assert!(eval.domain.half_coset.is_doubling_of(twiddles.root_coset));
        let line_twiddles = domain_line_twiddles_from_tree(eval.domain, &twiddles.itwiddles);

        if eval.domain.log_size() == 1 {
            let (mut val0, mut val1) = (values[0], values[1]);
            ibutterfly(
                &mut val0,
                &mut val1,
                eval.domain.half_coset.initial.y.inverse(),
            );
            let inv = BaseField::from_u32_unchecked(2).inverse();
            (values[0], values[1]) = (val0 * inv, val1 * inv);
            return CirclePoly::new(values);
        };

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
        // TODO(Andrew): Allocation here expensive for small polynomials.
        let mut mappings = vec![point.y, point.x];
        let mut x = point.x;
        for _ in 2..poly.log_size() {
            x = CirclePoint::double_x(x);
            mappings.push(x);
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
        let mut values = poly.extend(domain.log_size()).coeffs;

        assert!(domain.half_coset.is_doubling_of(twiddles.root_coset));
        let line_twiddles = domain_line_twiddles_from_tree(domain, &twiddles.twiddles);

        if domain.log_size() == 1 {
            let (mut val0, mut val1) = (values[0], values[1]);
            butterfly(&mut val0, &mut val1, domain.half_coset.initial.y.inverse());
            return CircleEvaluation::new(domain, values);
        };

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

    fn precompute_twiddles(mut coset: Coset) -> TwiddleTree<Self> {
        const CHUNK_LOG_SIZE: usize = 12;
        const CHUNK_SIZE: usize = 1 << CHUNK_LOG_SIZE;

        let root_coset = coset;
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
        twiddles.push(1.into());

        // Inverse twiddles.
        // Fallback to the non-chunked version if the domain is not big enough.
        if CHUNK_SIZE > coset.size() {
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
                BaseField::batch_inverse(src, dst);
            });

        TwiddleTree {
            root_coset,
            twiddles,
            itwiddles,
        }
    }
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
    for CircleEvaluation<CPUBackend, F, EvalOrder>
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
