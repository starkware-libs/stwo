use super::CPUBackend;
use crate::core::circle::CirclePoint;
use crate::core::fft::{butterfly, ibutterfly};
use crate::core::fields::m31::BaseField;
use crate::core::fields::{Col, ExtensionOf, Field, FieldOps};
use crate::core::poly::circle::{
    CanonicCoset, CircleDomain, CircleEvaluation, CirclePoly, PolyOps,
};
use crate::core::poly::utils::fold;
use crate::core::poly::BitReversedOrder;
use crate::core::utils::bit_reverse;

fn get_twiddles(domain: CircleDomain) -> Vec<Vec<BaseField>> {
    let mut coset = domain.half_coset;

    let mut res = vec![];
    res.push(coset.iter().map(|p| (p.y)).collect::<Vec<_>>());
    bit_reverse(res.last_mut().unwrap());
    for _ in 0..coset.log_size() {
        res.push(
            coset
                .iter()
                .take(coset.size() / 2)
                .map(|p| (p.x))
                .collect::<Vec<_>>(),
        );
        bit_reverse(res.last_mut().unwrap());
        coset = coset.double();
    }

    res
}

impl<F: ExtensionOf<BaseField>> PolyOps<F> for CPUBackend {
    fn new_canonical_ordered(
        coset: CanonicCoset,
        values: Col<Self, F>,
    ) -> CircleEvaluation<Self, F, BitReversedOrder> {
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

    fn interpolate(eval: CircleEvaluation<Self, F, BitReversedOrder>) -> CirclePoly<Self, F> {
        let twiddles = get_twiddles(eval.domain);

        let mut values = eval.values;
        for (i, layer_twiddles) in twiddles.iter().enumerate() {
            for (h, &t) in layer_twiddles.iter().enumerate() {
                for l in 0..(1 << i) {
                    let idx0 = (h << (i + 1)) + l;
                    let idx1 = idx0 + (1 << i);
                    let (mut val0, mut val1) = (values[idx0], values[idx1]);
                    ibutterfly(&mut val0, &mut val1, t.inverse());
                    (values[idx0], values[idx1]) = (val0, val1);
                }
            }
        }

        // Divide all values by 2^log_size.
        let inv = BaseField::from_u32_unchecked(eval.domain.size() as u32).inverse();
        for val in &mut values {
            *val *= inv;
        }

        CirclePoly::new(values)
    }

    fn eval_at_point<E: ExtensionOf<F>>(poly: &CirclePoly<Self, F>, point: CirclePoint<E>) -> E {
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

    fn extend(poly: &CirclePoly<Self, F>, log_size: u32) -> CirclePoly<Self, F> {
        assert!(log_size >= poly.log_size());
        let mut coeffs = Vec::with_capacity(1 << log_size);
        coeffs.extend_from_slice(&poly.coeffs);
        coeffs.resize(1 << log_size, F::zero());
        CirclePoly::new(coeffs)
    }

    fn evaluate(
        poly: &CirclePoly<Self, F>,
        domain: CircleDomain,
    ) -> CircleEvaluation<Self, F, BitReversedOrder> {
        let twiddles = get_twiddles(domain);

        let mut values = poly.extend(domain.log_size()).coeffs;
        for (i, layer_twiddles) in twiddles.iter().enumerate().rev() {
            for (h, &t) in layer_twiddles.iter().enumerate() {
                for l in 0..(1 << i) {
                    let idx0 = (h << (i + 1)) + l;
                    let idx1 = idx0 + (1 << i);
                    let (mut val0, mut val1) = (values[idx0], values[idx1]);
                    butterfly(&mut val0, &mut val1, t);
                    (values[idx0], values[idx1]) = (val0, val1);
                }
            }
        }
        CircleEvaluation::new(domain, values)
    }
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
