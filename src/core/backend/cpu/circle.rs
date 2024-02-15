use super::CPUBackend;
use crate::core::backend::Column;
use crate::core::circle::CirclePoint;
use crate::core::fft::{butterfly, ibutterfly};
use crate::core::fields::m31::BaseField;
use crate::core::fields::{ExtensionOf, Field};
use crate::core::poly::circle::{
    CanonicCoset, CircleDomain, CircleEvaluation, CirclePoly, PolyOps,
};
use crate::core::poly::utils::fold;

impl<F: ExtensionOf<BaseField>> PolyOps<F> for CPUBackend {
    fn new_canonical_ordered(
        coset: CanonicCoset,
        values: Column<Self, F>,
    ) -> CircleEvaluation<Self, F> {
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
        CircleEvaluation::new(domain, new_values)
    }

    fn interpolate(eval: CircleEvaluation<Self, F>) -> CirclePoly<Self, F> {
        // Use CFFT to interpolate.
        let mut coset = eval.domain.half_coset;
        let mut values = eval.values;
        let (l, r) = values.split_at_mut(coset.size());
        for (i, p) in coset.iter().enumerate() {
            ibutterfly(&mut l[i], &mut r[i], p.y.inverse());
        }
        while coset.size() > 1 {
            for chunk in values.chunks_exact_mut(coset.size()) {
                let (l, r) = chunk.split_at_mut(coset.size() / 2);
                for (i, p) in coset.iter().take(coset.size() / 2).enumerate() {
                    ibutterfly(&mut l[i], &mut r[i], p.x.inverse());
                }
            }
            coset = coset.double();
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
        fold(&poly.coeffs, &mappings)
    }

    fn extend(poly: &CirclePoly<Self, F>, log_size: u32) -> CirclePoly<Self, F> {
        assert!(log_size >= poly.log_size());
        let mut coeffs = vec![F::zero(); 1 << log_size];
        let log_jump = log_size - poly.log_size();
        for (i, val) in poly.coeffs.iter().enumerate() {
            coeffs[i << log_jump] = *val;
        }
        CirclePoly::new(coeffs)
    }

    fn evaluate(poly: &CirclePoly<Self, F>, domain: CircleDomain) -> CircleEvaluation<Self, F> {
        // Use CFFT to evaluate.
        let mut coset = domain.half_coset;
        let mut cosets = vec![];

        // TODO(spapini): extend better.
        assert!(domain.log_size() >= poly.log_size());
        let mut values = poly.clone().extend(domain.log_size()).coeffs;

        while coset.size() > 1 {
            cosets.push(coset);
            coset = coset.double();
        }
        for coset in cosets.iter().rev() {
            for chunk in values.chunks_exact_mut(coset.size()) {
                let (l, r) = chunk.split_at_mut(coset.size() / 2);
                for (i, p) in coset.iter().take(coset.size() / 2).enumerate() {
                    butterfly(&mut l[i], &mut r[i], p.x);
                }
            }
        }
        let coset = domain.half_coset;
        let (l, r) = values.split_at_mut(coset.size());
        for (i, p) in coset.iter().enumerate() {
            butterfly(&mut l[i], &mut r[i], p.y);
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

// impl<F: ExtensionOf<BaseField>> PolyOps<CPUBackend, F>
