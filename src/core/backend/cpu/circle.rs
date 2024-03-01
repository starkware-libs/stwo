use itertools::Itertools;

use super::CPUBackend;
use crate::core::circle::{CirclePoint, CirclePointIndex, Coset};
use crate::core::fft::{butterfly, ibutterfly};
use crate::core::fields::m31::BaseField;
use crate::core::fields::{Col, ExtensionOf, Field};
use crate::core::poly::circle::{
    CanonicCoset, CircleDomain, CircleEvaluation, CircleGroupEvaluation, CirclePoly, Group, PolyOps,
};
use crate::core::poly::utils::fold;
use crate::core::utils::bit_reverse;

impl<F: ExtensionOf<BaseField>> PolyOps<F> for CPUBackend {
    fn new_canonical_ordered(
        coset: CanonicCoset,
        values: Col<Self, F>,
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

// TODO(spapini): Document.

#[allow(dead_code)]
pub fn get_tower_points(log_size: u32) -> Vec<CirclePoint<BaseField>> {
    let mut points = vec![CirclePointIndex::zero().to_point()];
    for log_size in 0..log_size {
        let mut a = Coset::half_odds(log_size).iter().collect_vec();
        bit_reverse(&mut a);
        points.extend(a);
    }
    assert_eq!(points.len(), 1 << log_size);
    points
}

#[allow(dead_code)]
pub fn get_double_tower_points(log_size: u32) -> Vec<CirclePoint<BaseField>> {
    let mut points = vec![
        CirclePointIndex::zero().to_point(),
        CirclePointIndex::subgroup_gen(1).to_point(),
    ];
    for log_size in 0..(log_size - 1) {
        let mut a = Coset::half_odds(log_size).iter().collect_vec();
        bit_reverse(&mut a);
        points.extend(a.into_iter().flat_map(|p| [p, -p]));
    }
    assert_eq!(points.len(), 1 << log_size);
    points
}

#[allow(dead_code)]
pub fn evaluate_on_group(
    poly: &CirclePoly<CPUBackend, BaseField>,
    group: Group,
) -> CircleGroupEvaluation<CPUBackend> {
    let log_size = poly.log_size();
    assert_eq!(group.log_size(), log_size);
    let log_size_line = log_size - 1;

    // Get twiddles.
    let twiddles = get_tower_points(log_size_line)
        .iter()
        .map(|p| p.x)
        .collect_vec();
    assert_eq!(twiddles.len(), 1 << log_size_line);

    let mut values = poly.coeffs.clone();
    // TODO(spapini): Remove this bit_reverse when poly and eval are bit reversed.
    bit_reverse(&mut values);
    let mut eval_at_minus1 = values[..1 << log_size_line].to_vec();
    for i in (1..log_size).rev() {
        for h_point in 0..(1 << (log_size - i - 1)) {
            for l_poly in 0..(1 << i) {
                let idx = (h_point << (i + 1)) + l_poly;
                let (mut x, mut y) = (values[idx], values[idx + (1 << i)]);
                let twid = twiddles[h_point << 1];
                butterfly(&mut x, &mut y, twid);
                (values[idx], values[idx + (1 << i)]) = (x, y);
            }
        }
        for l_poly in 0..(1 << i) {
            std::mem::swap(&mut eval_at_minus1[l_poly], &mut values[l_poly + (1 << i)]);
        }
    }

    // Last layer.
    let twiddles = get_tower_points(log_size_line)
        .iter()
        .map(|p| p.y)
        .collect_vec();
    #[allow(clippy::needless_range_loop)]
    for h_point in 0..(1 << log_size_line) {
        let idx = h_point << 1;
        let (mut x, mut y) = (values[idx], values[idx + 1]);
        let twid = twiddles[h_point];
        butterfly(&mut x, &mut y, twid);
        (values[idx], values[idx + 1]) = (x, y);
    }
    values[1] = eval_at_minus1[0];

    CircleGroupEvaluation::new(group, values)
}

#[cfg(test)]
mod tests {
    use rand::rngs::StdRng;
    use rand::{Rng, SeedableRng};

    use crate::core::backend::cpu::circle::{evaluate_on_group, get_double_tower_points};
    use crate::core::backend::cpu::CPUCirclePoly;
    use crate::core::fields::m31::{BaseField, P};
    use crate::core::poly::circle::Group;
    use crate::core::utils::bit_reverse;

    #[test]
    fn test_evaluate_on_group() {
        const LOG_SIZE: usize = 4;
        let rng = &mut StdRng::seed_from_u64(0);
        let coeffs = (0..(1 << LOG_SIZE))
            .map(|_| BaseField::from(rng.gen::<u32>() % P))
            .collect::<Vec<_>>();
        let mut poly_coeffs = coeffs.clone();
        bit_reverse(&mut poly_coeffs);
        let poly = CPUCirclePoly::new(poly_coeffs);
        let eval = evaluate_on_group(&poly, Group::new(LOG_SIZE as u32));
        let points = get_double_tower_points(LOG_SIZE as u32);

        let expected = points
            .iter()
            .map(|p| poly.eval_at_point(*p))
            .collect::<Vec<_>>();
        assert_eq!(eval.values, expected);
    }
}
