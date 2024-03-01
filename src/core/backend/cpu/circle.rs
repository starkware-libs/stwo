use itertools::Itertools;

use super::CPUBackend;
use crate::core::circle::{CirclePoint, CirclePointIndex, Coset};
use crate::core::fft::{butterfly, ibutterfly};
use crate::core::fields::m31::BaseField;
use crate::core::fields::{Col, ExtensionOf, FieldExpOps, FieldOps};
use crate::core::poly::circle::{
    CanonicCoset, CircleDomain, CircleEvaluation, CircleGroupEvaluation, CirclePoly, Group, PolyOps,
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

// TODO(spapini): Move code and doc to a better place.

// Group position CFFT.
// ====================
// The group is a more complicated domain for circle FFT, since the standard mappings we use are
// not 2-to-1 on the entire domain. Recall that the standard mappings are the projection
// (x,y) -> x, followed by projected doubling x -> 2x^2 - 1. The projection is not 2-to-1 on the
// points (+-1, 0), and 2x^2 - 1 is not 2-to-1 on the x value 0.
//
// Group line domain:
// ==================
// The line domain is formed by the x-projection of a size 2^(n+1) group. It is a set of 2^n + 1
// points. We have one point more than we would like, because (+-1, 0) are 1-to-1.
// For example, the first few such sets are {-1, 0, 1}, {-1, 0, 1, -1/sqrt(2), 1/sqrt(2)}.
//
//    X
//  X   X
// X     X
//  X   X
//    X
// -------
// XX X XX
//
// Ordering:
// =========
// Recall that a canonic line domain is the projection of only the odd points of a group. This is
// used in the canonic fft. Denote it as C_n, with
//   C_n(k)  =  ((4 * bit_rev(k) + 1)G_{n+2}).x for k=0..2^n .
// For example, C_1 = {1/sqrt(2), -1/sqrt(2)}.
// We will choose the following ordering for the group line domain:
//   L_n = 1, 0, C_1, C_2, ..., C_{n-1}.
// Note that this is a domain of size 2^n, and it's missing a point: -1.
// The source of C_i under 2x^2-1 is C_{i+1} (in that order).
// Thus, the source of L_n under 2x^2-1 is *almost* L_{n+1}:
//   1, -1, C_1, C_2, ..., C_{n-1}, C_n.
// This means the ifft *almost* works as is.
//
// Line fft:
// =========
// The line fft is a bit-reversed fft on the line domain.
// At each step, we take evaluations of 2 polynomials f0, f1 on L_n and on {-1},
// This is stored in two buffers, one for L_n and one for {-1}.
// In each step we combine using f(x) = f0(2x^2 - 1) + x f1(2x^2 - 1),
// and end up with evaluations of f on L_{n+1} and on {-1}.
// Using a butterfly layer, we can get the evaluation of f on
//   1, -1, C_1, C_2, ..., C_{n-1}, C_n.
// To evaluate at 0, we note that f(0) = f0(-1).
// Thus, we can simply "swap" the evaluation of f at -1 in the resulting buffer, with the stored
// evaluation of f0(-1) in the second buffer.
// This transforms the evaluation buffer to:
//   1, 0, C_1, C_2, ..., C_{n-1}, C_n
// and stores the evaluation f(-1) instead in the second buffer.
//
// Circle fft:
// ===========
// Combine using f(x,y) = f0(x) + y f1(x), where f0,f1 are given as
// evaluations on L_{n}. Doing a butterfly will compute the evaluations on
//   (1,+0),(1,-0), (0,+1),(0,-1), (1/sqrt(2),1/sqrt(2)),(1/sqrt(2),-1/sqrt(2)),...,
// This is all the points on the circle group except (-1,0), but (1,0) is computed twice.
// We can use the evaluation of f0, f1 at -1 to compute the evaluation of f at (-1,0), and store
// it instead of the second evaluation of (1,0).
// This ordering will be called the group circle domain.

pub fn ordered_group_line_domain(log_size: u32) -> Vec<CirclePoint<BaseField>> {
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
pub fn ordered_group_circle_domain(log_size: u32) -> Vec<CirclePoint<BaseField>> {
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
    let twiddles = ordered_group_line_domain(log_size_line)
        .iter()
        .map(|p| p.x)
        .collect_vec();
    assert_eq!(twiddles.len(), 1 << log_size_line);

    let mut values = poly.coeffs.clone();
    let mut eval_at_minus1 = values[..1 << log_size_line].to_vec();
    for i in (1..log_size).rev() {
        for h_point in 0..(1 << (log_size - i - 1)) {
            for l_poly in 0..(1 << i) {
                let idx = (h_point << (i + 1)) + l_poly;
                // TODO: Get mutable references using get_many_unchecked_mut, by making butterfly
                // get an array.
                let (mut x, mut y) = (values[idx], values[idx + (1 << i)]);
                butterfly(&mut x, &mut y, twiddles[h_point << 1]);
                (values[idx], values[idx + (1 << i)]) = (x, y);
            }
        }
        // Swap the evaluation f0(-1)=f(0) with the evaluation f(-1).
        for l_poly in 0..(1 << i) {
            std::mem::swap(&mut eval_at_minus1[l_poly], &mut values[l_poly + (1 << i)]);
        }
    }

    // Last layer. Line polys to circle poly.
    let twiddles = ordered_group_line_domain(log_size_line)
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
    // Overwrite the second evaluation of (1,0) with the evaluation at (-1,0).
    values[1] = eval_at_minus1[0];

    CircleGroupEvaluation::new(group, values)
}

#[cfg(test)]
mod tests {
    use rand::rngs::StdRng;
    use rand::{Rng, SeedableRng};

    use crate::core::backend::cpu::circle::{evaluate_on_group, ordered_group_circle_domain};
    use crate::core::backend::cpu::CPUCirclePoly;
    use crate::core::fields::m31::{BaseField, P};
    use crate::core::poly::circle::Group;

    #[test]
    fn test_evaluate_on_group() {
        const LOG_SIZE: usize = 4;
        let rng = &mut StdRng::seed_from_u64(0);
        let coeffs = (0..(1 << LOG_SIZE))
            .map(|_| BaseField::from(rng.gen::<u32>() % P))
            .collect::<Vec<_>>();
        let poly_coeffs = coeffs.clone();
        let poly = CPUCirclePoly::new(poly_coeffs);
        let eval = evaluate_on_group(&poly, Group::new(LOG_SIZE as u32));
        let points = ordered_group_circle_domain(LOG_SIZE as u32);

        let expected = points
            .iter()
            .map(|p| poly.eval_at_point(*p))
            .collect::<Vec<_>>();
        assert_eq!(eval.values, expected);
    }
}
