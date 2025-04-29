use std::marker::PhantomData;
use std::ops::{Deref, Index};

use educe::Educe;
use itertools::Itertools;
use num_traits::{One, Zero};

use super::{CircleDomain, CirclePoly, PolyOps};
use crate::core::backend::cpu::CpuCircleEvaluation;
use crate::core::backend::simd::m31::N_LANES;
use crate::core::backend::simd::qm31::PackedSecureField;
use crate::core::backend::simd::SimdBackend;
use crate::core::backend::{Col, Column, ColumnOps, CpuBackend};
use crate::core::circle::{CirclePoint, CirclePointIndex, Coset};
use crate::core::constraints::{coset_vanishing, coset_vanishing_derivative, point_vanishing};
use crate::core::fields::m31::BaseField;
use crate::core::fields::qm31::SecureField;
use crate::core::fields::{batch_inverse_in_place, ExtensionOf};
use crate::core::poly::circle::CanonicCoset;
use crate::core::poly::twiddles::TwiddleTree;
use crate::core::poly::{BitReversedOrder, NaturalOrder};
use crate::core::utils::bit_reverse_index;

/// An evaluation defined on a [CircleDomain].
/// The values are ordered according to the [CircleDomain] ordering.
#[derive(Educe)]
#[educe(Clone, Debug)]
pub struct CircleEvaluation<B: ColumnOps<F>, F: ExtensionOf<BaseField>, EvalOrder = NaturalOrder> {
    pub domain: CircleDomain,
    pub values: Col<B, F>,
    _eval_order: PhantomData<EvalOrder>,
}

impl<B: ColumnOps<F>, F: ExtensionOf<BaseField>, EvalOrder> CircleEvaluation<B, F, EvalOrder> {
    pub fn new(domain: CircleDomain, values: Col<B, F>) -> Self {
        assert_eq!(domain.size(), values.len());
        Self {
            domain,
            values,
            _eval_order: PhantomData,
        }
    }
}

// Note: The concrete implementation of the poly operations is in the specific backend used.
// For example, the CPU backend implementation is in `src/core/backend/cpu/poly.rs`.
// TODO(first) Remove NaturalOrder.
impl<F: ExtensionOf<BaseField>, B: ColumnOps<F>> CircleEvaluation<B, F, NaturalOrder> {
    // TODO(alont): Remove. Is this even used.
    pub fn get_at(&self, point_index: CirclePointIndex) -> F {
        self.values
            .at(self.domain.find(point_index).expect("Not in domain"))
    }

    pub fn bit_reverse(mut self) -> CircleEvaluation<B, F, BitReversedOrder> {
        B::bit_reverse_column(&mut self.values);
        CircleEvaluation::new(self.domain, self.values)
    }
}

impl<F: ExtensionOf<BaseField>> CpuCircleEvaluation<F, NaturalOrder> {
    pub fn fetch_eval_on_coset(&self, coset: Coset) -> CosetSubEvaluation<'_, F> {
        assert!(coset.log_size() <= self.domain.half_coset.log_size());
        if let Some(offset) = self.domain.half_coset.find(coset.initial_index) {
            return CosetSubEvaluation::new(
                &self.values[..self.domain.half_coset.size()],
                offset,
                coset.step_size / self.domain.half_coset.step_size,
            );
        }
        if let Some(offset) = self.domain.half_coset.conjugate().find(coset.initial_index) {
            return CosetSubEvaluation::new(
                &self.values[self.domain.half_coset.size()..],
                offset,
                (-coset.step_size) / self.domain.half_coset.step_size,
            );
        }
        panic!("Coset not found in domain");
    }
}

impl<B: PolyOps> CircleEvaluation<B, BaseField, BitReversedOrder> {
    /// Computes a minimal [CirclePoly] that evaluates to the same values as this evaluation.
    pub fn interpolate(self) -> CirclePoly<B> {
        let coset = self.domain.half_coset;
        B::interpolate(self, &B::precompute_twiddles(coset))
    }

    /// Computes a minimal [CirclePoly] that evaluates to the same values as this evaluation, using
    /// precomputed twiddles.
    pub fn interpolate_with_twiddles(self, twiddles: &TwiddleTree<B>) -> CirclePoly<B> {
        B::interpolate(self, twiddles)
    }
}

impl<B: ColumnOps<F>, F: ExtensionOf<BaseField>> CircleEvaluation<B, F, BitReversedOrder> {
    pub fn bit_reverse(mut self) -> CircleEvaluation<B, F, NaturalOrder> {
        B::bit_reverse_column(&mut self.values);
        CircleEvaluation::new(self.domain, self.values)
    }

    pub fn get_at(&self, point_index: CirclePointIndex) -> F {
        self.values.at(bit_reverse_index(
            self.domain.find(point_index).expect("Not in domain"),
            self.domain.log_size(),
        ))
    }
}

impl<F: ExtensionOf<BaseField>, EvalOrder> CircleEvaluation<SimdBackend, F, EvalOrder>
where
    SimdBackend: ColumnOps<F>,
{
    pub fn to_cpu(&self) -> CircleEvaluation<CpuBackend, F, EvalOrder> {
        CircleEvaluation::new(self.domain, self.values.to_cpu())
    }
}

impl<B: ColumnOps<F>, F: ExtensionOf<BaseField>, EvalOrder> Deref
    for CircleEvaluation<B, F, EvalOrder>
{
    type Target = Col<B, F>;

    fn deref(&self) -> &Self::Target {
        &self.values
    }
}

/// A part of a [CircleEvaluation], for a specific coset that is a subset of the circle domain.
pub struct CosetSubEvaluation<'a, F: ExtensionOf<BaseField>> {
    evaluation: &'a [F],
    offset: usize,
    step: isize,
}

impl<'a, F: ExtensionOf<BaseField>> CosetSubEvaluation<'a, F> {
    fn new(evaluation: &'a [F], offset: usize, step: isize) -> Self {
        assert!(evaluation.len().is_power_of_two());
        Self {
            evaluation,
            offset,
            step,
        }
    }
}

impl<F: ExtensionOf<BaseField>> Index<isize> for CosetSubEvaluation<'_, F> {
    type Output = F;

    fn index(&self, index: isize) -> &Self::Output {
        let index =
            ((self.offset as isize) + index * self.step) & ((self.evaluation.len() - 1) as isize);
        &self.evaluation[index as usize]
    }
}

impl<F: ExtensionOf<BaseField>> Index<usize> for CosetSubEvaluation<'_, F> {
    type Output = F;

    fn index(&self, index: usize) -> &Self::Output {
        &self[index as isize]
    }
}

// TODO(Gali): Remove.
#[allow(dead_code)]
fn weights(log_size: u32, sample_point: CirclePoint<SecureField>) -> Col<CpuBackend, SecureField> {
    let domain = CanonicCoset::new(log_size).circle_domain();

    for i in 0..domain.size() {
        if domain.at(i).into_ef() == sample_point {
            let mut weights = vec![SecureField::zero(); domain.size()];
            weights[i] = SecureField::one();
            return weights;
        }
    }

    let p_0 = domain.at(0).into_ef::<SecureField>();
    let weights_first_half = SecureField::one()
        / (-(p_0.y + p_0.y)
            * coset_vanishing_derivative(
                Coset::new(CirclePointIndex::generator(), domain.log_size()),
                p_0,
            ));
    let p_0_inverse = domain.at(domain.half_coset.size()).into_ef::<SecureField>();
    let weights_second_half = SecureField::one()
        / (-(p_0_inverse.y + p_0_inverse.y)
            * coset_vanishing_derivative(
                Coset::new(CirclePointIndex::generator(), domain.log_size()),
                p_0_inverse,
            ));

    let domain_points_vanishing_evaluated_at_point = (0..domain.size())
        .map(|i| {
            point_vanishing(
                domain.at(i).into_ef::<SecureField>(),
                sample_point.into_ef::<SecureField>(),
            )
        })
        .collect_vec();
    let mut inversed_domain_points_vanishing_evaluated_at_point =
        vec![unsafe { std::mem::zeroed() }; domain.size()];

    batch_inverse_in_place(
        &domain_points_vanishing_evaluated_at_point,
        &mut inversed_domain_points_vanishing_evaluated_at_point,
    );

    let coset_vanishing_evaluated_at_point: SecureField = coset_vanishing(
        CanonicCoset::new(domain.log_size()).coset,
        sample_point.into_ef::<SecureField>(),
    );

    (0..domain.size())
        .map(|i| {
            if i < domain.half_coset.size() {
                weights_first_half
                    * inversed_domain_points_vanishing_evaluated_at_point[i]
                    * coset_vanishing_evaluated_at_point
            } else {
                weights_second_half
                    * inversed_domain_points_vanishing_evaluated_at_point[i]
                    * coset_vanishing_evaluated_at_point
            }
        })
        .collect_vec()
}

// TODO(Gali): Remove.
#[allow(dead_code)]
fn barycentric_eval_at_point(
    evals: &CircleEvaluation<CpuBackend, BaseField, BitReversedOrder>,
    point: CirclePoint<SecureField>,
    weights: &Col<CpuBackend, SecureField>,
) -> SecureField {
    for i in 0..evals.domain.size() {
        if point == evals.domain.at(i).into_ef() {
            return evals.values[bit_reverse_index(i, evals.domain.log_size())].into();
        }
    }

    (0..evals.domain.size()).fold(SecureField::zero(), |acc, i| {
        acc + (evals.values[bit_reverse_index(i, evals.domain.log_size())] * weights[i])
    })
}

// TODO(Gali): Remove.
#[allow(dead_code)]
fn simd_weights(
    log_size: u32,
    sample_point: CirclePoint<SecureField>,
) -> Col<SimdBackend, SecureField> {
    let domain = CanonicCoset::new(log_size).circle_domain();
    let weights_vec_len = domain.size().div_ceil(N_LANES);

    for i in 0..domain.size() {
        if domain.at(i).into_ef() == sample_point {
            let mut weights = Col::<SimdBackend, SecureField>::zeros(domain.size());
            weights.set(i, SecureField::one());
            return weights;
        }
    }

    let p_0 = domain.at(0).into_ef::<SecureField>();
    let weights_first_half = SecureField::one()
        / (-(p_0.y + p_0.y)
            * coset_vanishing_derivative(
                Coset::new(CirclePointIndex::generator(), domain.log_size()),
                p_0,
            ));
    let p_0_inverse = domain.at(domain.half_coset.size()).into_ef::<SecureField>();
    let weights_second_half = SecureField::one()
        / (-(p_0_inverse.y + p_0_inverse.y)
            * coset_vanishing_derivative(
                Coset::new(CirclePointIndex::generator(), domain.log_size()),
                p_0_inverse,
            ));

    let domain_points_vanishing_evaluated_at_point = (0..weights_vec_len)
        .map(|i| {
            PackedSecureField::from_array(std::array::from_fn(|j| {
                if domain.size() <= i * N_LANES + j {
                    SecureField::one()
                } else {
                    point_vanishing(
                        domain.at(i * N_LANES + j).into_ef::<SecureField>(),
                        sample_point.into_ef::<SecureField>(),
                    )
                }
            }))
        })
        .collect_vec();
    let mut inversed_domain_points_vanishing_evaluated_at_point =
        vec![unsafe { std::mem::zeroed() }; weights_vec_len];

    batch_inverse_in_place(
        &domain_points_vanishing_evaluated_at_point,
        &mut inversed_domain_points_vanishing_evaluated_at_point,
    );

    let coset_vanishing_evaluated_at_point: SecureField = coset_vanishing(
        CanonicCoset::new(domain.log_size()).coset,
        sample_point.into_ef::<SecureField>(),
    );

    if weights_vec_len == 1 {
        return (0..N_LANES)
            .map(|i| {
                let inversed_domain_points_vanishing_evaluated_at_point =
                    inversed_domain_points_vanishing_evaluated_at_point[0].to_array();
                if i < domain.size() / 2 {
                    inversed_domain_points_vanishing_evaluated_at_point[i]
                        * (weights_first_half * coset_vanishing_evaluated_at_point)
                } else {
                    inversed_domain_points_vanishing_evaluated_at_point[i]
                        * (weights_second_half * coset_vanishing_evaluated_at_point)
                }
            })
            .collect();
    }

    let weights: Col<SimdBackend, SecureField> = (0..weights_vec_len)
        .map(|i| {
            if i < weights_vec_len / 2 {
                inversed_domain_points_vanishing_evaluated_at_point[i]
                    * (weights_first_half * coset_vanishing_evaluated_at_point)
            } else {
                inversed_domain_points_vanishing_evaluated_at_point[i]
                    * (weights_second_half * coset_vanishing_evaluated_at_point)
            }
        })
        .collect();

    weights
}

// TODO(Gali): Remove.
#[allow(dead_code)]
fn simd_barycentric_eval_at_point(
    evals: &CircleEvaluation<SimdBackend, BaseField, BitReversedOrder>,
    point: CirclePoint<SecureField>,
    weights: &Col<SimdBackend, SecureField>,
) -> SecureField {
    for i in 0..evals.domain.size() {
        if point == evals.domain.at(i).into_ef() {
            return evals
                .values
                .at(bit_reverse_index(i, evals.domain.log_size()))
                .into();
        }
    }
    let evals = evals.clone().bit_reverse();
    if evals.domain.size() < N_LANES {
        return (0..evals.domain.size()).fold(SecureField::zero(), |acc, i| {
            acc + (weights.at(i) * evals.values.at(i))
        });
    }
    (0..evals.domain.size().div_ceil(N_LANES))
        .fold(PackedSecureField::zero(), |acc, i| {
            acc + (weights.data[i] * evals.values.data[i])
        })
        .pointwise_sum()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::backend::cpu::{CpuCircleEvaluation, CpuCirclePoly};
    use crate::core::backend::simd::column::BaseColumn;
    use crate::core::backend::simd::SimdBackend;
    use crate::core::circle::{CirclePoint, Coset};
    use crate::core::fields::m31::BaseField;
    use crate::core::poly::circle::{CanonicCoset, CirclePoly};
    use crate::core::poly::NaturalOrder;
    use crate::m31;

    #[test]
    fn test_interpolate_non_canonic() {
        let domain = CanonicCoset::new(3).circle_domain();
        assert_eq!(domain.log_size(), 3);
        let evaluation = CpuCircleEvaluation::<_, NaturalOrder>::new(
            domain,
            (0..8).map(BaseField::from_u32_unchecked).collect(),
        )
        .bit_reverse();
        let poly = evaluation.interpolate();
        for (i, point) in domain.iter().enumerate() {
            assert_eq!(poly.eval_at_point(point.into_ef()), m31!(i as u32).into());
        }
    }

    #[test]
    pub fn test_get_at_circle_evaluation() {
        let domain = CanonicCoset::new(7).circle_domain();
        let values = (0..domain.size()).map(|i| m31!(i as u32)).collect();
        let circle_evaluation = CpuCircleEvaluation::<_, NaturalOrder>::new(domain, values);
        let bit_reversed_circle_evaluation = circle_evaluation.clone().bit_reverse();
        for index in domain.iter_indices() {
            assert_eq!(
                circle_evaluation.get_at(index),
                bit_reversed_circle_evaluation.get_at(index)
            );
        }
    }

    #[test]
    fn test_sub_evaluation() {
        let domain = CanonicCoset::new(7).circle_domain();
        let values = (0..domain.size()).map(|i| m31!(i as u32)).collect();
        let circle_evaluation = CpuCircleEvaluation::new(domain, values);
        let coset = Coset::new(domain.index_at(17), 3);
        let sub_eval = circle_evaluation.fetch_eval_on_coset(coset);
        for i in 0..coset.size() {
            assert_eq!(sub_eval[i], circle_evaluation.get_at(coset.index_at(i)));
        }
    }

    #[test]
    fn test_cpu_barycentric_evaluation() {
        let poly = CpuCirclePoly::new(
            [691, 805673, 5, 435684, 4832, 23876431, 197, 897346068]
                .map(BaseField::from)
                .to_vec(),
        );
        let s = CanonicCoset::new(3);
        let domain = s.circle_domain();
        let eval = poly.evaluate(domain);
        let sampled_points = [
            CirclePoint::get_point(348),
            CirclePoint::get_point(9736524),
            CirclePoint::get_point(13),
            CirclePoint::get_point(346752),
            domain.at(0).into_ef(),
            domain.at(3).into_ef(),
        ];
        let sampled_values = sampled_points
            .iter()
            .map(|point| poly.eval_at_point(*point))
            .collect::<Vec<_>>();

        let sampled_barycentric_values = sampled_points
            .iter()
            .map(|point| {
                barycentric_eval_at_point(&eval, *point, &weights(eval.domain.log_size(), *point))
            })
            .collect::<Vec<_>>();

        assert_eq!(
            sampled_barycentric_values, sampled_values,
            "Barycentric evaluation should be equal to the polynomial evaluation"
        );
    }

    #[test]
    fn test_simd_barycentric_evaluation() {
        let poly = CirclePoly::<SimdBackend>::new(BaseColumn::from_cpu(
            [691, 805673, 5, 435684, 4832, 23876431, 197, 897346068]
                .map(BaseField::from)
                .to_vec(),
        ));
        let s = CanonicCoset::new(3);
        let domain = s.circle_domain();
        let eval = poly.evaluate(domain);
        let sampled_points = [
            CirclePoint::get_point(348),
            CirclePoint::get_point(9736524),
            CirclePoint::get_point(13),
            CirclePoint::get_point(346752),
            domain.at(0).into_ef(),
            domain.at(3).into_ef(),
        ];
        let sampled_values = sampled_points
            .iter()
            .map(|point| poly.eval_at_point(*point))
            .collect::<Vec<_>>();

        let sampled_barycentric_values = sampled_points
            .iter()
            .map(|point| {
                simd_barycentric_eval_at_point(
                    &eval,
                    *point,
                    &simd_weights(eval.domain.log_size(), *point),
                )
            })
            .collect::<Vec<_>>();

        assert_eq!(
            sampled_barycentric_values, sampled_values,
            "Barycentric evaluation should be equal to the polynomial evaluation"
        );
    }
}
