use std::marker::PhantomData;
use std::ops::{Deref, Index};

use educe::Educe;

use super::{CanonicCoset, CircleDomain, CirclePoly, PolyOps};
use crate::core::backend::cpu::CpuCircleEvaluation;
use crate::core::backend::simd::SimdBackend;
use crate::core::backend::{Col, Column, ColumnOps, CpuBackend};
use crate::core::circle::{CirclePointIndex, Coset};
use crate::core::fields::m31::BaseField;
use crate::core::fields::ExtensionOf;
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
    /// Creates a [CircleEvaluation] from values ordered according to
    /// [CanonicCoset]. For example, the canonic coset might look like this:
    ///   G_8, G_8 + G_4, G_8 + 2G_4, G_8 + 3G_4.
    /// The circle domain will be ordered like this:
    ///   G_8, G_8 + 2G_4, -G_8, -G_8 - 2G_4.
    pub fn new_canonical_ordered(coset: CanonicCoset, values: Col<B, BaseField>) -> Self {
        B::new_canonical_ordered(coset, values)
    }

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

#[cfg(test)]
mod tests {
    use crate::core::backend::cpu::CpuCircleEvaluation;
    use crate::core::circle::Coset;
    use crate::core::fields::m31::BaseField;
    use crate::core::poly::circle::CanonicCoset;
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
    fn test_interpolate_canonic() {
        let coset = CanonicCoset::new(3);
        let evaluation = CpuCircleEvaluation::new_canonical_ordered(
            coset,
            (0..8).map(BaseField::from_u32_unchecked).collect(),
        );
        let poly = evaluation.interpolate();
        for (i, point) in Coset::odds(3).iter().enumerate() {
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
}
