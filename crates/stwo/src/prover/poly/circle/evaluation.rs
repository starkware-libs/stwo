use std::marker::PhantomData;
use std::ops::{Deref, Index};

use educe::Educe;

use super::{CirclePoly, PolyOps};
use crate::core::fields::m31::BaseField;
use crate::core::fields::ExtensionOf;
use crate::core::poly::circle::CircleDomain;
use crate::prover::backend::simd::SimdBackend;
use crate::prover::backend::{Col, Column, ColumnOps, CpuBackend};
use crate::prover::poly::twiddles::TwiddleTree;
use crate::prover::poly::{BitReversedOrder, NaturalOrder};

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
    pub fn bit_reverse(mut self) -> CircleEvaluation<B, F, BitReversedOrder> {
        B::bit_reverse_column(&mut self.values);
        CircleEvaluation::new(self.domain, self.values)
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
    use crate::core::fields::m31::BaseField;
    use crate::core::poly::circle::CanonicCoset;
    use crate::m31;
    use crate::prover::backend::cpu::CpuCircleEvaluation;
    use crate::prover::poly::NaturalOrder;

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
}
