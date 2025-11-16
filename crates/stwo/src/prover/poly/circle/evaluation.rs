use std::marker::PhantomData;
use std::ops::{Deref, Index};

use educe::Educe;

use super::{CircleCoefficients, PolyOps};
use crate::core::circle::CirclePoint;
use crate::core::fields::m31::BaseField;
use crate::core::fields::qm31::SecureField;
use crate::core::fields::ExtensionOf;
use crate::core::poly::circle::{CanonicCoset, CircleDomain};
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

impl<B: PolyOps + ColumnOps<SecureField>> CircleEvaluation<B, BaseField, BitReversedOrder> {
    /// Computes a minimal [CircleCoefficients] that evaluates to the same values as this
    /// evaluation.
    pub fn interpolate(self) -> CircleCoefficients<B> {
        let coset = self.domain.half_coset;
        B::interpolate(self, &B::precompute_twiddles(coset))
    }

    /// Computes a minimal [CircleCoefficients] that evaluates to the same values as this
    /// evaluation, using precomputed twiddles.
    pub fn interpolate_with_twiddles(self, twiddles: &TwiddleTree<B>) -> CircleCoefficients<B> {
        B::interpolate(self, twiddles)
    }

    /// For a canonic coset `coset` of size 2^n and a point `p` not in `coset`, the weight at a
    /// coset point i is computed as:
    ///
    /// W_i = S_i(p) / S_i(i) = V_n(p) / (-2 * V'_n(i_x) * i_y * V_i(p))
    ///
    /// using the following identities from the circle stark paper:
    ///
    /// S_i(p) = V_n(p) / V_i(p)
    /// S_i(i) = -2 * V'(i_x) * i_y
    ///
    /// where:
    /// - S_i(point) is the vanishing polynomial on the coset except i, evaluated at a point.
    /// - V_n(p) is the vanishing polynomial on the coset, evaluated at p.
    /// - V_i(p) is the vanishing polynomial on point i, evaluated at p.
    /// - V'(i_x) is the derivative of V(i) (evaluated at that point), see
    ///   [`coset_vanishing_derivative`].
    pub fn barycentric_weights(
        coset: CanonicCoset,
        p: CirclePoint<SecureField>,
    ) -> Col<B, SecureField> {
        B::barycentric_weights(coset, p)
    }

    /// Evaluation = Σ W_i * Poly(i) for all i in the evaluation domain.
    /// For more information on barycentric weights calculation see [`barycentric_weights`].
    pub fn barycentric_eval_at_point(&self, weights: &Col<B, SecureField>) -> SecureField {
        B::barycentric_eval_at_point(self, weights)
    }

    pub fn eval_at_point_by_folding(
        &self,
        point: CirclePoint<SecureField>,
        twiddles: &TwiddleTree<B>,
    ) -> SecureField {
        // The evaluation by folding is done by running the FRI algorithm on the given polynomial's
        // evaluations, with folding alphas computed from the point instead of drawn from the
        // channel.
        // The folding alphas are corresponding to the FRI basis: y, x, pi(x), pi^2(x), ...
        // such that by arriving at the last layer, we get the value of the polynomial at the point.
        // Note: This function is slower than `eval_at_point` and is not fully optimized
        // (theoretically can be as fast as `eval_at_point`). Consider using barycentric
        // evaluation for better performance.
        B::eval_at_point_by_folding(self, point, twiddles)
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
