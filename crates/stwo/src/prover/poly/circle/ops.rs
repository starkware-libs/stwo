#[cfg(feature = "parallel")]
use rayon::prelude::*;

use super::{CircleCoefficients, CircleEvaluation};
use crate::core::circle::{CirclePoint, Coset};
use crate::core::fields::m31::BaseField;
use crate::core::fields::qm31::SecureField;
use crate::core::poly::circle::{CanonicCoset, CircleDomain};
use crate::core::ColumnVec;
use crate::prover::air::component_prover::Poly;
use crate::prover::backend::{Col, ColumnOps};
use crate::prover::mempool::BaseColumnPool;
use crate::prover::poly::twiddles::{TwiddleBuffer, TwiddleTree};
use crate::prover::poly::BitReversedOrder;

/// Operations on BaseField polynomials.
pub trait PolyOps: ColumnOps<BaseField> + ColumnOps<SecureField> + Sized {
    // TODO(alont): Use a column instead of this type.
    /// The type for precomputed twiddles.
    type Twiddles: TwiddleBuffer<BitReversedOrder>;

    // Return the number of threads that the FFT can use, might be higher than the number of threads
    // available.
    fn fft_thread_usage(log_size: u32) -> usize;

    /// Computes a minimal [CircleCoefficients] that evaluates to the same values as this
    /// evaluation. Used by the [`CircleEvaluation::interpolate()`] function.
    fn interpolate(
        eval: CircleEvaluation<Self, BaseField, BitReversedOrder>,
        itwiddles: &TwiddleTree<Self>,
    ) -> CircleCoefficients<Self>;

    fn interpolate_columns(
        columns: Vec<CircleEvaluation<Self, BaseField, BitReversedOrder>>,
        twiddles: &TwiddleTree<Self>,
    ) -> Vec<CircleCoefficients<Self>> {
        #[cfg(feature = "parallel")]
        let iter = columns.into_par_iter();
        #[cfg(not(feature = "parallel"))]
        let iter = columns.into_iter();

        iter.map(|eval| eval.interpolate_with_twiddles(twiddles))
            .collect()
    }

    /// Evaluates the polynomial at a single point.
    /// Used by the [`CircleCoefficients::eval_at_point()`] function.
    fn eval_at_point(
        poly: &CircleCoefficients<Self>,
        point: CirclePoint<SecureField>,
    ) -> SecureField;

    /// Computes the weights for Barycentric Lagrange interpolation for point `p` on `coset`.
    /// `p` must not be in the domain.
    /// Used by the [`CircleEvaluation::barycentric_weights()`] function.
    fn barycentric_weights(
        coset: CanonicCoset,
        p: CirclePoint<SecureField>,
    ) -> Col<Self, SecureField>;

    /// Evaluates a polynomial at a point using the barycentric interpolation formula,
    /// given its evaluations on a circle domain and precomputed barycentric weights for the domain
    /// at the sampled point.
    /// Used by the [`CircleEvaluation::barycentric_eval_at_point()`] function.
    fn barycentric_eval_at_point(
        evals: &CircleEvaluation<Self, BaseField, BitReversedOrder>,
        weights: &Col<Self, SecureField>,
    ) -> SecureField;

    /// Evaluates a polynomial, represented by it's evaluations, at a point using folding.
    /// Used by the [`CircleEvaluation::eval_at_point_by_folding()`] function.
    fn eval_at_point_by_folding(
        evals: &CircleEvaluation<Self, BaseField, BitReversedOrder>,
        point: CirclePoint<SecureField>,
        twiddles: &TwiddleTree<Self>,
    ) -> SecureField;

    /// Extends the polynomial to a larger degree bound.
    /// Used by the [`CircleCoefficients::extend()`] function.
    fn extend(poly: &CircleCoefficients<Self>, log_size: u32) -> CircleCoefficients<Self>;

    /// Evaluates the polynomial at all points in the domain.
    /// Used by the [`CircleCoefficients::evaluate()`] function.
    fn evaluate(
        poly: &CircleCoefficients<Self>,
        domain: CircleDomain,
        twiddles: &TwiddleTree<Self>,
    ) -> CircleEvaluation<Self, BaseField, BitReversedOrder>;

    /// Evaluates the polynomial at all points in the domain, writing results into the provided
    /// buffer instead of allocating a new one. The buffer must have size `domain.size()`.
    fn evaluate_into(
        poly: &CircleCoefficients<Self>,
        domain: CircleDomain,
        twiddles: &TwiddleTree<Self>,
        buffer: Col<Self, BaseField>,
    ) -> CircleEvaluation<Self, BaseField, BitReversedOrder>;

    fn evaluate_polynomials(
        polynomials: ColumnVec<CircleCoefficients<Self>>,
        log_blowup_factor: u32,
        twiddles: &TwiddleTree<Self>,
        store_polynomials_coefficients: bool,
        pool: &BaseColumnPool<Self>,
    ) -> Vec<Poly<Self>>
    where
        Self: crate::prover::backend::Backend,
    {
        // Pre-take all buffers from the pool before the parallel section.
        let buffers: Vec<_> = polynomials
            .iter()
            .map(|poly_coeffs| {
                let log_eval_size = poly_coeffs.log_size() + log_blowup_factor;
                pool.take_or_alloc(log_eval_size)
            })
            .collect();

        let evaluate = |poly_coeffs: CircleCoefficients<Self>, buffer| {
            let domain =
                CanonicCoset::new(poly_coeffs.log_size() + log_blowup_factor).circle_domain();
            let evals = Self::evaluate_into(&poly_coeffs, domain, twiddles, buffer);
            Poly::new(store_polynomials_coefficients.then_some(poly_coeffs), evals)
        };

        #[cfg(feature = "parallel")]
        {
            // Sort indices small-to-large so small FFTs get spawned first (filling the thread
            // pool), then large FFTs that saturate all threads run inline.
            let mut sorted_indices: Vec<usize> = (0..polynomials.len()).collect();
            sorted_indices.sort_by_key(|&i| polynomials[i].log_size());

            let mut polys: Vec<Option<CircleCoefficients<Self>>> =
                polynomials.into_iter().map(Some).collect();
            let mut bufs: Vec<Option<Col<Self, BaseField>>> =
                buffers.into_iter().map(Some).collect();
            let mut result: Vec<Option<Poly<Self>>> =
                (0..polys.len()).map(|_| None).collect();

            let num_threads = rayon::current_num_threads();
            // SAFETY: We use UnsafeCell to allow mutable access from spawned tasks.
            // Each index is visited exactly once, so there are no data races.
            // The cell lives on the stack for the entire rayon::scope.
            let result_cell = std::cell::UnsafeCell::new(&mut result);
            struct SendSyncCell<'a, T>(&'a std::cell::UnsafeCell<T>);
            unsafe impl<T> Send for SendSyncCell<'_, T> {}
            unsafe impl<T> Sync for SendSyncCell<'_, T> {}
            let result_cell = SendSyncCell(&result_cell);

            rayon::scope(|s| {
                for i in sorted_indices {
                    let poly_coeffs = polys[i].take().unwrap();
                    let buffer = bufs[i].take().unwrap();
                    if Self::fft_thread_usage(poly_coeffs.log_size()) >= num_threads {
                        (*unsafe { &mut *result_cell.0.get() })[i] =
                            Some(evaluate(poly_coeffs, buffer));
                    } else {
                        let result_cell = &result_cell;
                        s.spawn(move |_| {
                            (*unsafe { &mut *result_cell.0.get() })[i] =
                                Some(evaluate(poly_coeffs, buffer));
                        });
                    }
                }
            });
            result.into_iter().map(|r| r.unwrap()).collect()
        }
        #[cfg(not(feature = "parallel"))]
        {
            polynomials
                .into_iter()
                .zip(buffers)
                .map(|(poly_coeffs, buffer)| evaluate(poly_coeffs, buffer))
                .collect()
        }
    }

    /// Precomputes twiddles for a given coset.
    fn precompute_twiddles(coset: Coset) -> TwiddleTree<Self>;

    /// Given a polynomial `p`, it outputs two polynomials `p_left`, `p_right` of half the degree,
    /// which satisfy the identity
    ///
    /// `p(z) = p_left(z) + pi^{L-2}(z.x) * p_right(z)`.
    ///
    /// where `L` is the log size of the coefficient vector and `z` is a circle point.
    /// If a polynomial is given by its vector of coefficients (in terms of the FFT basis in natural
    /// order), this decomposition corresponds exactly to dividing the coefficient vector in the
    /// middle. In fact, for `n` in `[0, 2^L)`, the basis element corresponding to the n-th
    /// coefficient is
    ///
    /// `(pi^{L-2}(x))^b_{L-1} * ... * (pi(x))^b_2 * x^b_1* y^b_0`,
    ///
    /// where `b_{L-1}, ... , b_0` is the bit decomposition of n (from most to least significant
    /// bit). Therefore, splitting the coefficient vector in the middle, corresponds to separating
    /// the ones with the MSB, b_{L-1} == 1, from the ones with the MSB, b_{L-1} == 0, meaning
    /// separating the basis elements divisible by `pi^{L-2}(x)` from those that are not.
    fn split_at_mid(
        poly: CircleCoefficients<Self>,
    ) -> (CircleCoefficients<Self>, CircleCoefficients<Self>);
}
