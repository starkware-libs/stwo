use std::cmp::Reverse;
use std::iter::zip;

use itertools::{izip, zip_eq, Itertools};
use num_traits::{One, Zero};

use super::mle::Mle;
use crate::core::backend::cpu::circle::{circle_twiddles_from_line_twiddles, fft_layer_loop};
use crate::core::backend::cpu::{CpuCircleEvaluation, CpuCirclePoly};
use crate::core::backend::{ColumnOps, CpuBackend};
use crate::core::channel::Channel;
use crate::core::circle::{CirclePoint, Coset, SECURE_FIELD_CIRCLE_GEN};
use crate::core::constraints::coset_vanishing;
use crate::core::fft::ibutterfly;
use crate::core::fields::m31::BaseField;
use crate::core::fields::qm31::SecureField;
use crate::core::fields::FieldExpOps;
use crate::core::lookups::gkr::GkrOps;
use crate::core::poly::circle::{CanonicCoset, CircleEvaluation, CirclePoly, PolyOps};
use crate::core::poly::line::{LineDomain, LineEvaluation, LinePoly};
use crate::core::poly::utils::domain_line_twiddles_from_tree;
use crate::core::poly::{BitReversedOrder, NaturalOrder};

/// Univariate IOP for multilinear eval at point.
pub struct EvalAtPointProver {
    pub aggregate_column_lde_by_size: Vec<CpuCircleEvaluation<BaseField, BitReversedOrder>>,
    pub eq_lde_by_size: Vec<CpuCircleEvaluation<BaseField, BitReversedOrder>>,
    pub product_polynomials_by_size: Vec<UnivariateSumcheckPoly>,
}

impl EvalAtPointProver {
    /// # Panics
    ///
    /// Panics if:
    /// * `columns` is empty or not sorted in ascending order by domain size.
    // TODO: Should accept base field multilinear extensions.
    // TODO: Point be a vector of secure field.
    pub fn eval_at_point(
        channel: &mut impl Channel,
        columns: &[Mle<CpuBackend, BaseField>],
        point: &[BaseField],
    ) -> Self {
        assert!(!columns.is_empty());
        assert!(columns.is_sorted_by_key(|e| Reverse(e.len())), "not sorted");

        // Draw randomness for folding columns of the same size together.
        let alpha = channel.draw_felt().0 .0;

        let aggregate_column_by_size = columns
            .group_by(|a, b| a.len() == b.len())
            .map(|col_group| random_linear_combination(col_group, alpha))
            .collect_vec();

        let mut aggregate_column_lde_by_size = Vec::new();
        let mut eq_lde_by_size = Vec::new();
        let mut product_polynomials_by_size = Vec::new();

        for aggregate_column in aggregate_column_by_size {
            let log_size = aggregate_column.len().ilog2();
            let column_r = &point[point.len() - log_size as usize..];
            let eq_evals = gen_eq_evals(column_r);

            let interpolation_domain = CanonicCoset::new(log_size).circle_domain();
            let lde_domain = CanonicCoset::new(log_size + 1).circle_domain();

            let eq_evals = CpuCircleEvaluation::new(interpolation_domain, eq_evals.into_evals());
            let eq_evals_poly = eq_evals.interpolate();
            let eq_evals_lde = eq_evals_poly.evaluate(lde_domain);

            // TODO: Could use `Cow` for CircleEvaluation and CirclePoly.
            let aggregate_column_evals =
                CpuCircleEvaluation::new(interpolation_domain, aggregate_column.into_evals());
            let aggregate_column_poly = aggregate_column_evals.interpolate();
            let aggregate_column_lde = aggregate_column_poly.evaluate(lde_domain);

            // Multiply the eq and aggregate column.
            let product_column = zip_eq(&*eq_evals_lde, &*aggregate_column_lde)
                .map(|(&eq_eval, &aggregate_eval)| eq_eval * aggregate_eval)
                .collect();
            let product_interpolation_domain = lde_domain;
            let product_evals =
                CpuCircleEvaluation::new(product_interpolation_domain, product_column);

            // Obtain the product polynomial as its univariate sum-check polynomial constituents.
            let sum_coset = CanonicCoset::new(log_size);
            let product_polynomial =
                UnivariateSumcheckPoly::decompose(product_evals.interpolate(), sum_coset);

            eq_lde_by_size.push(eq_evals_lde);
            aggregate_column_lde_by_size.push(aggregate_column_lde);
            product_polynomials_by_size.push(product_polynomial);
        }

        Self {
            aggregate_column_lde_by_size,
            eq_lde_by_size,
            product_polynomials_by_size,
        }
    }
}

fn gen_eval_at_point_sumcheck_poly(
    sum_domain: CanonicCoset,
    eq_evals: CircleEvaluation<CpuBackend, BaseField, BitReversedOrder>,
    col_evals: CircleEvaluation<CpuBackend, BaseField, BitReversedOrder>,
) -> UnivariateSumcheckPoly {
    assert_eq!(eq_evals.domain, sum_domain.circle_domain());
    assert_eq!(col_evals.domain, sum_domain.circle_domain());

    let n = 1 << sum_domain.log_size();

    let g_plus_beta_evals = CircleEvaluation::<CpuBackend, BaseField, BitReversedOrder>::new(
        sum_domain.circle_domain(),
        zip_eq(&*eq_evals, &*col_evals)
            .map(|(&eq_eval, &col_eval)| eq_eval * col_eval)
            .collect(),
    );

    let lde_domain = CanonicCoset::new(sum_domain.log_size() + 1).circle_domain();
    let eq_lde = eq_evals.interpolate().evaluate(lde_domain);
    let col_lde = col_evals.interpolate().evaluate(lde_domain);

    let beta = g_plus_beta_evals.iter().sum::<BaseField>() / BaseField::from(n);
    let mut g = g_plus_beta_evals.interpolate();
    g.coeffs[0] -= beta;

    let g_lde = g.evaluate(lde_domain);

    let periodic_vanish_evals = lde_domain.repeated_double(sum_domain.log_size()).iter();
    let periodic_vanish_evals_inv = periodic_vanish_evals.map(|v| v.x.inverse()).collect_vec();
    assert_eq!(periodic_vanish_evals_inv.len(), 2);

    let h_evals = CircleEvaluation::<CpuBackend, BaseField, BitReversedOrder>::new(
        lde_domain,
        izip!(eq_lde, col_lde, g_lde)
            .enumerate()
            .map(|(i, (eq_eval, col_eval, g_eval))| {
                let vanish_eval_inv = periodic_vanish_evals_inv[i % 2];
                (eq_eval * col_eval - g_eval - beta) * vanish_eval_inv
            })
            .collect(),
    );

    let h = h_evals.interpolate();
    let mut h_coeffs = h.coeffs;
    assert!(h_coeffs[h_coeffs.len() / 2..].iter().all(|c| c.is_zero()));
    h_coeffs.truncate(h_coeffs.len() / 2);
    let h = CirclePoly::new(h_coeffs);

    let (g0, g1) = decompose_g(g);

    UnivariateSumcheckPoly {
        g0,
        g1,
        h,
        beta,
        sum_domain,
    }
}

/// Returns `g0` and `g1` given `g(x, y) = x * g0(x) + y * g1(x)`.
// TODO: Pass precomputed twiddles.
fn decompose_g(g: CirclePoly<CpuBackend>) -> (LinePoly<BaseField>, LinePoly<BaseField>) {
    assert!(g.len() > 4);

    let domain = CanonicCoset::new(g.len().ilog2()).circle_domain();
    let twiddle_tree = CpuBackend::precompute_twiddles(domain.half_coset);
    let line_twiddles = domain_line_twiddles_from_tree(domain, &twiddle_tree.twiddles);
    let circle_twiddles = circle_twiddles_from_line_twiddles(line_twiddles[0]);

    let mut xg0_coeffs = Vec::with_capacity(g.len() / 2);
    let mut g1_coeffs = Vec::with_capacity(g.len() / 2);

    for (i, twiddle) in circle_twiddles.enumerate() {
        let mut xg0_coeff = g[i * 2];
        let mut g1_coeff = g[i * 2 + 1];
        ibutterfly(&mut xg0_coeff, &mut g1_coeff, twiddle);
        xg0_coeffs.push(xg0_coeff);
        g1_coeffs.push(g1_coeff);
    }

    let xg0 = LinePoly::new(xg0_coeffs);
    // Check `g` was properly formed
    assert!(xg0.eval_at_point(BaseField::zero().into()).is_zero());
    let g0 = divide_by_x(xg0);
    let g1 = LinePoly::new(g1_coeffs);

    (g0, g1)
}

// TODO: Replace with `B::gen_eq_evals` once supporting SecureField.
fn gen_eq_evals(r: &[BaseField]) -> Mle<CpuBackend, BaseField> {
    let r = r.iter().map(|&ri| ri.into()).collect_vec();
    let eq_evals = CpuBackend::gen_eq_evals(&r, SecureField::one());
    Mle::new(eq_evals.into_evals().into_iter().map(|v| v.0 .0).collect())
}

/// # Panics
///
/// Panics if `columns` is empty or not all of the same size.
// TODO: Use SecureField for `z`.
fn random_linear_combination(
    columns: &[Mle<CpuBackend, BaseField>],
    alpha: BaseField,
) -> Mle<CpuBackend, BaseField> {
    let mut columns = columns.iter();
    let first = columns.next().unwrap().clone().into_evals();
    Mle::new(columns.fold(first, |mut acc, column| {
        zip_eq(&mut acc, &**column).for_each(|(acc_v, &col_v)| *acc_v += alpha * col_v);
        acc
    }))
}

/// Represents polynomial `f` as constituent parts needed for performing univariate sumcheck.
///
/// Let `f(x, y) = beta + g(x, y) + Z_H(x) * h(x, y)` where `g(x, y) = x * g0(x) + y * g1(x)` with
/// `deg(g0), deg(g1) < |H| - 1` and `Z_H` is the vanishing polynomial of circle domain `H`.
///
/// See https://eprint.iacr.org/2018/828.pdf (section 5)
pub struct UnivariateSumcheckPoly {
    g0: LinePoly<BaseField>,
    g1: LinePoly<BaseField>,
    h: CpuCirclePoly,
    beta: BaseField,
    sum_domain: CanonicCoset,
}

impl UnivariateSumcheckPoly {
    /// Decomposes circle polynomial `f` into the constituents required for univariate sum-check.
    pub fn decompose(f: CpuCirclePoly, sum_domain: CanonicCoset) -> Self {
        if f.len().ilog2() > sum_domain.log_size() {
            let eval_domain = CanonicCoset::new(f.log_size()).circle_domain();

            let g_plus_beta_evals = CpuCircleEvaluation::<BaseField, NaturalOrder>::new(
                sum_domain.circle_domain(),
                sum_domain
                    .circle_domain()
                    .iter()
                    .map(|p| f.eval_at_point(p.into_ef()).0 .0)
                    .collect(),
            );

            let g_plus_beta = g_plus_beta_evals.bit_reverse().interpolate();

            let mut g_plus_beta_evals = g_plus_beta.evaluate(eval_domain).values;
            CpuBackend::bit_reverse_column(&mut g_plus_beta_evals);

            let mut f_evals = f.evaluate(eval_domain).values;
            CpuBackend::bit_reverse_column(&mut f_evals);

            let h_evals = izip!(f_evals, g_plus_beta_evals, eval_domain)
                .map(|(f_eval, g_plus_beta_eval, p)| {
                    (f_eval - g_plus_beta_eval) / coset_vanishing(sum_domain.coset(), p)
                })
                .collect();

            let h = CpuCircleEvaluation::<BaseField, NaturalOrder>::new(eval_domain, h_evals)
                .bit_reverse()
                .interpolate();

            let p = SECURE_FIELD_CIRCLE_GEN;
            assert_eq!(
                g_plus_beta.eval_at_point(p)
                    + coset_vanishing(sum_domain.coset(), p) * h.eval_at_point(p),
                f.eval_at_point(p),
            );

            let (g0, g1, beta) = decompose_g_plus_beta(g_plus_beta);

            Self {
                g0,
                g1,
                h,
                beta,
                sum_domain,
            }
        } else {
            // Note `f(x, y) = beta + g(x, y) + Z_H(x) * 0`.
            let h = CpuCirclePoly::new(vec![BaseField::zero()]);
            let (g0, g1, beta) = decompose_g_plus_beta(f);

            Self {
                g0,
                g1,
                h,
                beta,
                sum_domain,
            }
        }
    }

    pub fn eval_at_point(&self, p: CirclePoint<SecureField>) -> SecureField {
        let Self {
            g0,
            g1,
            h,
            beta,
            sum_domain,
        } = self;

        let g_eval = p.x * g0.eval_at_point(p.x) + p.y * g1.eval_at_point(p.x);

        g_eval + coset_vanishing(sum_domain.coset(), p) * h.eval_at_point(p) + *beta
    }
}

// Returns `g0, g1, beta` when given polynomial `g(x, y) + beta` where
// `g(x, y) = x * g0(x) + y * g1(x)`
fn decompose_g_plus_beta(
    g_plus_beta: CpuCirclePoly,
) -> (LinePoly<BaseField>, LinePoly<BaseField>, BaseField) {
    let xg0_plus_beta_coeffs = g_plus_beta.array_chunks().map(|[v, _]| *v).collect_vec();
    let mut xg0_plus_beta_coeffs_bit_rev = xg0_plus_beta_coeffs;
    CpuBackend::bit_reverse_column(&mut xg0_plus_beta_coeffs_bit_rev);
    let xg0_plus_beta = LinePoly::new(xg0_plus_beta_coeffs_bit_rev);
    let beta = xg0_plus_beta.eval_at_point(SecureField::zero()).0 .0;
    let mut xg0 = xg0_plus_beta;
    xg0[0] -= beta;
    let g0 = divide_by_x(xg0);

    let g1_coeffs = g_plus_beta.array_chunks().map(|[_, v]| *v).collect_vec();
    let mut g1_coeffs_bit_rev = g1_coeffs;
    CpuBackend::bit_reverse_column(&mut g1_coeffs_bit_rev);
    let g1 = LinePoly::new(g1_coeffs_bit_rev);

    let p = SECURE_FIELD_CIRCLE_GEN;
    assert_eq!(
        beta + p.x * g0.eval_at_point(p.x) + p.y * g1.eval_at_point(p.x),
        g_plus_beta.eval_at_point(p)
    );

    (g0, g1, beta)
}

/// Returns `f(x) / x mod Z(x)`.
fn divide_by_x(f: LinePoly<BaseField>) -> LinePoly<BaseField> {
    let domain = LineDomain::new(Coset::half_odds(f.len().ilog2()));

    let f_div_x_evals = domain
        .iter()
        .map(|x| f.eval_at_point(x.into()) / x)
        .collect_vec();

    let mut f_div_x_evals_bit_rev = f_div_x_evals;
    CpuBackend::bit_reverse_column(&mut f_div_x_evals_bit_rev);

    let f_div_x =
        LineEvaluation::<CpuBackend>::new(domain, f_div_x_evals_bit_rev.into_iter().collect())
            .interpolate();

    LinePoly::new(f_div_x.iter().map(|v| v.0 .0).collect())
}

#[cfg(test)]
mod tests {
    use rand::rngs::SmallRng;
    use rand::{Rng, SeedableRng};

    use crate::core::backend::cpu::CpuCirclePoly;
    use crate::core::circle::{CirclePoint, SECURE_FIELD_CIRCLE_GEN};
    use crate::core::constraints::coset_vanishing;
    use crate::core::lookups::eval_at_point::UnivariateSumcheckPoly;
    use crate::core::poly::circle::CanonicCoset;

    #[test]
    fn univariate_sumcheck_eval_works() {
        const N: usize = 1 << 5;
        let point = SECURE_FIELD_CIRCLE_GEN;
        let mut rng = SmallRng::seed_from_u64(0);
        let f = CpuCirclePoly::new((0..N).map(|_| rng.gen()).collect());
        let f_eval = f.eval_at_point(point);
        let sumcheck_poly = UnivariateSumcheckPoly::decompose(f, CanonicCoset::new(6));

        let sumcheck_poly_eval = sumcheck_poly.eval_at_point(point);

        assert_eq!(sumcheck_poly_eval, f_eval);
    }

    #[test]
    fn coset_doubling() {
        let vanish_domain = CanonicCoset::new(4);
        let domain = CanonicCoset::new(5);
        for point in domain.circle_domain() {
            println!("{}", coset_vanishing(vanish_domain.coset(), point));
        }

        for point in domain.circle_domain() {
            println!("{}", coset_vanishing(vanish_domain.coset(), point));
        }
    }
}
