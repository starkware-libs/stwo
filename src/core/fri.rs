use std::iter::zip;

use super::fields::m31::BaseField;

/// Performs a degree respecting projection (DRP) of line evaluations.
///
/// i.e. when our evaluation domain is `E = c + <G>, |E| = 8` and
/// `Φ(x) = 2x^2 - 1` is the circle's x-coordinate doubling map:
///
/// ```text
/// 1. interpolate evals over the evaluation domain to obtain coefficients of f(x):
///    ┌─────────┬───┬───┬───┬───┬───┬───┬───┬───┐
///    │ i       │ 0 │ 1 │ 2 │ 3 │ 4 │ 5 │ 6 │ 7 │
///    ├─────────┼───┼───┼───┼───┼───┼───┼───┼───┤
///    │ eval[i] │ 9 │ 2 │ 3 │ 5 │ 9 │ 2 │ 3 │ 5 │
///    └─────────┴───┴───┴───┴───┴───┴───┴───┴───┘
///    ┌────────┬───────┬───────┬───────┬───────┬───────┬───────┬───────┬───────┐
///    │ p      │ c+0*G │ c+1*G │ c+2*G │ c+3*G │ c+4*G │ c+5*G │ c+6*G │ c+7*G │
///    ├────────┼───────┼───────┼───────┼───────┼───────┼───────┼───────┼───────┤
///    │ f(p.x) │ 9     │ 2     │ 3     │ 5     │ 9     │ 2     │ 3     │ 5     │
///    └────────┴───────┴───────┴───────┴───────┴───────┴───────┴───────┴───────┘
///      f(x) = c0 +
///             c1 * x +
///             c2 * Φ(x) +
///             c3 * Φ(x)*x +
///             c4 * Φ(Φ(x)) +
///             c5 * Φ(Φ(x))*x +
///             c6 * Φ(Φ(x))*Φ(x) +
///             c7 * Φ(Φ(x))*Φ(x)*x
///
/// 2. perform a random linear combination of odd and even coefficients of f(x):
///    f_e(x)  = c0 + c2 * x + c4 * Φ(x) + c6 * Φ(x)*x
///    f_o(x)  = c1 + c3 * x + c5 * Φ(x) + c7 * Φ(x)*x
///    f(x)    = f_e(Φ(x)) + x * f_o(Φ(x))
///    f'(x)   = f_e(x) + α * f_o(x)
///    deg(f') ≤ deg(f) / 2
///    α       = <random field element sent from verifier>
///
/// 4. obtain the DRP by evaluating f'(x) over a new domain of half the size:
///    ┌─────────┬───────────┬───────────┬───────────┬───────────┐
///    │ p       │ 2*(c+0*G) │ 2*(c+1*G) │ 2*(c+2*G) │ 2*(c+3*G) │
///    ├─────────┼───────────┼───────────┼───────────┼───────────┤
///    │ f'(p.x) │ 82        │ 12        │ 57        │ 34        │
///    └─────────┴───────────┴───────────┴───────────┴───────────┘
///    ┌────────┬────┬────┬────┬────┐
///    │ i      │ 0  │ 1  │ 2  │ 3  │
///    ├────────┼────┼────┼────┼────┤
///    │ drp[i] │ 82 │ 12 │ 57 │ 34 │
///    └────────┴────┴────┴────┴────┘
/// ```
/// Evaluations should be in their natural order.
// TODO: tests
// TODO: alpha from extension field
// TODO: create LineEvaluation
fn drp(evals: LineEvaluation, alpha: BaseField) -> LineEvaluation {
    // TODO: support different folding factors and handle bit-reversed evals
    // TODO: use 2-local transformation instead of full interpolation and evaluation
    let coeffs = evals.interpolate();
    let [even_coeffs, odd_coeffs] = coeffs.split_at(evals.len() / 2);
    let drp_coeffs = zip(even_coeffs, odd_coeffs)
        .map(|(&e, &o)| e + alpha * o)
        .collect::<Vec<BaseField>>();
    // TODO: create LinePoly
    let drp_poly = LinePoly::new(drp_coeffs);
    let drp_domain = evals.domain.double();
    drp_poly.eval(drp_domain)
}
