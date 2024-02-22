use std::borrow::Cow;
use std::iter::zip;

use num_traits::{One, Zero};
use prover_research::core::channel::Channel;
use prover_research::core::fields::m31::{BaseField, P};
use prover_research::core::fields::qm31::SecureField;
use prover_research::core::fields::Field;

use crate::multivariate::{self, MultivariatePolynomial};
use crate::sumcheck::{self, MultiLinearExtension, SumcheckOracle, SumcheckProof};
use crate::utils::{Fraction, Polynomial};

pub struct Layer<'a, 'b> {
    p: &'a dyn MultivariatePolynomial,
    q: &'b dyn MultivariatePolynomial,
}

impl<'a, 'b> Layer<'a, 'b> {
    pub fn new(p: &'a dyn MultivariatePolynomial, q: &'b dyn MultivariatePolynomial) -> Self {
        assert_eq!(p.num_variables(), q.num_variables());
        Self { p, q }
    }
}

pub struct MleLayer {
    num_variables: usize,
    p: MultiLinearExtension<SecureField>,
    q: MultiLinearExtension<SecureField>,
}

impl MleLayer {
    pub fn new<F: Field + Into<SecureField>>(f: &Vec<Fraction<F>>) -> Self {
        assert!(f.len().is_power_of_two());
        let (p, q) = f
            .iter()
            .map(
                |&Fraction {
                     numerator,
                     denominator,
                 }| (numerator.into(), denominator.into()),
            )
            .unzip();

        Self {
            num_variables: f.len().ilog2() as usize,
            p: MultiLinearExtension::new(p),
            q: MultiLinearExtension::new(q),
        }
    }
}

/// Evaluates the lagrange kernel of the boolean hypercube.
///
/// When y is an elements of `{+-1}^n` then this function evaluates the Lagrange polynomial which is
/// the unique multilinear polynomial equal to 1 if `x = y` and equal to 0 whenever x is an element
/// of `{+-1}^n`.
///
/// From: <https://eprint.iacr.org/2023/1284.pdf>.
pub fn eq(x_assignments: &[SecureField], y_assignments: &[SecureField]) -> SecureField {
    assert_eq!(x_assignments.len(), y_assignments.len());
    zip(x_assignments, y_assignments)
        .map(|(&xi, &wi)| xi * wi + (SecureField::one() - xi) * (SecureField::one() - wi))
        .product::<SecureField>()
}

//

// <https://people.cs.georgetown.edu/jthaler/ProofsArgsAndZK.pdf> (page 65)
pub fn prove(channel: &mut impl Channel, layers: &[Layer]) -> GkrProof {
    let Layer { p: p1, q: q1 } = &layers[0];

    let zero = SecureField::zero();
    let one = SecureField::one();

    let p1_eval_encoding = {
        let [x0, x1] = [zero, one];
        let [y0, y1] = [p1.eval(&[x0]), p1.eval(&[x1])];
        Polynomial::interpolate_lagrange(&[x0, x1], &[y0, y1])
    };

    let q1_eval_encoding = {
        let [x0, x1] = [zero, one];
        let [y0, y1] = [q1.eval(&[x0]), q1.eval(&[x1])];
        Polynomial::interpolate_lagrange(&[x0, x1], &[y0, y1])
    };

    channel.mix_felts(&p1_eval_encoding);
    channel.mix_felts(&q1_eval_encoding);

    let r0 = channel.draw_felt();

    let layer_proofs = layers
        .array_windows()
        .scan(vec![r0], |r, [prev_layer, Layer { p, q }]| {
            let lambda = channel.draw_felt();

            let g = multivariate::from_fn(prev_layer.p.num_variables(), |y| {
                let y_zero = [y, &[zero]].concat();
                let y_one = [y, &[one]].concat();

                let v0 = Fraction::new(p.eval(&y_zero), q.eval(&y_zero));
                let v1 = Fraction::new(p.eval(&y_one), q.eval(&y_one));
                let res = v0 + v1;

                eq(r, y) * (res.numerator + lambda * res.denominator)
            });

            println!("lambda: {}", lambda);

            let (sumcheck_proof, eval_point) = sumcheck::prove(&g, channel);

            println!("eval point: {:?}", eval_point);

            // b* and c*
            let b_star = [&*eval_point, &[zero]].concat();
            let c_star = [&*eval_point, &[one]].concat();

            let p_eval_encoding =
                Polynomial::interpolate_lagrange(&[zero, one], &[p.eval(&b_star), p.eval(&c_star)]);

            let q_eval_encoding =
                Polynomial::interpolate_lagrange(&[zero, one], &[q.eval(&b_star), q.eval(&c_star)]);

            channel.mix_felts(&p_eval_encoding);
            channel.mix_felts(&q_eval_encoding);

            let r_star = channel.draw_felt();
            *r = [&*eval_point, &[r_star]].concat();

            Some(GkrLayerProof {
                sumcheck_proof,
                p_eval_encoding,
                q_eval_encoding,
            })
        })
        .collect();

    GkrProof {
        layer_proofs,
        p1_eval_encoding,
        q1_eval_encoding,
    }
}

/// Evaluates the canonic line `l(x)` such that `l(0) = b` and `l(1) = c` at `x`.
fn eval_canonic_line(b: &[SecureField], c: &[SecureField], x: SecureField) -> Vec<SecureField> {
    assert_eq!(b.len(), c.len());
    zip(b, c).map(|(&bi, &ci)| bi + x * (ci - bi)).collect()
}

/// Partially verifies a GKR proof.
///
/// Returns the variable assignment and claimed evaluation in the top layer. This claim is left to
/// the verifier to check - hence partial verification.
///
/// Output of the form `(variable_assignment, p_claimed_eval, q_claimed_eval)`.
pub fn partially_verify(
    proof: &GkrProof,
    channel: &mut impl Channel,
) -> Option<(Vec<SecureField>, SecureField, SecureField)> {
    let zero = SecureField::zero();
    let one = SecureField::one();

    let GkrProof {
        p1_eval_encoding,
        q1_eval_encoding,
        layer_proofs,
    } = proof;

    let v0 = Fraction {
        numerator: p1_eval_encoding.eval(zero),
        denominator: q1_eval_encoding.eval(zero),
    };

    let v1 = Fraction {
        numerator: p1_eval_encoding.eval(one),
        denominator: q1_eval_encoding.eval(one),
    };

    println!("made it yo");

    if !(v0 + v1).is_zero() {
        return None;
    }

    channel.mix_felts(p1_eval_encoding);
    channel.mix_felts(q1_eval_encoding);

    let r0 = channel.draw_felt();
    let p_m0 = p1_eval_encoding.eval(r0);
    let q_m0 = q1_eval_encoding.eval(r0);
    println!("made it yo1");

    layer_proofs
        .iter()
        .try_fold((vec![r0], p_m0, q_m0), |(r, p_m, q_m), layer_proof| {
            let GkrLayerProof {
                sumcheck_proof,
                p_eval_encoding,
                q_eval_encoding,
            } = layer_proof;
            println!("made it yo2");

            let lambda = channel.draw_felt();

            // The sumcheck claim.
            let m = p_m + lambda * q_m;

            let (eval_point, eval) = sumcheck::partially_verify(m, sumcheck_proof, channel)?;

            println!("OY HERE::: EXPECTED EVAL: {}", eval);

            let v0 = Fraction {
                numerator: p_eval_encoding.eval(zero),
                denominator: q_eval_encoding.eval(zero),
            };

            let v1 = Fraction {
                numerator: p_eval_encoding.eval(one),
                denominator: q_eval_encoding.eval(one),
            };

            let res = v0 + v1;

            if eval != eq(&r, &eval_point) * (res.numerator + lambda * res.denominator) {
                return None;
            }

            channel.mix_felts(p_eval_encoding);
            channel.mix_felts(q_eval_encoding);

            let b_star = [&*eval_point, &[zero]].concat();
            let c_star = [&*eval_point, &[one]].concat();
            let r_star = channel.draw_felt();

            Some((
                [&*eval_point, &[r_star]].concat(),
                p_eval_encoding.eval(r_star),
                q_eval_encoding.eval(r_star),
            ))
        })
}

pub struct GkrProof {
    layer_proofs: Vec<GkrLayerProof>,
    p1_eval_encoding: Polynomial<SecureField>,
    q1_eval_encoding: Polynomial<SecureField>,
}

struct GkrLayerProof {
    sumcheck_proof: SumcheckProof,
    p_eval_encoding: Polynomial<SecureField>,
    q_eval_encoding: Polynomial<SecureField>,
}

struct Oracle<F: Field> {
    /// p_{i + 1}
    // TODO: Consider `Cow<Vec<F>>`
    p: Vec<F>,
    /// q_{i + 1}
    q: Vec<F>,
    // TODO: docs.
    c: Vec<F>,
    num_variables: u32,
    z: Vec<F>,
    lambda: F,
}

impl Oracle<SecureField> {
    fn new(z: &[SecureField], p: &[SecureField], q: &[SecureField], lambda: SecureField) -> Self {
        let num_variables = z.len() as u32;
        assert_eq!(p.len(), 2 << num_variables);
        assert_eq!(q.len(), 2 << num_variables);
        Self {
            p: p.to_vec(),
            q: q.to_vec(),
            num_variables,
            c: c0(z),
            z: z.to_vec(),
            lambda,
        }
    }
}

impl SumcheckOracle for Oracle<SecureField> {
    type NextRoundOracle = Self;

    fn num_variables(&self) -> u32 {
        self.num_variables
    }

    fn univariate_sum(&self) -> Polynomial<SecureField> {
        let eval_at_t = |t: SecureField| {
            let zero = SecureField::zero();
            let one = SecureField::one();

            let t_bar = one - t;
            let z = self.z[0];
            let z_bar = one - z;
            let z_bar_inv = z_bar.inverse();

            let n_terms = self.c.len() / 2;
            let c_vals = &self.c[0..n_terms];
            let (p_lhs_pairs, p_rhs_pairs) = self.p.as_chunks().0.split_at(n_terms);
            let (q_lhs_pairs, q_rhs_pairs) = self.q.as_chunks().0.split_at(n_terms);

            zip(
                c_vals,
                zip(zip(p_lhs_pairs, p_rhs_pairs), zip(q_lhs_pairs, q_rhs_pairs)),
            )
            .fold(
                SecureField::zero(),
                |acc,
                 (
                    &c,
                    (
                        (&[p0_lhs, p1_lhs], &[p0_rhs, p1_rhs]),
                        (&[q0_lhs, q1_lhs], &[q0_rhs, q1_rhs]),
                    ),
                )| {
                    // TODO: `z_bar_inv * (t * z + t_bar * z_bar)` is a constant. Compute outside.
                    let eq_eval = c * z_bar_inv * (t * z + t_bar * z_bar);
                    let p0_eval = p0_lhs * t_bar + p0_rhs * t;
                    let p1_eval = p1_lhs * t_bar + p1_rhs * t;
                    let q0_eval = q0_lhs * t_bar + q0_rhs * t;
                    let q1_eval = q1_lhs * t_bar + q1_rhs * t;

                    let a = Fraction::new(p0_eval, q0_eval);
                    let b = Fraction::new(p1_eval, q1_eval);
                    let c = a + b;

                    acc + eq_eval * (c.numerator + self.lambda * c.denominator)
                },
            )
        };

        let x0 = SecureField::zero();
        let x1 = SecureField::one();
        let x2 = BaseField::from(2).into();
        let x3 = BaseField::from(3).into();
        let x4 = BaseField::from(4).into();

        let y0 = eval_at_t(x0);
        let y1 = eval_at_t(x1);
        let y2 = eval_at_t(x2);
        let y3 = eval_at_t(x3);
        let y4 = eval_at_t(x4);

        println!("eval at 0: {y0}");
        println!("eval at 1: {y1}");

        let poly = Polynomial::interpolate_lagrange(&[x0, x1, x2, x3, x4], &[y0, y1, y2, y3, y4]);

        println!("degree YO is: {}", poly.degree());

        poly
    }

    fn fix_first(self, challenge: SecureField) -> Self {
        let c = collapse_c(self.c, self.z[0], challenge);

        let collapse_p_or_q = |v: Vec<SecureField>| {
            let one = SecureField::one();
            let (v_lhs, v_rhs) = v.split_at(v.len() / 2);
            zip(v_lhs, v_rhs)
                .map(|(&lhs, &rhs)| lhs * (one - challenge) + rhs * challenge)
                .collect()
        };

        let p = collapse_p_or_q(self.p);
        let q = collapse_p_or_q(self.q);

        Self {
            p,
            q,
            c,
            num_variables: self.num_variables - 1,
            z: self.z[1..].to_vec(),
            lambda: self.lambda,
        }
    }
}

/// Computes all TODO in `O(2^|z|)`
///
/// Source: <https://eprint.iacr.org/2013/351.pdf> (Section 5.4.1)
fn c0(z: &[SecureField]) -> Vec<SecureField> {
    match z {
        [] => vec![],
        &[z1] => vec![SecureField::one() - z1, z1],
        &[zj, ref z @ ..] => {
            let c = c0(z);
            let zj_bar = SecureField::one() - zj;
            let lhs = c.iter().map(|&v| zj_bar * v);
            let rhs = c.iter().map(|&v| zj * v);
            Iterator::chain(lhs, rhs).collect()
        }
    }
}

/// Source: <https://eprint.iacr.org/2013/351.pdf> (Section 5.4.1)
fn collapse_c(mut c: Vec<SecureField>, z: SecureField, r: SecureField) -> Vec<SecureField> {
    // TODO: `z` can be zero! Just divide out (1 - z) instead and take even values of `c`. Don't
    // want to implement this just noting here.
    assert!(!z.is_zero());

    let z_bar = SecureField::one() - z;
    let r_bar = SecureField::one() - r;
    // TODO: Shift not right word.
    let shift = z_bar.inverse() * (r * z + r_bar * z_bar);

    c.truncate(c.len() / 2);
    c.iter_mut().for_each(|v| *v *= shift);

    // for i in 0..c.len() / 2 {
    //     c[i] = shift * c[i * 2 + 1];
    // }

    c
}

// <https://people.cs.georgetown.edu/jthaler/ProofsArgsAndZK.pdf> (page 65)
pub fn prove2(channel: &mut (impl Channel + Clone), layers: &[MleLayer]) -> GkrProof {
    let MleLayer { p: p1, q: q1, .. } = &layers[0];

    let zero = SecureField::zero();
    let one = SecureField::one();

    let p1_eval_encoding = {
        let [x0, x1] = [zero, one];
        let [y0, y1] = [p1[0], p1[1]];
        Polynomial::interpolate_lagrange(&[x0, x1], &[y0, y1])
    };

    let q1_eval_encoding = {
        let [x0, x1] = [zero, one];
        let [y0, y1] = [q1[0], q1[1]];
        Polynomial::interpolate_lagrange(&[x0, x1], &[y0, y1])
    };

    channel.mix_felts(&p1_eval_encoding);
    channel.mix_felts(&q1_eval_encoding);

    let r0 = channel.draw_felt();

    let layer_proofs = layers
        .array_windows()
        .scan(vec![r0], |r, [prev_layer, MleLayer { p, q, .. }]| {
            let lambda = channel.draw_felt();
            println!("lambda: {}", lambda);

            let oracle = Oracle::new(r, p, q, lambda);

            let mut chanel_clone = channel.clone();
            let (sumcheck_proof, eval_point, oracle) = sumcheck::prove3(oracle, channel);

            println!("eval point: {:?}", eval_point);

            let claim = sumcheck_proof.round_polynomials[0].eval(SecureField::zero())
                + sumcheck_proof.round_polynomials[0].eval(SecureField::one());
            let (eval_point, eval) =
                sumcheck::partially_verify(claim, &sumcheck_proof, &mut chanel_clone).unwrap();
            println!("verify res: YES",);

            // b* and c*
            let b_star = [&*eval_point, &[zero]].concat();
            let c_star = [&*eval_point, &[one]].concat();

            let p_eval_encoding = {
                let [x0, x1] = [SecureField::zero(), SecureField::one()];
                // TODO: Note, can eval both evals efficiently in single pass.
                let [y0, y1] = [p.eval(&b_star), p.eval(&c_star)];
                Polynomial::interpolate_lagrange(&[x0, x1], &[y0, y1])
            };

            let q_eval_encoding = {
                let [x0, x1] = [SecureField::zero(), SecureField::one()];
                // TODO: Note, can eval both evals efficiently in single pass.
                let [y0, y1] = [q.eval(&b_star), q.eval(&c_star)];
                Polynomial::interpolate_lagrange(&[x0, x1], &[y0, y1])
            };

            {
                println!("- claim is: {}", claim);
                let p0 = p_eval_encoding.eval(SecureField::zero());
                println!("yo (p0): {}", oracle.p[0]);
                println!("yo (p0): {}", p0);
                let p1 = p_eval_encoding.eval(SecureField::one());
                println!("yo (p1): {}", oracle.p[1]);
                println!("yo (p1): {}", p1);
                let q0 = q_eval_encoding.eval(SecureField::zero());
                println!("yo (q0): {}", oracle.q[0]);
                println!("yo (q0): {}", q0);
                let q1 = q_eval_encoding.eval(SecureField::one());
                println!("yo (q1): {}", oracle.q[1]);
                println!("yo (q1): {}", q1);
                let a = Fraction::new(p0, q0);
                let b = Fraction::new(p1, q1);
                println!("my EQ: {}", oracle.c[0]);
                println!("my EQ: {}", eq(r, &eval_point));
                let c = a + b;
                println!("- expected: {}", eval);
                println!(
                    "- actual: {}",
                    eq(r, &eval_point) * (c.numerator + lambda * c.denominator)
                );
            }

            channel.mix_felts(&p_eval_encoding);
            channel.mix_felts(&q_eval_encoding);

            let r_star = channel.draw_felt();
            *r = [&*eval_point, &[r_star]].concat();

            Some(GkrLayerProof {
                sumcheck_proof,
                p_eval_encoding,
                q_eval_encoding,
            })
        })
        .collect();

    GkrProof {
        layer_proofs,
        p1_eval_encoding,
        q1_eval_encoding,
    }
}

// fn gen_lagrange_multipliers(y: &[SecureField]) {
//     let num_variables = y.len();
//     let y = y.iter()
// }

// xs = {0, 1}^1
// ys = r0
// (0 * r0)
//
//

// fn lagrange_extension(y: &[SecureField]) -> Vec<SecureField> {
//     let num_variables = y.len();
//     let n = 1 << num_variables;

//     let n = domain.size() / 2;
//     let log_n = n.ilog2() as usize;
//     // Compute all the required mappings of our generator point.
//     let mut mappings = Vec::with_capacity(log_n);
//     let mut g = domain.coset().step;
//     for _ in 0..log_n {
//         mappings.push(g);
//         g = g.double();
//     }
//     // Incrementally produce bit-reversed elements.
//     let mut elements = Vec::with_capacity(n);
//     elements.push(domain.coset().initial);
//     let mut segment_index = 0;
//     while let Some(mapping) = mappings.pop() {
//         for i in 0..1 << segment_index {
//             let element = mapping + elements[i];
//             elements.push(element);
//         }
//         segment_index += 1;
//     }
//     // We only need the x-coordinates for a [LineDomain].
//     elements.into_iter().map(|v| v.x).collect()
// }

#[cfg(test)]
mod tests {
    use std::iter::{self, zip, Sum};
    use std::ops::{Add, Mul, Neg, Sub};
    use std::time::Instant;

    use num_traits::{One, Zero};
    use prover_research::commitment_scheme::blake2_hash::Blake2sHasher;
    use prover_research::commitment_scheme::hasher::Hasher;
    use prover_research::core::channel::{Blake2sChannel, Channel};
    use prover_research::core::fields::m31::BaseField;
    use prover_research::core::fields::qm31::SecureField;
    use prover_research::core::fields::{ExtensionOf, Field};

    use crate::gkr::{c0, collapse_c, eq, partially_verify, prove, prove2, Layer, MleLayer};
    use crate::multivariate::{self, MultivariatePolynomial};
    use crate::sumcheck::MultiLinearExtension;
    use crate::utils::{Fraction, Polynomial};

    #[test]
    fn gkr_logup() {
        let zero = SecureField::zero();
        let one = SecureField::one();

        // Random fractions.
        let a = Fraction::new(BaseField::one(), BaseField::from(123));
        let b = Fraction::new(BaseField::one(), BaseField::from(1));
        let c = Fraction::new(BaseField::one(), BaseField::from(9999999));

        // List of fractions that sum to zero.
        let fractions = [
            a + a + a, //
            a + b + b + c,
            -a,
            -a,
            -a,
            c - b - b,
            -a - c,
            -c,
        ];

        assert!(fractions.into_iter().sum::<Fraction<BaseField>>().is_zero());

        let p3 = multivariate::from_const_fn(|[z0, z1, z2]: [SecureField; 3]| {
            eq(&[z0, z1, z2], &[zero, zero, zero]) * fractions[0].numerator
                + eq(&[z0, z1, z2], &[zero, zero, one]) * fractions[1].numerator
                + eq(&[z0, z1, z2], &[zero, one, zero]) * fractions[2].numerator
                + eq(&[z0, z1, z2], &[zero, one, one]) * fractions[3].numerator
                + eq(&[z0, z1, z2], &[one, zero, zero]) * fractions[4].numerator
                + eq(&[z0, z1, z2], &[one, zero, one]) * fractions[5].numerator
                + eq(&[z0, z1, z2], &[one, one, zero]) * fractions[6].numerator
                + eq(&[z0, z1, z2], &[one, one, one]) * fractions[7].numerator
        });

        let q3 = multivariate::from_const_fn(|[z0, z1, z2]: [SecureField; 3]| {
            eq(&[z0, z1, z2], &[zero, zero, zero]) * fractions[0].denominator
                + eq(&[z0, z1, z2], &[zero, zero, one]) * fractions[1].denominator
                + eq(&[z0, z1, z2], &[zero, one, zero]) * fractions[2].denominator
                + eq(&[z0, z1, z2], &[zero, one, one]) * fractions[3].denominator
                + eq(&[z0, z1, z2], &[one, zero, zero]) * fractions[4].denominator
                + eq(&[z0, z1, z2], &[one, zero, one]) * fractions[5].denominator
                + eq(&[z0, z1, z2], &[one, one, zero]) * fractions[6].denominator
                + eq(&[z0, z1, z2], &[one, one, one]) * fractions[7].denominator
        });

        let p2 = multivariate::from_const_fn(|[z0, z1]: [SecureField; 2]| {
            // To prevent `p3` and `q3` being `move`d into the closure.
            let (p3, q3) = (&p3, &q3);

            let g = move |[x0, x1]: [SecureField; 2]| {
                eq(&[z0, z1], &[x0, x1])
                    * (p3.eval(&[x0, x1, one]) * q3.eval(&[x0, x1, zero])
                        + p3.eval(&[x0, x1, zero]) * q3.eval(&[x0, x1, one]))
            };

            g([zero, zero]) + g([zero, one]) + g([one, zero]) + g([one, one])
        });

        let q2 = multivariate::from_const_fn(|[z0, z1]: [SecureField; 2]| {
            // To prevent `q3` being `move`d into the closure.
            let q3 = &q3;

            let g = move |[x0, x1]: [SecureField; 2]| {
                eq(&[z0, z1], &[x0, x1]) * q3.eval(&[x0, x1, one]) * q3.eval(&[x0, x1, zero])
            };

            // {
            //     let x0 = SecureField::zero();
            //     let x1 = SecureField::one();
            //     let x2 = -SecureField::one();
            //     let x3 = x1 + x1;
            //     let x4 = x1 + x1 + x1;

            //     let y0 = g([x0, zero]) + g([x0, one]);
            //     let y1 = g([x1, zero]) + g([x1, one]);
            //     let y2 = g([x2, zero]) + g([x2, one]);
            //     let y3 = g([x3, zero]) + g([x3, one]);
            //     let y4 = g([x4, zero]) + g([x4, one]);

            //     let poly =
            //         Polynomial::interpolate_lagrange(&[x0, x1, x2, x3, x4], &[y0, y1, y2, y3,
            // y4]);

            //     println!("degree: {:?}", poly);
            //     println!("degree: {}", poly.degree());
            // }

            g([zero, zero]) + g([zero, one]) + g([one, zero]) + g([one, one])
        });

        let p1 = multivariate::from_const_fn(|[z0]: [SecureField; 1]| {
            // To prevent `p2` and `q2` being `move`d into the closure.
            let (p2, q2) = (&p2, &q2);

            let g = move |[x0]: [SecureField; 1]| {
                eq(&[z0], &[x0])
                    * (p2.eval(&[x0, one]) * q2.eval(&[x0, zero])
                        + p2.eval(&[x0, zero]) * q2.eval(&[x0, one]))
            };

            g([zero]) + g([one])
        });

        let q1 = multivariate::from_const_fn(|[z0]: [SecureField; 1]| {
            let q2 = &q2;

            let g = move |[x0]: [SecureField; 1]| {
                eq(&[z0], &[x0]) * q2.eval(&[x0, zero]) * q2.eval(&[x0, one])
            };

            g([zero]) + g([one])
        });

        let layers = vec![
            Layer::new(&p1, &q1),
            Layer::new(&p2, &q2),
            Layer::new(&p3, &q3),
        ];

        let proof = prove(&mut test_channel(), &layers);
        let (assignment, p3_claim, q3_claim) =
            partially_verify(&proof, &mut test_channel()).unwrap();

        assert_eq!(p3.eval(&assignment), p3_claim);
        assert_eq!(q3.eval(&assignment), q3_claim);

        println!("====V2====");

        let mut random_fractions = fractions.to_vec();

        let now = Instant::now();
        let mut layers = Vec::new();
        while random_fractions.len() > 1 {
            layers.push(MleLayer::new(&random_fractions));
            random_fractions = random_fractions
                .array_chunks()
                .map(|&[a, b]| a + b)
                .collect();
        }
        layers.reverse();

        println!("layer gen time: {:?}", now.elapsed());

        // println!("yo: {}" )

        let proof = prove2(&mut test_channel(), &layers);
        // let (assignment, p3_claim, q3_claim) =
        //     partially_verify(&proof, &mut test_channel()).unwrap();
        let res = partially_verify(&proof, &mut test_channel());
        println!("result: {:?}", res)
    }

    #[test]
    fn fooling_around() {
        let one = SecureField::one();
        let zero = SecureField::zero();

        let random = SecureField::from_m31(
            BaseField::from(1238),
            BaseField::from(4305),
            BaseField::from(899120),
            BaseField::from(987),
        );

        println!("YO: {}", eq(&[random, random], &[one, zero]));
        println!("YO: {}", eq_one_zero(&[random, random], &[one, zero]));
    }

    pub fn eq_one_zero(
        x_assignments: &[SecureField],
        w_assignments: &[SecureField],
    ) -> SecureField {
        assert_eq!(x_assignments.len(), w_assignments.len());

        let n = x_assignments.len();
        let one = SecureField::one();

        zip(x_assignments, w_assignments)
            .map(|(&xi, &wi)| xi * wi + (one - xi) * (one - wi))
            .product::<SecureField>()
    }

    #[test]
    fn mle_bench() {
        // Random fractions.
        let a = Fraction::new(BaseField::one(), BaseField::from(123));
        let b = Fraction::new(BaseField::one(), BaseField::from(1));
        let c = Fraction::new(BaseField::one(), BaseField::from(9999999));

        const N: usize = 1 << 3;

        let mut channel = test_channel();
        let mut random_fractions = zip(channel.draw_felts(N), channel.draw_felts(N))
            .map(|(numerator, denominator)| Fraction::new(numerator, denominator))
            .collect::<Vec<Fraction<SecureField>>>();

        // Make the fractions sum to zero.
        let sum = random_fractions.iter().sum::<Fraction<SecureField>>();
        random_fractions[0] = random_fractions[0] - sum;

        let now = Instant::now();
        let mut layers = Vec::new();
        while random_fractions.len() > 1 {
            layers.push(MleLayer::new(&random_fractions));
            random_fractions = random_fractions
                .array_chunks()
                .map(|&[a, b]| a + b)
                .collect();
        }
        layers.reverse();

        println!("layer gen time: {:?}", now.elapsed());

        // println!("yo: {}" )

        let proof = prove2(&mut test_channel(), &layers);
        // let (assignment, p3_claim, q3_claim) =
        //     partially_verify(&proof, &mut test_channel()).unwrap();
        let res = partially_verify(&proof, &mut test_channel());
        println!("result: {:?}", res)

        // // List of fractions that sum to zero.
        // let fractions = [
        //     a + a + a, //
        //     a + b + b + c,
        //     -a,
        //     -a,
        //     -a,
        //     c - b - b,
        //     -a - c,
        //     -c,
        // ];
    }

    #[test]
    fn collapsing_test() {
        let one = SecureField::one();
        let zero = SecureField::zero();

        let mut channel = test_channel();
        let random = channel.draw_felt();
        let random2 = channel.draw_felt();
        let random3 = channel.draw_felt();
        let mle = MultiLinearExtension::<SecureField>::new(
            (0..8).map(|i| BaseField::from(i).into()).collect(),
        );

        let g = multivariate::from_const_fn(|[random]| {
            eq(&[zero, zero, random], &[zero, zero, random2]) * mle[0]
                + eq(&[zero, zero, random], &[zero, zero, random2]) * mle[1]
                + eq(&[zero, one, random], &[zero, one, random2]) * mle[2]
                + eq(&[zero, one, random], &[zero, one, random2]) * mle[3]
                + eq(&[one, zero, random], &[one, zero, random2]) * mle[4]
                + eq(&[one, zero, random], &[one, zero, random2]) * mle[5]
                + eq(&[one, one, random], &[one, one, random2]) * mle[6]
                + eq(&[one, one, random], &[one, one, random2]) * mle[7]
        });

        let collapse = |mle: &MultiLinearExtension<SecureField>,
                        x: SecureField|
         -> (
            MultiLinearExtension<SecureField>,
            Polynomial<SecureField>,
            SecureField,
        ) {
            let [r0, r1] = mle.array_chunks().fold([SecureField::zero(); 2], sum_simd);

            // TODO: Send evaluations to verifier not coefficients
            let r = Polynomial::interpolate_lagrange(
                &[SecureField::zero(), SecureField::one()],
                &[r0, r1],
            );
            // channel.mix_felts(&r);
            // let challenge = channel.draw_felt();
            let challenge = random;

            let collapsed_mle = MultiLinearExtension::new(
                mle.array_chunks()
                    // Computes `(1 - challenge) * e + challenge * o` with one less multiplication.
                    // .map(|&[e, o]| challenge * (o - e) + e)
                    .map(|&[e, o]| {
                        ((random2 * challenge)
                            + (SecureField::one() - challenge) * (SecureField::one() - random2))
                            * e
                            + ((random2 * challenge)
                                + (SecureField::one() - challenge) * (SecureField::one() - random2))
                                * o
                    })
                    .collect(),
            );

            // x * y + (1 - x)*(1 - y)
            //

            (collapsed_mle, r, challenge)
        };

        fn sum_simd<F: Field, const N: usize>(mut a: [F; N], b: &[F; N]) -> [F; N] {
            zip(&mut a, b).for_each(|(a, b)| *a += *b);
            a
        }

        // let (collapsed_mle, poly, _) = collapse(&mle);
        // println!("yo: {:?}", poly.eval(random));
        // println!("yoyo: {:?}", g.eval(&[random]));

        // println!("yo: {}", eq(&[one, zero], &[random, random2]));
        // println!("yo: {}", eq(&[one], &[random]));
        println!("yo: {}", eq(&[one, zero], &[random, random2]));
        let c = c0(&[random, random2]);
        println!("YO: {}", c[2]);

        println!("yo: {}", eq(&[random3, zero], &[random, random2]));
        println!("yo: {}", collapse_c(c, random, random3)[0])
        // println!("YO: {}", c0(&[random, random2])[2]);
    }

    fn test_channel() -> Blake2sChannel {
        let seed = Blake2sHasher::hash(&[]);
        Blake2sChannel::new(seed)
    }
}
