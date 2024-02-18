use std::iter::zip;

use num_traits::{One, Zero};
use prover_research::core::channel::Channel;
use prover_research::core::fields::m31::{BaseField, P};
use prover_research::core::fields::qm31::SecureField;

use crate::multivariate::{self, MultivariatePolynomial};
use crate::sumcheck::{self, SumcheckProof};
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

/// Evaluates the lagrange kernel of the boolean hypercube.
///
/// When y is an elements of `{+-1}^n` then this function evaluates the Lagrange polynomial which is
/// the unique multilinear polynomial equal to 1 if `x = y` and equal to 0 whenever x is an element
/// of `{+-1}^n`.
///
/// From: <https://eprint.iacr.org/2023/1284.pdf>.
pub fn eq(x_assignments: &[SecureField], y_assignments: &[SecureField]) -> SecureField {
    assert_eq!(x_assignments.len(), y_assignments.len());

    let n = x_assignments.len();
    let norm = BaseField::from_u32_unchecked(2u32.pow(n as u32));

    zip(x_assignments, y_assignments)
        .map(|(&x, &y)| SecureField::one() + x * y)
        .product::<SecureField>()
        / norm
}

// <https://people.cs.georgetown.edu/jthaler/ProofsArgsAndZK.pdf> (page 65)
pub fn prove2(channel: &mut impl Channel, layers: &[Layer]) -> GkrProof {
    let zero = SecureField::zero();
    let one = SecureField::one();

    let Layer { p: p1, q: q1 } = &layers[0];

    let p1_eval_encoding = {
        let [y0, y1] = [p1.eval(&[-one]), p1.eval(&[one])];
        Polynomial::interpolate_lagrange(&[zero, one], &[y0, y1])
    };

    let q1_eval_encoding = {
        let [y0, y1] = [q1.eval(&[-one]), q1.eval(&[one])];
        Polynomial::interpolate_lagrange(&[zero, one], &[y0, y1])
    };

    channel.mix_felts(&p1_eval_encoding);
    channel.mix_felts(&q1_eval_encoding);

    let r_star = channel.draw_random_secure_felts()[0];
    let r0 = eval_canonic_line(&[-one], &[one], r_star);

    let layer_proofs = layers
        .array_windows()
        .scan(r0, |r, [layer, next_layer]| {
            let lambda = channel.draw_random_secure_felts()[0];

            let g = multivariate::from_fn(layer.p.num_variables(), |y| {
                let y_one = [y, &[one]].concat();
                let y_neg_one = [y, &[-one]].concat();

                let v0 = Fraction {
                    numerator: next_layer.p.eval(&y_neg_one),
                    denominator: next_layer.q.eval(&y_neg_one),
                };

                let v1 = Fraction {
                    numerator: next_layer.p.eval(&y_one),
                    denominator: next_layer.q.eval(&y_one),
                };

                let res = v0 + v1;

                eq(r, y) * (res.numerator + lambda * res.denominator)
            });

            let (sumcheck_proof, eval_point) = sumcheck::prove(&g, channel);

            let Layer {
                p: p_next,
                q: q_next,
            } = next_layer;

            // b* and c*
            let b_star = [&*eval_point, &[-one]].concat();
            let c_star = [&*eval_point, &[one]].concat();

            let p_eval_encoding = Polynomial::interpolate_lagrange(
                &[zero, one],
                &[p_next.eval(&b_star), p_next.eval(&c_star)],
            );

            let q_eval_encoding = Polynomial::interpolate_lagrange(
                &[zero, one],
                &[q_next.eval(&b_star), q_next.eval(&c_star)],
            );

            channel.mix_felts(&p_eval_encoding);
            channel.mix_felts(&q_eval_encoding);

            let r_star = channel.draw_random_secure_felts()[0];
            *r = eval_canonic_line(&b_star, &c_star, r_star);

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

    if !(v0 + v1).is_zero() {
        return None;
    }

    channel.mix_felts(p1_eval_encoding);
    channel.mix_felts(q1_eval_encoding);

    let r_star = channel.draw_random_secure_felts()[0];
    let r0 = eval_canonic_line(&[-one], &[one], r_star);

    let p_m0 = p1_eval_encoding.eval(r_star);
    let q_m0 = q1_eval_encoding.eval(r_star);

    layer_proofs
        .iter()
        .try_fold((r0, p_m0, q_m0), |(r, p_m, q_m), layer_proof| {
            let GkrLayerProof {
                sumcheck_proof,
                p_eval_encoding,
                q_eval_encoding,
            } = layer_proof;

            let lambda = channel.draw_random_secure_felts()[0];
            let m = p_m + lambda * q_m;
            let (assignment, eval) = sumcheck::partially_verify(m, sumcheck_proof, channel)?;

            let v0 = Fraction {
                numerator: p_eval_encoding.eval(zero),
                denominator: q_eval_encoding.eval(zero),
            };

            let v1 = Fraction {
                numerator: p_eval_encoding.eval(one),
                denominator: q_eval_encoding.eval(one),
            };

            let res = v0 + v1;

            if eval != eq(&r, &assignment) * (res.numerator + lambda * res.denominator) {
                return None;
            }

            channel.mix_felts(p_eval_encoding);
            channel.mix_felts(q_eval_encoding);

            let b_star = [assignment.clone(), vec![-one]].concat();
            let c_star = [assignment.clone(), vec![one]].concat();
            let r_star = channel.draw_random_secure_felts()[0];

            Some((
                eval_canonic_line(&b_star, &c_star, r_star),
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

#[cfg(test)]
mod tests {
    use std::iter::Sum;
    use std::ops::{Add, Mul, Neg, Sub};

    use num_traits::{One, Zero};
    use prover_research::commitment_scheme::blake2_hash::Blake2sHasher;
    use prover_research::commitment_scheme::hasher::Hasher;
    use prover_research::core::channel::{Blake2sChannel, Channel};
    use prover_research::core::fields::m31::BaseField;
    use prover_research::core::fields::qm31::SecureField;

    use crate::gkr::{eq, partially_verify, prove2, Layer};
    use crate::multivariate::{self, MultivariatePolynomial};
    use crate::utils::Fraction;

    #[test]
    fn gkr_logup() {
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
            eq(&[z0, z1, z2], &[-one, -one, -one]) * fractions[0].numerator
                + eq(&[z0, z1, z2], &[-one, -one, one]) * fractions[1].numerator
                + eq(&[z0, z1, z2], &[-one, one, -one]) * fractions[2].numerator
                + eq(&[z0, z1, z2], &[-one, one, one]) * fractions[3].numerator
                + eq(&[z0, z1, z2], &[one, -one, -one]) * fractions[4].numerator
                + eq(&[z0, z1, z2], &[one, -one, one]) * fractions[5].numerator
                + eq(&[z0, z1, z2], &[one, one, -one]) * fractions[6].numerator
                + eq(&[z0, z1, z2], &[one, one, one]) * fractions[7].numerator
        });

        let q3 = multivariate::from_const_fn(|[z0, z1, z2]: [SecureField; 3]| {
            eq(&[z0, z1, z2], &[-one, -one, -one]) * fractions[0].denominator
                + eq(&[z0, z1, z2], &[-one, -one, one]) * fractions[1].denominator
                + eq(&[z0, z1, z2], &[-one, one, -one]) * fractions[2].denominator
                + eq(&[z0, z1, z2], &[-one, one, one]) * fractions[3].denominator
                + eq(&[z0, z1, z2], &[one, -one, -one]) * fractions[4].denominator
                + eq(&[z0, z1, z2], &[one, -one, one]) * fractions[5].denominator
                + eq(&[z0, z1, z2], &[one, one, -one]) * fractions[6].denominator
                + eq(&[z0, z1, z2], &[one, one, one]) * fractions[7].denominator
        });

        let p2 = multivariate::from_const_fn(|[z0, z1]: [SecureField; 2]| {
            // To prevent `p3` and `q3` being `move`d into the closure.
            let (p3, q3) = (&p3, &q3);

            let g = move |[x0, x1]: [SecureField; 2]| {
                eq(&[z0, z1], &[x0, x1])
                    * (p3.eval(&[x0, x1, one]) * q3.eval(&[x0, x1, -one])
                        + p3.eval(&[x0, x1, -one]) * q3.eval(&[x0, x1, one]))
            };

            g([-one, -one]) + g([-one, one]) + g([one, -one]) + g([one, one])
        });

        let q2 = multivariate::from_const_fn(|[z0, z1]: [SecureField; 2]| {
            // To prevent `q3` being `move`d into the closure.
            let q3 = &q3;

            let g = move |[x0, x1]: [SecureField; 2]| {
                eq(&[z0, z1], &[x0, x1]) * q3.eval(&[x0, x1, one]) * q3.eval(&[x0, x1, -one])
            };

            g([-one, -one]) + g([-one, one]) + g([one, -one]) + g([one, one])
        });

        let p1 = multivariate::from_const_fn(|[z0]: [SecureField; 1]| {
            // To prevent `p2` and `q2` being `move`d into the closure.
            let (p2, q2) = (&p2, &q2);

            let g = move |[x0]: [SecureField; 1]| {
                eq(&[z0], &[x0])
                    * (p2.eval(&[x0, one]) * q2.eval(&[x0, -one])
                        + p2.eval(&[x0, -one]) * q2.eval(&[x0, one]))
            };

            g([-one]) + g([one])
        });

        let q1 = multivariate::from_const_fn(|[z0]: [SecureField; 1]| {
            let q2 = &q2;

            let g = move |[x0]: [SecureField; 1]| {
                eq(&[z0], &[x0]) * q2.eval(&[x0, -one]) * q2.eval(&[x0, one])
            };

            g([-one]) + g([one])
        });

        let layers = vec![
            Layer::new(&p1, &q1),
            Layer::new(&p2, &q2),
            Layer::new(&p3, &q3),
        ];

        let proof = prove2(&mut test_channel(), &layers);
        let (assignment, p3_claim, q3_claim) =
            partially_verify(&proof, &mut test_channel()).unwrap();
        assert_eq!(p3.eval(&assignment), p3_claim);
        assert_eq!(q3.eval(&assignment), q3_claim);
        println!("res: {:?}", "dasdsd");
    }

    fn test_channel() -> Blake2sChannel {
        let seed = Blake2sHasher::hash(&[]);
        Blake2sChannel::new(seed)
    }
}
