use std::iter::zip;

use num_traits::{One, Zero};
use prover_research::core::channel::Channel;
use prover_research::core::fields::m31::BaseField;
use prover_research::core::fields::qm31::SecureField;
use prover_research::core::fields::Field;
use thiserror::Error;

use crate::mle::MultiLinearExtension;
use crate::q31::FastSecureField;
use crate::sumcheck::{self, SumcheckError, SumcheckOracle, SumcheckProof};
use crate::utils::{Fraction, Polynomial};

/// Multi-linear extension trace.
// TODO: Implement.
pub struct MleTrace<F: Field> {
    _columns: Vec<MultiLinearExtension<F>>,
}

#[derive(Debug, Clone)]
pub struct MleLayer {
    p: MultiLinearExtension<SecureField>,
    q: MultiLinearExtension<SecureField>,
    num_variables: u32,
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
            // n_vars: f.len().ilog2() as usize,
            p: MultiLinearExtension::new(p),
            q: MultiLinearExtension::new(q),
            num_variables: f.len().ilog2(),
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

/// Partially verifies a GKR proof.
///
/// Returns the variable assignment and claimed evaluation in the top layer. This claim and  is left
/// to the verifier to check - hence partial verification.
///
/// Output of the form `(variable_assignment, p_claimed_eval, q_claimed_eval)`.
pub fn partially_verify(
    proof: &GkrProof,
    channel: &mut impl Channel,
) -> Result<(Vec<SecureField>, SecureField, SecureField), GkrError> {
    let zero = SecureField::zero();
    let one = SecureField::one();

    let GkrProof {
        output_layer,
        layer_proofs,
    } = proof;

    channel.mix_felts(&output_layer.p);
    channel.mix_felts(&output_layer.q);

    // TODO: Better variable names.
    let mut round_r = channel.draw_felts(output_layer.num_variables as usize);
    let mut round_p_claim = output_layer.p.eval(&round_r);
    let mut round_q_claim = output_layer.q.eval(&round_r);

    for (layer, layer_proof) in (1..).zip(layer_proofs) {
        let GkrLayerProof {
            sumcheck_proof,
            p_eval_encoding,
            q_eval_encoding,
        } = layer_proof;

        let lambda = channel.draw_felt();

        let sumcheck_claim = round_p_claim + lambda * round_q_claim;
        let (variable_assignment, eval) =
            sumcheck::partially_verify(sumcheck_claim, sumcheck_proof, channel)
                .map_err(|source| GkrError::InvalidSumcheck { layer, source })?;

        let circuit_output = {
            let a = Fraction {
                numerator: p_eval_encoding.eval(zero),
                denominator: q_eval_encoding.eval(zero),
            };

            let b = Fraction {
                numerator: p_eval_encoding.eval(one),
                denominator: q_eval_encoding.eval(one),
            };

            let c = a + b;

            eq(&round_r, &variable_assignment) * (c.numerator + lambda * c.denominator)
        };

        if eval != circuit_output {
            return Err(GkrError::CircuitCheckFailure {
                claim: eval,
                output: circuit_output,
                layer,
            });
        }

        channel.mix_felts(p_eval_encoding);
        channel.mix_felts(q_eval_encoding);

        let r_star = channel.draw_felt();
        round_p_claim = p_eval_encoding.eval(r_star);
        round_q_claim = q_eval_encoding.eval(r_star);
        round_r = variable_assignment;
        round_r.push(r_star);
    }

    Ok((round_r, round_p_claim, round_q_claim))
}

/// Error encountered during GKR protocol verification.
///
/// Layer 1 corresponds to the output layer.
#[derive(Error, Debug)]
pub enum GkrError {
    #[error("sum-check invalid in layer {layer}: {source}")]
    InvalidSumcheck { layer: usize, source: SumcheckError },
    #[error("circuit check failed in layer {layer} (calculated {output}, claim {claim})")]
    CircuitCheckFailure {
        claim: SecureField,
        output: SecureField,
        layer: usize,
    },
}

#[derive(Debug, Clone)]
pub struct GkrProof {
    layer_proofs: Vec<GkrLayerProof>,
    output_layer: MleLayer,
    // p1_eval_encoding: Polynomial<SecureField>,
    // q1_eval_encoding: Polynomial<SecureField>,
}

#[derive(Debug, Clone)]
struct GkrLayerProof {
    sumcheck_proof: SumcheckProof,
    p_eval_encoding: Polynomial<SecureField>,
    q_eval_encoding: Polynomial<SecureField>,
}

struct Oracle {
    /// p_{i + 1}
    // TODO: Consider `Cow<Vec<F>>`
    p: MultiLinearExtension<FastSecureField>,
    /// q_{i + 1}
    q: MultiLinearExtension<FastSecureField>,
    // TODO: docs.
    c: Vec<FastSecureField>,
    num_variables: u32,
    z: Vec<FastSecureField>,
    lambda: FastSecureField,
}

impl Oracle {
    fn new(
        z: &[SecureField],
        p: MultiLinearExtension<SecureField>,
        q: MultiLinearExtension<SecureField>,
        lambda: SecureField,
    ) -> Self {
        let num_variables = z.len() as u32;
        assert_eq!(p.len(), 2 << num_variables);
        assert_eq!(q.len(), 2 << num_variables);
        // let now = Instant::now();
        let c = c0(unsafe { std::mem::transmute(z) });
        // println!("c gen time: {:?}", now.elapsed());
        Self {
            p: unsafe { std::mem::transmute(p) },
            q: unsafe { std::mem::transmute(q) },
            num_variables,
            c,
            z: unsafe { std::mem::transmute(z.to_vec()) },
            lambda: lambda.into(),
        }
    }
}

impl SumcheckOracle for Oracle {
    fn num_variables(&self) -> u32 {
        self.num_variables
    }

    #[no_mangle]
    fn univariate_sum(&self) -> Polynomial<SecureField> {
        let zero = FastSecureField::zero();
        let one = FastSecureField::one();

        // let t_bar = one - t;
        let z = self.z[0];
        let z_bar = one - z;
        let z_bar_inv = z_bar.inverse();
        // let eq_shift = z_bar_inv * (t * z + t_bar * z_bar);
        // z / (1 - z) * (1/z)/(1/z)
        // 1 / (1/z - 1)

        let eq_shift1 = {
            let t = one;
            let t_bar = one - t;
            z_bar_inv * (t * z + t_bar * z_bar)
        };

        let eq_shift_neg1 = {
            let t = -one;
            let t_bar = one - t;
            z_bar_inv * (t * z + t_bar * z_bar)
        };

        let eq_shift2 = {
            let t = one + one;
            let t_bar = one - t;
            z_bar_inv * (t * z + t_bar * z_bar)
        };

        let n_terms = self.c.len() / 2;
        let c_vals = &self.c[0..n_terms];
        let (p_lhs_pairs, p_rhs_pairs) = self.p.as_chunks().0.split_at(n_terms);
        let (q_lhs_pairs, q_rhs_pairs) = self.q.as_chunks().0.split_at(n_terms);

        let [y0, y1, y2, y3] = zip(
            c_vals,
            zip(zip(p_lhs_pairs, p_rhs_pairs), zip(q_lhs_pairs, q_rhs_pairs)),
        )
        .fold(
            [zero; 4],
            |acc,
             (
                &c,
                ((&[p0_lhs, p1_lhs], &[p0_rhs, p1_rhs]), (&[q0_lhs, q1_lhs], &[q0_rhs, q1_rhs])),
            )| {
                // eval at 0:
                let eval0 = {
                    let a = Fraction::new(p0_lhs, q0_lhs);
                    let b = Fraction::new(p1_lhs, q1_lhs);
                    let res = a + b;
                    c * (res.numerator + self.lambda * res.denominator)
                };

                // eval at 1:
                let eval1 = {
                    let a = Fraction::new(p0_rhs, q0_rhs);
                    let b = Fraction::new(p1_rhs, q1_rhs);
                    let res = a + b;
                    c * eq_shift1 * (res.numerator + self.lambda * res.denominator)
                };

                // eval at -1:
                let evaln1 = {
                    let p0_eval = p0_lhs.double() - p0_rhs;
                    let p1_eval = p1_lhs.double() - p1_rhs;
                    let q0_eval = q0_lhs.double() - q0_rhs;
                    let q1_eval = q1_lhs.double() - q1_rhs;
                    let a = Fraction::new(p0_eval, q0_eval);
                    let b = Fraction::new(p1_eval, q1_eval);
                    let res = a + b;
                    c * eq_shift_neg1 * (res.numerator + self.lambda * res.denominator)
                };

                // eval at 2:
                let eval2 = {
                    let p0_eval = p0_rhs.double() - p0_lhs;
                    let p1_eval = p1_rhs.double() - p1_lhs;
                    let q0_eval = q0_rhs.double() - q0_lhs;
                    let q1_eval = q1_rhs.double() - q1_lhs;
                    let a = Fraction::new(p0_eval, q0_eval);
                    let b = Fraction::new(p1_eval, q1_eval);
                    let res = a + b;
                    c * eq_shift2 * (res.numerator + self.lambda * res.denominator)
                };

                [
                    acc[0] + eval0,
                    acc[1] + eval1,
                    acc[2] + evaln1,
                    acc[3] + eval2,
                ]
            },
        );

        let x0 = BaseField::zero();
        let x1 = BaseField::one();
        let x2 = -BaseField::one();
        let x3 = BaseField::from(2);

        Polynomial::interpolate_lagrange(
            &[x0.into(), x1.into(), x2.into(), x3.into()],
            &[y0.into(), y1.into(), y2.into(), y3.into()],
        )
    }

    #[no_mangle]
    fn fix_first(self, challenge: SecureField) -> Self {
        let challenge: FastSecureField = challenge.into();

        let c = collapse_c(self.c, self.z[0], challenge);
        // let now = Instant::now();
        // println!("collapsing time: {:?}", now.elapsed());

        Self {
            p: self.p.fix_first(challenge),
            q: self.q.fix_first(challenge),
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
fn c0(z: &[FastSecureField]) -> Vec<FastSecureField> {
    match z {
        [] => vec![],
        &[z1] => vec![FastSecureField::one() - z1, z1],
        &[zj, ref z @ ..] => {
            let c = c0(z);
            let zj_bar = FastSecureField::one() - zj;
            let lhs = c.iter().map(|&v| zj_bar * v);
            let rhs = c.iter().map(|&v| zj * v);
            Iterator::chain(lhs, rhs).collect()
        }
    }
}

/// Source: <https://eprint.iacr.org/2013/351.pdf> (Section 5.4.1)
fn collapse_c(
    mut c: Vec<FastSecureField>,
    z: FastSecureField,
    r: FastSecureField,
) -> Vec<FastSecureField> {
    // TODO: `z` can be zero! Just divide out (1 - z) instead and take even values of `c`. Don't
    // want to implement this just noting here.
    assert!(!z.is_zero());

    let z_bar = FastSecureField::one() - z;
    let r_bar = FastSecureField::one() - r;
    // TODO: Shift not right word.
    let shift = z_bar.inverse() * (r * z + r_bar * z_bar);

    c.truncate(c.len() / 2);
    c.iter_mut().for_each(|v| *v *= shift);

    c
}

// <https://people.cs.georgetown.edu/jthaler/ProofsArgsAndZK.pdf> (page 65)
pub fn prove(channel: &mut (impl Channel + Clone), layers: Vec<MleLayer>) -> GkrProof {
    let mut layers = layers.into_iter();

    let output_layer = layers.next().expect("must contain a layer");
    channel.mix_felts(&output_layer.p);
    channel.mix_felts(&output_layer.q);

    let mut r = channel.draw_felts(output_layer.num_variables as usize);

    let layer_proofs = layers
        .map(|MleLayer { p, q, .. }| {
            let lambda = channel.draw_felt();

            let oracle = Oracle::new(&r, p, q, lambda);
            let (sumcheck_proof, variable_assignment, oracle) = sumcheck::prove(oracle, channel);

            let p_eval_encoding = {
                let [x0, x1] = [SecureField::zero(), SecureField::one()];
                let [y0, y1] = [oracle.p[0].into(), oracle.p[1].into()];
                Polynomial::interpolate_lagrange(&[x0, x1], &[y0, y1])
            };

            let q_eval_encoding = {
                let [x0, x1] = [SecureField::zero(), SecureField::one()];
                let [y0, y1] = [oracle.q[0].into(), oracle.q[1].into()];
                Polynomial::interpolate_lagrange(&[x0, x1], &[y0, y1])
            };

            channel.mix_felts(&p_eval_encoding);
            channel.mix_felts(&q_eval_encoding);

            r = variable_assignment;
            r.push(channel.draw_felt());

            GkrLayerProof {
                sumcheck_proof,
                p_eval_encoding,
                q_eval_encoding,
            }
        })
        .collect();

    GkrProof {
        layer_proofs,
        output_layer,
    }
}

#[cfg(test)]
mod tests {
    use std::iter::zip;
    use std::time::Instant;

    use prover_research::commitment_scheme::blake2_hash::Blake2sHasher;
    use prover_research::commitment_scheme::hasher::Hasher;
    use prover_research::core::channel::{Blake2sChannel, Channel};
    use prover_research::core::fields::qm31::SecureField;

    use crate::gkr::{partially_verify, prove, MleLayer};
    use crate::utils::Fraction;

    #[test]
    fn mle_bench() {
        const N: usize = 1 << 21;

        let mut channel = test_channel();
        let mut random_fractions = zip(
            channel.draw_felts(N).into_iter().map(SecureField::from),
            channel.draw_felts(N).into_iter().map(SecureField::from),
        )
        .map(|(numerator, denominator)| Fraction::new(numerator, denominator))
        .collect::<Vec<Fraction<SecureField>>>();

        // Make the fractions sum to zero.
        let now = Instant::now();
        let sum = random_fractions.iter().sum::<Fraction<SecureField>>();
        println!("layer sum time: {:?}", now.elapsed());
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

        let now = Instant::now();
        let proof = prove(&mut test_channel(), layers);
        println!("proof gen time: {:?}", now.elapsed());

        // let (assignment, p3_claim, q3_claim) =
        //     partially_verify(&proof, &mut test_channel()).unwrap();
        let now = Instant::now();
        let res = partially_verify(&proof, &mut test_channel());
        println!("verify time: {:?}", now.elapsed());
        assert!(res.is_ok());

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

    fn test_channel() -> Blake2sChannel {
        let seed = Blake2sHasher::hash(&[]);
        Blake2sChannel::new(seed)
    }
}
