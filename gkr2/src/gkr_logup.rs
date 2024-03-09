use std::iter::zip;
use std::ops::Add;
use std::time::Instant;

use num_traits::{One, Zero};
use prover_research::core::fields::m31::BaseField;
use prover_research::core::fields::qm31::SecureField;
use prover_research::core::fields::{ExtensionOf, Field};

use crate::gkr_generic::{GkrLayer, GkrSumcheckOracle};
use crate::mle::{Mle, MleTrace};
use crate::sumcheck::SumcheckOracle;
use crate::utils::{horner_eval, Polynomial};

// TODO: docs
pub enum LogupTrace {
    /// All numerators implicitly equal "1".
    Singles { denominators: Mle<SecureField> },
    Multiplicities {
        /// The multiplicities.
        numerators: Mle<BaseField>,
        denominators: Mle<SecureField>,
    },
    Generic {
        numerators: Mle<SecureField>,
        denominators: Mle<SecureField>,
    },
}

impl LogupTrace {
    pub fn new(numerators: Mle<SecureField>, denominators: Mle<SecureField>) -> Self {
        assert_eq!(numerators.num_variables(), denominators.num_variables());
        Self::Generic {
            numerators,
            denominators,
        }
    }

    pub fn new_multiplicities(numerators: Mle<BaseField>, denominators: Mle<SecureField>) -> Self {
        assert_eq!(numerators.num_variables(), denominators.num_variables());
        Self::Multiplicities {
            numerators,
            denominators,
        }
    }

    pub fn new_singles(denominators: Mle<SecureField>) -> Self {
        Self::Singles { denominators }
    }

    fn len(&self) -> usize {
        match self {
            Self::Singles { denominators }
            | Self::Multiplicities { denominators, .. }
            | Self::Generic { denominators, .. } => denominators.len(),
        }
    }
}

impl GkrLayer for LogupTrace {
    type SumcheckOracle = LogupOracle;

    fn next(&self) -> Option<Self> {
        if self.len() == 2 {
            return None;
        }

        let mut next_numerators = Vec::new();
        let mut next_denominators = Vec::new();

        match self {
            Self::Singles { denominators } => {
                // 1/d0 + 1/d1 = (d0 + d1)/(d0 * d1)
                for &[d0, d1] in denominators.array_chunks() {
                    let a = Fraction::new(BaseField::one(), d0);
                    let b = Fraction::new(BaseField::one(), d1);
                    let res = a + b;
                    next_numerators.push(res.numerator);
                    next_denominators.push(res.denominator);
                }
            }
            Self::Multiplicities {
                numerators,
                denominators,
            } => {
                for (&[n0, n1], &[d0, d1]) in
                    zip(numerators.array_chunks(), denominators.array_chunks())
                {
                    let a = Fraction::new(n0, d0);
                    let b = Fraction::new(n1, d1);
                    let res = a + b;
                    next_numerators.push(res.numerator);
                    next_denominators.push(res.denominator);
                }
            }
            Self::Generic {
                numerators,
                denominators,
            } => {
                for (&[n0, n1], &[d0, d1]) in
                    zip(numerators.array_chunks(), denominators.array_chunks())
                {
                    let a = Fraction::new(n0, d0);
                    let b = Fraction::new(n1, d1);
                    let res = a + b;
                    next_numerators.push(res.numerator);
                    next_denominators.push(res.denominator);
                }
            }
        }

        println!("yo: {}", next_numerators.len());
        println!("yo: {}", next_denominators.len());

        Some(Self::Generic {
            numerators: Mle::new(next_numerators),
            denominators: Mle::new(next_denominators),
        })
    }

    fn into_sumcheck_oracle(
        self,
        lambda: SecureField,
        layer_assignment: &[SecureField],
    ) -> LogupOracle {
        LogupOracle {
            trace: self,
            eq_evals: gen_eq_evals(layer_assignment),
            num_variables: layer_assignment.len(),
            z: layer_assignment.to_vec(),
            lambda,
        }
    }

    fn into_trace(self) -> MleTrace<SecureField> {
        let columns = match self {
            Self::Generic {
                numerators,
                denominators,
            } => vec![numerators, denominators],
            // Don't need to implement these. `into_trace` is only called on the output layer which
            // should always be `Generic` (`Multiplicities` and `Singles` are for top layers only).
            Self::Multiplicities { .. } | Self::Singles { .. } => unimplemented!(),
        };

        MleTrace::new(columns)
    }
}

/// Sumcheck oracle for a logup+GKR layer.
pub struct LogupOracle {
    /// Multi-linear extension of the numerators and denominators
    trace: LogupTrace,
    /// Evaluations of `eq_z(x_1, ..., x_n)` (see [`gen_eq_evals`] docs).
    eq_evals: Vec<SecureField>,
    /// The random point sampled during the GKR protocol for the sumcheck.
    // TODO: Better docs.
    z: Vec<SecureField>,
    /// Random value used to combine two sum-checks, for numerators and denominators, into one.
    lambda: SecureField,
    num_variables: usize,
}

impl SumcheckOracle for LogupOracle {
    fn num_variables(&self) -> usize {
        1 << self.z.len()
    }

    fn univariate_sum(&self, claim: SecureField) -> Polynomial<SecureField> {
        let now = Instant::now();

        let zero = SecureField::zero();
        let one = SecureField::one();

        let lambda = self.lambda;

        let n_terms = 1 << self.num_variables;

        let eval_at_0 = SecureField::zero();
        let eval_at_2 = SecureField::zero();

        match self.trace {
            LogupTrace::Singles { denominators } => {
                let (pairs, _) = denominators.as_chunks();

                for i in 0..n_terms {
                    // The circuit sums neighbors.
                    let [lhs0, lhs1] = pairs[i];

                    let fraction0 = { Fraction::new(lhs0 + lhs1, lhs0 * lhs1) };

                    let fraction2 = {
                        let [rhs0, rhs1] = pairs[n_terms + i];
                        let d0 = rhs0.double() - lhs0;
                        let d1 = rhs1.double() - lhs1;
                        Fraction::new(d0 + d1, d0 * d1)
                    };

                    let eq_eval = self.eq_evals[i];
                    eval_at_0 += eq_eval * (fraction0.numerator + lambda * fraction0.denominator);
                    eval_at_2 += eq_eval * (fraction2.numerator + lambda * fraction2.denominator);
                }
            }
            LogupTrace::Multiplicities {
                numerators,
                denominators,
            } => todo!(),
            LogupTrace::Generic {
                numerators,
                denominators,
            } => todo!(),
        }

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

        let n_terms = 1 << self.num_variables;
        let (p_lhs_pairs, p_rhs_pairs) = self.p.as_chunks().0.split_at(n_terms);
        let (q_lhs_pairs, q_rhs_pairs) = self.q.as_chunks().0.split_at(n_terms);

        // println!("n_TERMS: {n_terms}");
        #[allow(unused_mut)]
        let [y0, mut y2] = zip(
            &self.c,
            zip(zip(p_lhs_pairs, p_rhs_pairs), zip(q_lhs_pairs, q_rhs_pairs)),
        )
        .fold(
            [zero; 2],
            |acc,
             (
                &c,
                ((&[p0_lhs, p1_lhs], &[p0_rhs, p1_rhs]), (&[q0_lhs, q1_lhs], &[q0_rhs, q1_rhs])),
            )| {
                // eval at 0:
                let eval0 = {
                    let a = Fraction::new(p0_lhs, q0_lhs);
                    let b = Fraction::new(p1_lhs, q1_lhs);
                    unsafe { SUMCHECK_ADDS += 1 };
                    unsafe { SUMCHECK_MULTS += 3 };
                    // let res = a + b;
                    // res.numerator + self.lambda * res.denominator
                    a + b
                };

                // 2(q0_lhs + q1_lhs) - (q0_rhs + q1_rhs)

                // 1/q0_lhs + 1/q1_lhs = (q0_lhs + q1_lhs) / (q0_lhs * q1_lhs)
                // 1/(2*q0_lhs - q0_rhs) + 1/(2*q1_lhs - q1_rhs) =
                //   2(q0_lhs + q1_lhs) - (q0_rhs + q1_rhs) /
                //
                // (2*q0_lhs - q0_rhs)(2*q1_lhs - q1_rhs) = 4*q0_lhs*q1_lhs - 2*q0_lhs*q1_rhs -
                // 2*q1_lhs*q0_rhs + q0_rhs*q1_rhs (q0)

                // eval at -1:
                let evaln1 = {
                    let p0_eval = p0_lhs.double() - p0_rhs;
                    let p1_eval = p1_lhs.double() - p1_rhs;
                    let q0_eval = q0_lhs.double() - q0_rhs;
                    let q1_eval = q1_lhs.double() - q1_rhs;
                    let a = Fraction::new(p0_eval, q0_eval);
                    let b = Fraction::new(p1_eval, q1_eval);
                    unsafe { SUMCHECK_ADDS += 9 };
                    unsafe { SUMCHECK_MULTS += 3 };
                    // let res = a + b;
                    // res.numerator + self.lambda * res.denominator
                    a + b
                };

                unsafe { SUMCHECK_ADDS += 4 };
                unsafe { SUMCHECK_MULTS += 4 };
                [
                    c * (eval0.numerator + self.lambda * eval0.denominator) + acc[0],
                    c * (evaln1.numerator + self.lambda * evaln1.denominator) + acc[1],
                ]
            },
        );

        let x0 = BaseField::zero();
        let x1 = BaseField::one();
        let x2 = -BaseField::one();
        let x3 = BaseField::from(2);

        unsafe { UNIVARIATE_SUM_DUR += now.elapsed() }

        let y1 = SecureField::from(self.claim) - y0;

        let pre_shift_poly = Polynomial::<SecureField>::interpolate_lagrange(
            &[x0.into(), x1.into(), x2.into()],
            &[
                y0.into(),
                SecureField::from(y1 * eq_shift1.inverse()),
                y2.into(),
            ],
        );

        y2 *= eq_shift_neg1;
        let y3 = SecureField::from(eq_shift2) * pre_shift_poly.eval(x3.into());

        Polynomial::interpolate_lagrange(
            &[x0.into(), x1.into(), x2.into(), x3.into()],
            &[y0.into(), y1.into(), y2.into(), y3],
        )
    }

    fn fix_first(self, challenge: SecureField) -> Self {
        let trace = match self.trace {
            LogupTrace::Generic {
                mut numerators,
                mut denominators,
            } => {
                numerators.fix_first_mut(challenge);
                denominators.fix_first_mut(challenge);
                LogupTrace::Generic {
                    numerators,
                    denominators,
                }
            }
            LogupTrace::Multiplicities {
                numerators,
                mut denominators,
            } => {
                let numerators = numerators.fix_first(challenge);
                denominators.fix_first_mut(challenge);
                LogupTrace::Generic {
                    numerators,
                    denominators,
                }
            }
            LogupTrace::Singles { mut denominators } => {
                denominators.fix_first_mut(challenge);
                LogupTrace::Singles { denominators }
            }
        };

        Self {
            trace,
            eq_evals: self.eq_evals,
            z: self.z,
            lambda: self.lambda,
            num_variables: self.num_variables - 1,
        }
    }
}

impl GkrSumcheckOracle for LogupOracle {
    fn into_inputs(self) -> MleTrace<SecureField> {
        self.trace.into_trace()
    }
}

/// Evaluates [`eq(x, y)`] for all values of `x` in `{0, 1}^n`.
///
/// Algorithm from [Thaler13] Section 5.4.1.
/// Generates evaluations in `O(2^n)` vs naive `O(n * 2^n)`.
///
/// [`eq(x, y)`]: crate::gkr_generic::eq
/// [Thaler13]: https://eprint.iacr.org/2013/351.pdf
#[allow(dead_code)]
fn gen_eq_evals(y: &[SecureField]) -> Vec<SecureField> {
    match y {
        [] => vec![],
        &[y1] => vec![SecureField::one() - y1, y1],
        &[yj, ref y @ ..] => {
            let c = gen_eq_evals(y);
            let yj_bar = SecureField::one() - yj;
            // TODO: This can be reduced to single mult and addition.
            let c0 = c.iter().map(|&v| yj_bar * v);
            let c1 = c.iter().map(|&v| yj * v);
            Iterator::chain(c0, c1).collect()
        }
    }
}

struct Fraction<F> {
    numerator: F,
    denominator: SecureField,
}

impl<F> Fraction<F> {
    fn new(numerator: F, denominator: SecureField) -> Self {
        Self {
            numerator,
            denominator,
        }
    }
}

impl<F> Add for Fraction<F>
where
    F: ExtensionOf<BaseField>,
    SecureField: ExtensionOf<F> + Field,
{
    type Output = Fraction<SecureField>;

    fn add(self, rhs: Self) -> Fraction<SecureField> {
        if self.numerator.is_one() && rhs.numerator.is_one() {
            Fraction {
                numerator: self.denominator + rhs.denominator,
                denominator: self.denominator * rhs.denominator,
            }
        } else {
            Fraction {
                numerator: rhs.denominator * self.numerator + self.denominator * rhs.numerator,
                denominator: self.denominator * rhs.denominator,
            }
        }
    }
}
