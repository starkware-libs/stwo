#![feature(array_windows, exact_size_is_empty, array_chunks)]
#![allow(dead_code, unused_variables, unused_imports)]

use std::array;
use std::fmt::{Display, Formatter};
use std::iter::{from_fn, repeat, successors, zip, Sum};
use std::ops::{Add, Deref, Mul, Neg, Sub};
use std::process::Output;
use std::time::Instant;

use num_traits::{One, Zero};
use prover_research::core::channel;
use prover_research::core::fields::m31::BaseField;
use prover_research::core::fields::qm31::ExtensionField;
use prover_research::core::fields::Field;
use rand::{thread_rng, Rng};

fn main() {
    println!("running sumcheck: {}", SumCheck::verify());
}

struct SumCheck;

impl SumCheck {
    const NUM_ROUNDS: usize = 3;

    /// Sum over the boolean hypercube should be `12`.
    fn g([x1, x2, x3]: [BaseField; 3]) -> BaseField {
        x1.pow(3).double() + x1 * x3 + x2 * x3
    }

    fn prove_round(round: usize, challenges: &[BaseField]) -> Polynomial<BaseField> {
        assert!(round < Self::NUM_ROUNDS);
        assert_eq!(round, challenges.len());

        let mut assigmnets = [BaseField::zero(); 3];

        // Assign challenges.
        zip(&mut assigmnets, challenges).for_each(|(a, &c)| *a = c);

        let hypercube_dim = Self::NUM_ROUNDS - round - 1;

        let x0 = BaseField::zero();
        let x1 = BaseField::one();
        let x2 = -BaseField::one();
        let x3 = BaseField::one() + BaseField::one();

        let mut y0 = BaseField::zero();
        let mut y1 = BaseField::zero();
        let mut y2 = BaseField::zero();
        let mut y3 = BaseField::zero();

        // Iterate over this round's hypercube vertecies.
        for vertex in 0..1 << hypercube_dim {
            // Make variable assignments.
            for (bit, assigment) in assigmnets[round + 1..].iter_mut().rev().enumerate() {
                *assigment = if vertex & (1 << bit) != 0 {
                    BaseField::one()
                } else {
                    BaseField::zero()
                };
            }

            assigmnets[round] = x0;
            y0 += Self::g(assigmnets);

            assigmnets[round] = x1;
            y1 += Self::g(assigmnets);

            assigmnets[round] = x2;
            y2 += Self::g(assigmnets);

            assigmnets[round] = x3;
            y3 += Self::g(assigmnets);
        }

        let res = Polynomial::interpolate_lagrange(&[x0, x1, x2, x3], &[y0, y1, y2, y3]);
        assert_eq!(res.eval(x0), y0);
        assert_eq!(res.eval(x1), y1);
        assert_eq!(res.eval(x2), y2);
        assert_eq!(res.eval(x3), y3);

        res
    }

    /// Implementation of 4.1 example: <https://people.cs.georgetown.edu/jthaler/ProofsArgsAndZK.pdf>
    fn verify() -> bool {
        let zero = BaseField::zero();
        let one = BaseField::one();

        // Prover sends verifier the claimed sum `H = 12`.
        let claimed_sum = BaseField::from(BaseField::from(12));

        // Verifier and prover engage in sumcheck protocol.
        let mut rng = thread_rng();

        // 1st round (round 0).
        let s1 = Self::prove_round(0, &[]);
        // Check `deg(s1) <= deg_x1(g(x1, x2, x3))`
        if s1.degree() > 3 {
            return false;
        }
        // Check against claim
        if claimed_sum != s1.eval(zero) + s1.eval(one) {
            return false;
        }

        // 2nd round (round 1).
        let r1 = rng.gen();
        let s2 = Self::prove_round(1, &[r1]);
        // Check `deg(s2) <= deg_x2(g(x1, x2, x3))`
        if s2.degree() > 1 {
            return false;
        }
        // Check against round 1
        if s1.eval(r1) != s2.eval(zero) + s2.eval(one) {
            return false;
        }

        // 3rd round (round 2).
        let r2 = rng.gen();
        let s3 = Self::prove_round(2, &[r1, r2]);
        // Check `deg(s3) <= deg_x3(g(x1, x2, x3))`
        if s3.degree() > 1 {
            return false;
        }
        // Check against round 2
        if s2.eval(r2) != s3.eval(zero) + s3.eval(one) {
            return false;
        }

        // Checks `s3` against oracle query to `g`
        let r3 = rng.gen();
        if s3.eval(r3) != Self::g([r1, r2, r3]) {
            return false;
        }

        true
    }
}

#[derive(Debug)]
pub struct Polynomial<F: Field>(Vec<F>);

impl<F: Field> Polynomial<F> {
    fn x() -> Self {
        Self(vec![F::zero(), F::one()])
    }

    fn zero() -> Self {
        Self(vec![])
    }

    fn is_zero(&self) -> bool {
        self.0.iter().all(|v| F::zero() == *v)
    }

    pub fn degree(&self) -> usize {
        let mut coeffs = self.0.iter().rev();
        let _ = (&mut coeffs).take_while(|v| v.is_zero());
        coeffs.len().saturating_sub(1)
    }

    pub fn new(coeffs: Vec<F>) -> Self {
        let mut polynomial = Self(coeffs);
        polynomial.truncate_leading_zeros();
        polynomial
    }

    fn truncate_leading_zeros(&mut self) {
        while self.0.last() == Some(&F::zero()) {
            self.0.pop();
        }
    }

    pub fn eval(&self, x: F) -> F {
        self.0
            .iter()
            .rfold(F::zero(), |acc, &coeff| acc * x + coeff)
    }

    // https://en.wikibooks.org/wiki/Algorithm_Implementation/Mathematics/Polynomial_interpolation
    pub fn interpolate_lagrange(xs: &[F], ys: &[F]) -> Self {
        let mut coeffs = Polynomial::new(vec![]);
        for (i, (&xi, &yi)) in zip(xs, ys).enumerate() {
            let mut tmp = yi;
            for (j, &xj) in xs.iter().enumerate() {
                if i != j {
                    tmp /= xi - xj;
                }
            }
            let mut term = Polynomial::new(vec![tmp]);
            for (j, &xj) in xs.iter().enumerate() {
                if i != j {
                    term = term * (Self::x() - Self::new(vec![xj]));
                }
            }
            coeffs = coeffs + term;
        }
        coeffs.truncate_leading_zeros();
        coeffs
    }
}

impl<F: Field> Deref for Polynomial<F> {
    type Target = [F];

    fn deref(&self) -> &[F] {
        &self.0
    }
}

impl<F: Field> Mul for Polynomial<F> {
    type Output = Self;

    fn mul(mut self, mut rhs: Self) -> Self {
        if self.is_zero() || rhs.is_zero() {
            return Polynomial::new(vec![]);
        }
        self.truncate_leading_zeros();
        rhs.truncate_leading_zeros();
        let mut res = vec![F::zero(); self.0.len() + rhs.0.len() - 1];
        for (i, coeff_a) in self.0.into_iter().enumerate() {
            for (j, &coeff_b) in rhs.0.iter().enumerate() {
                res[i + j] += coeff_a * coeff_b;
            }
        }
        Self::new(res)
    }
}

impl<F: Field> Add for Polynomial<F> {
    type Output = Self;

    fn add(self, rhs: Self) -> Self {
        let n = self.0.len().max(rhs.0.len());
        let mut res = Vec::new();
        for i in 0..n {
            res.push(match (self.0.get(i), rhs.0.get(i)) {
                (Some(&a), Some(&b)) => a + b,
                (Some(&a), None) | (None, Some(&a)) => a,
                _ => unreachable!(),
            })
        }
        Self(res)
    }
}

impl<F: Field> Sub for Polynomial<F> {
    type Output = Self;

    fn sub(self, rhs: Self) -> Self {
        self + (-rhs)
    }
}

impl<F: Field> Neg for Polynomial<F> {
    type Output = Self;

    fn neg(self) -> Self {
        Self(self.0.into_iter().map(|v| -v).collect())
    }
}
