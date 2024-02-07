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

    const V0: BaseField = BaseField::from_u32_unchecked(5);
    const V1: BaseField = BaseField::from_u32_unchecked(1);
    const V2: BaseField = BaseField::from_u32_unchecked(2);
    const V3: BaseField = BaseField::from_u32_unchecked(2);
    const V4: BaseField = BaseField::from_u32_unchecked(5);
    const V5: BaseField = BaseField::from_u32_unchecked(19);
    const V6: BaseField = BaseField::from_u32_unchecked(1);
    const V7: BaseField = BaseField::from_u32_unchecked(3);
    const V8: BaseField = BaseField::from_u32_unchecked(3);
    const V9: BaseField = BaseField::from_u32_unchecked(3);
    const V10: BaseField = BaseField::from_u32_unchecked(3);
    const V11: BaseField = BaseField::from_u32_unchecked(3);
    const V12: BaseField = BaseField::from_u32_unchecked(3);
    const V13: BaseField = BaseField::from_u32_unchecked(3);
    const V14: BaseField = BaseField::from_u32_unchecked(3);
    const V15: BaseField = BaseField::from_u32_unchecked(3);

    fn g4([x1, x2, x3, x4]: [BaseField; 4]) -> BaseField {
        let one = BaseField::one();

        eq(&[x1, x2, x3, x4], &[-one, -one, -one, -one]) * Self::V0
            + eq(&[x1, x2, x3, x4], &[-one, -one, -one, one]) * Self::V1
            + eq(&[x1, x2, x3, x4], &[-one, -one, one, -one]) * Self::V2
            + eq(&[x1, x2, x3, x4], &[-one, -one, one, one]) * Self::V3
            + eq(&[x1, x2, x3, x4], &[-one, one, -one, -one]) * Self::V4
            + eq(&[x1, x2, x3, x4], &[-one, one, -one, one]) * Self::V5
            + eq(&[x1, x2, x3, x4], &[-one, one, one, -one]) * Self::V6
            + eq(&[x1, x2, x3, x4], &[-one, one, one, one]) * Self::V7
            + eq(&[x1, x2, x3, x4], &[one, -one, -one, -one]) * Self::V8
            + eq(&[x1, x2, x3, x4], &[one, -one, -one, one]) * Self::V9
            + eq(&[x1, x2, x3, x4], &[one, -one, one, -one]) * Self::V10
            + eq(&[x1, x2, x3, x4], &[one, -one, one, one]) * Self::V11
            + eq(&[x1, x2, x3, x4], &[one, one, -one, -one]) * Self::V12
            + eq(&[x1, x2, x3, x4], &[one, one, -one, one]) * Self::V13
            + eq(&[x1, x2, x3, x4], &[one, one, one, -one]) * Self::V14
            + eq(&[x1, x2, x3, x4], &[one, one, one, one]) * Self::V15
    }

    fn g3([x1, x2, x3]: [BaseField; 3]) -> BaseField {
        let one = BaseField::one();

        eq(&[x1, x2, x3], &[-one, -one, -one])
            * Self::g4([-one, -one, -one, -one])
            * Self::g4([-one, -one, -one, one])
            + eq(&[x1, x2, x3], &[-one, -one, one])
                * Self::g4([-one, -one, one, -one])
                * Self::g4([-one, -one, one, one])
            + eq(&[x1, x2, x3], &[-one, one, -one])
                * Self::g4([-one, one, -one, -one])
                * Self::g4([-one, one, -one, one])
            + eq(&[x1, x2, x3], &[-one, one, one])
                * Self::g4([-one, one, one, -one])
                * Self::g4([-one, one, one, one])
            + eq(&[x1, x2, x3], &[one, -one, -one])
                * Self::g4([one, -one, -one, -one])
                * Self::g4([one, -one, -one, one])
            + eq(&[x1, x2, x3], &[one, -one, one])
                * Self::g4([one, -one, one, -one])
                * Self::g4([one, -one, one, one])
            + eq(&[x1, x2, x3], &[one, one, -one])
                * Self::g4([one, one, -one, -one])
                * Self::g4([one, one, -one, one])
            + eq(&[x1, x2, x3], &[one, one, one])
                * Self::g4([one, one, one, -one])
                * Self::g4([one, one, one, one])
    }

    fn g2([x1, x2]: [BaseField; 2]) -> BaseField {
        let one = BaseField::one();

        eq(&[x1, x2], &[-one, -one]) * Self::g3([-one, -one, -one]) * Self::g3([-one, -one, one])
            + eq(&[x1, x2], &[-one, one]) * Self::g3([-one, one, -one]) * Self::g3([-one, one, one])
            + eq(&[x1, x2], &[one, -one]) * Self::g3([one, -one, -one]) * Self::g3([one, -one, one])
            + eq(&[x1, x2], &[one, one]) * Self::g3([one, one, -one]) * Self::g3([one, one, one])
    }

    // g1(x) = sum_{}^{} eq(x, -1) * g2(-1, -1) * g2(-1, 1)
    fn g1([x1]: [BaseField; 1]) -> BaseField {
        let one = BaseField::one();

        eq(&[x1], &[-one]) * Self::g2([-one, -one]) * Self::g2([-one, one])
            + eq(&[x1], &[one]) * Self::g2([one, -one]) * Self::g2([one, one])
    }

    // g0 = g1(-1) * g1(1)
    fn g0() -> BaseField {
        let one = BaseField::one();
        Self::g1([-one]) * Self::g1([one])
    }

    fn prove_round<const DIMENSION: usize, P: Fn([BaseField; DIMENSION]) -> BaseField>(
        polynomial: P,
        challenges: &[BaseField],
    ) -> Polynomial<BaseField> {
        assert_eq!(DIMENSION - 1, challenges.len());

        let mut assignments = [BaseField::zero(); DIMENSION];

        // Assign challenges.
        zip(&mut assignments, challenges).for_each(|(a, &c)| *a = c);

        let x0 = BaseField::zero();
        let x1 = BaseField::one();
        let x2 = -BaseField::one();
        let x3 = BaseField::from(2);
        let x4 = BaseField::from(4);
        let x5 = BaseField::from(8);

        assignments[DIMENSION - 1] = x0;
        let y0 = polynomial(assignments);

        assignments[DIMENSION - 1] = x1;
        let y1 = polynomial(assignments);

        assignments[DIMENSION - 1] = x2;
        let y2 = polynomial(assignments);

        assignments[DIMENSION - 1] = x3;
        let y3 = polynomial(assignments);

        assignments[DIMENSION - 1] = x4;
        let y4 = polynomial(assignments);

        assignments[DIMENSION - 1] = x5;
        let y5 = polynomial(assignments);

        let res =
            Polynomial::interpolate_lagrange(&[x0, x1, x2, x3, x4, x5], &[y0, y1, y2, y3, y4, y5]);
        assert_eq!(res.eval(x0), y0);
        assert_eq!(res.eval(x1), y1);
        assert_eq!(res.eval(x2), y2);
        assert_eq!(res.eval(x3), y3);
        assert_eq!(res.eval(x4), y4);
        assert_eq!(res.eval(x5), y5);

        res
    }

    fn verify() -> bool {
        let zero = BaseField::zero();
        let one = BaseField::one();

        // Prover sends verifier the claimed circuit evaluation.
        let claimed_eval = Self::g0();
        assert_eq!(
            claimed_eval,
            Self::V0
                * Self::V1
                * Self::V2
                * Self::V3
                * Self::V4
                * Self::V5
                * Self::V6
                * Self::V7
                * Self::V8
                * Self::V9
                * Self::V10
                * Self::V11
                * Self::V12
                * Self::V13
                * Self::V14
                * Self::V15
        );

        let one = BaseField::one();

        // Verifier and prover engage in GKR protocol.
        let mut rng = thread_rng();

        // 1st round (round 0).
        let s1 = Self::prove_round(Self::g1, &[]);
        // Check `deg(s1) <= deg_x1(g1(x1))`
        if s1.degree() > 1 {
            return false;
        }
        // s1 sumcheck
        {
            // g1(r) = sum_{x in {-1, 1}} eq(r, x) * g2(x, -1) * g2(x, 1)
            // g1' = sum_{x in {-1, 1}} eq(r, x) * g2(x, -1) * g2(x, 1)
            //
            // g2(r1, r2) = sum_{x in {-1, 1}^2} eq([r1, r2], x) * g3(x, -1) * g3(x, 1)
        }
        // Check against claim
        if claimed_eval != s1.eval(-one) * s1.eval(one) {
            return false;
        }
        let s1_sumcheck = s1.eval(-one) + s1.eval(one);

        // 2nd round (round 1).
        let r1 = rng.gen();
        let s2 = Self::prove_round(Self::g2, &[r1]);
        // // Check `deg(s2) <= deg_x2(g(x1, x2, x3))`
        // if s2.degree() > 1 {
        //     return false;
        // }
        // Check against round 1
        println!("yo: {:?}", s2);
        println!("s1 sumcheck: {}", s1_sumcheck);
        println!("s1 sumcheck: {}", s1_sumcheck);
        println!("yo: {}", s1.eval(r1));
        // println!("yo: {}", s2.eval(-one) + s2.eval(one));
        println!("yo: {}", s2.eval(-one) + s2.eval(one));
        println!("yo: {}", eq(&[r1], &[-one]) + eq(&[r1], &[one]));
        println!("yo: {}", eq(&[r1], &[one]));
        if s1.eval(r1) != s2.eval(-one) * s2.eval(one) {
            return false;
        }
        println!("FOOOO");

        // 3rd round (round 2).
        let r2 = rng.gen();
        let s3 = Self::prove_round(Self::g3, &[r1, r2]);
        // // Check `deg(s3) <= deg_x3(g(x1, x2, x3))`
        // if s3.degree() > 1 {
        //     return false;
        // }
        // Check against round 2
        if s2.eval(r2) != s3.eval(zero) * s3.eval(one) {
            return false;
        }

        // // Checks `s3` against oracle query to `g`
        // let r3 = rng.gen();
        // if s3.eval(r3) != Self::g([r1, r2, r3]) {
        //     return false;
        // }

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

/// Evaluates the lagrange kernel of the boolean hypercube.
///
/// When y is an elements of `{+-1}^n` then this function evaluates the Lagrange polynomial which is
/// the unique multilinear polynomial equal to 1 if `x = y` and equal to 0 whenever x is an element
/// of `{+-1}^n`.
///
/// From: <https://eprint.iacr.org/2023/1284.pdf>.
fn eq(x_assignments: &[BaseField], y_assignments: &[BaseField]) -> BaseField {
    assert_eq!(x_assignments.len(), y_assignments.len());

    let n = x_assignments.len();
    let norm = BaseField::from_u32_unchecked(2u32.pow(n as u32));

    zip(x_assignments, y_assignments)
        .map(|(&x, &y)| BaseField::one() + x * y)
        .product::<BaseField>()
        / norm
}
