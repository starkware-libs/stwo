#![feature(array_windows, exact_size_is_empty, array_chunks)]
#![allow(dead_code, unused_variables, unused_imports)]

pub mod gkr;
mod multivariate;
pub mod sumcheck;
mod utils;

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
        let res = partially_verify(&proof, &mut test_channel());
        println!("res: {:?}", res)
    }

    fn test_channel() -> Blake2sChannel {
        let seed = Blake2sHasher::hash(&[]);
        Blake2sChannel::new(seed)
    }
}
