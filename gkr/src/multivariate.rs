use num_traits::{One, Zero};

use crate::q31::FastSecureField as SecureField;

pub trait MultivariatePolynomial {
    fn eval(&self, assigments: &[SecureField]) -> SecureField;

    fn num_variables(&self) -> usize;
}

pub fn from_const_fn<const N: usize, F: Fn([SecureField; N]) -> SecureField>(
    f: F,
) -> FromFn<impl Fn(&[SecureField]) -> SecureField> {
    FromFn {
        f: move |assignment| f(assignment.try_into().unwrap()),
        num_variables: N,
    }
}

pub fn from_fn<F: Fn(&[SecureField]) -> SecureField>(num_variables: usize, f: F) -> FromFn<F> {
    FromFn { f, num_variables }
}

pub struct FromFn<F: Fn(&[SecureField]) -> SecureField> {
    f: F,
    num_variables: usize,
}

impl<F: Fn(&[SecureField]) -> SecureField> MultivariatePolynomial for FromFn<F> {
    fn eval(&self, assigments: &[SecureField]) -> SecureField {
        (self.f)(assigments)
    }

    fn num_variables(&self) -> usize {
        self.num_variables
    }
}

pub fn hypercube_sum(
    dimension: usize,
    mut f: impl FnMut(&[SecureField]) -> SecureField,
) -> SecureField {
    if dimension == 0 {
        return (f)(&[]);
    }

    let mut res = SecureField::zero();

    let mut assignment = vec![SecureField::zero(); dimension];

    for point in 0..1 << dimension {
        assignment
            .iter_mut()
            .enumerate()
            .for_each(|(i, assignment)| {
                *assignment = if (point >> i) & 1 == 1 {
                    SecureField::one()
                } else {
                    SecureField::zero()
                };
            });

        res += (f)(&assignment);
    }

    res
}
