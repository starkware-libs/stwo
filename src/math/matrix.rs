use crate::core::fields::m31::BaseField;
use crate::core::fields::ExtensionOf;

pub struct CircularMatrix<F: ExtensionOf<BaseField>> {
    size: usize,
    unique_values: Vec<F>,
}

pub trait Matrix<F: ExtensionOf<BaseField>> {
    fn get_at(&self, i: isize, j: isize) -> F;
    fn mul(&self, v: Vec<F>) -> Vec<F>;
}

/// A square matrix of size n, with the following property:
/// M(i, j) is equal to unique_values at index '(j - i) % unique_values.len()'.
/// Assumes that unique_values.len() >= n.
impl<F: ExtensionOf<BaseField>> CircularMatrix<F> {
    #[allow(dead_code)]
    fn new(n: usize, unique_values: Vec<F>) -> Self {
        Self {
            size: n,
            unique_values,
        }
    }
}

impl<F: ExtensionOf<BaseField>> Matrix<F> for CircularMatrix<F> {
    fn get_at(&self, i: isize, j: isize) -> F {
        let mut index = j - i;

        // Matrix is Circular.
        if index < 0 {
            index += self.unique_values.len() as isize;
        }
        self.unique_values[index as usize]
    }

    fn mul(&self, v: Vec<F>) -> Vec<F> {
        let mut result = vec![F::zero(); self.size];
        for (i, result_i) in result.iter_mut().enumerate() {
            for (j, v_j) in v.iter().enumerate().take(v.len()) {
                *result_i += self.get_at(i as isize, j as isize) * *v_j;
            }
        }
        result
    }
}

#[cfg(test)]
mod tests {
    use crate::core::fields::m31::M31;
    use crate::core::fields::qm31::QM31;
    use crate::m31;
    use crate::math::matrix::{CircularMatrix, Matrix};

    #[test]
    fn test_get_index() {
        let n_unique_values = 32;
        let matrix = CircularMatrix::<QM31>::new(
            24,
            (0..n_unique_values)
                .map(|x| QM31::from(m31!(x)))
                .collect::<Vec<_>>(),
        );

        assert_eq!(matrix.get_at(0, 0), QM31::from(m31!(0)));
        assert_eq!(matrix.get_at(1, 3), QM31::from(m31!(2)));
        assert_eq!(matrix.get_at(3, 1), QM31::from(m31!(n_unique_values - 2)));
    }

    #[test]
    fn test_mul() {
        let matrix = CircularMatrix::<M31>::new(3, (1..6).map(|x| m31!(x)).collect::<Vec<_>>());
        let state = (1..4).map(|x| m31!(x)).collect::<Vec<_>>();
        let expected_result = [
            14, // 1 * 1 + 2 * 2 + 3 * 3
            13, // 5 * 1 + 1 * 2 + 2 * 3
            17, // 4 * 1 + 5 * 2 + 1 * 3
        ]
        .iter()
        .map(|x| m31!(*x))
        .collect::<Vec<_>>();
        let result = matrix.mul(state);

        assert_eq!(result, expected_result);
    }
}
