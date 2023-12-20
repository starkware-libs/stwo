use crate::core::fields::m31::BaseField;
use crate::core::fields::ExtensionOf;

pub trait SquareMatrix<F: ExtensionOf<BaseField>, const N: usize> {
    fn get_at(&self, i: usize, j: usize) -> F;
    fn mul(&self, v: [F; N]) -> Vec<F> {
        (0..N)
            .map(|i| {
                (0..N)
                    .map(|j| self.get_at(i, j) * v[j])
                    .fold(F::zero(), |acc, x| acc + x)
            })
            .collect()
    }
}

pub struct RowMajorMatrix<F: ExtensionOf<BaseField>, const N: usize> {
    values: [[F; N]; N],
}

impl<F: ExtensionOf<BaseField>, const N: usize> RowMajorMatrix<F, N> {
    pub fn new(values: Vec<F>) -> Self {
        assert_eq!(values.len(), N * N);
        Self {
            values: values
                .chunks(N)
                .map(|chunk| chunk.try_into().unwrap())
                .collect::<Vec<_>>()
                .try_into()
                .unwrap(),
        }
    }
}

impl<F: ExtensionOf<BaseField>, const N: usize> SquareMatrix<F, N> for RowMajorMatrix<F, N> {
    fn get_at(&self, i: usize, j: usize) -> F {
        self.values[i][j]
    }
}

/// A square matrix of size N, with the following property:
/// M(i, j) is equal to unique_values at index '(j - i) % unique_values.len()'.
pub struct CircularMatrix<F: ExtensionOf<BaseField>, const N: usize> {
    unique_values: Vec<F>,
}

impl<F: ExtensionOf<BaseField>, const N: usize> CircularMatrix<F, N> {
    pub fn new(values: Vec<F>) -> Self {
        assert!(values.len() >= N);
        Self {
            unique_values: values,
        }
    }
}

impl<F: ExtensionOf<BaseField>, const N: usize> SquareMatrix<F, N> for CircularMatrix<F, N> {
    fn get_at(&self, i: usize, j: usize) -> F {
        let mut index = j as isize - i as isize;

        if index >= 0 {
            return self.unique_values[index as usize];
        }

        // Matrix is Circular.
        index += self.unique_values.len() as isize;
        self.unique_values[index as usize]
    }
}

#[cfg(test)]
mod tests {
    use crate::core::fields::m31::M31;
    use crate::core::fields::qm31::QM31;
    use crate::m31;
    use crate::math::matrix::{CircularMatrix, RowMajorMatrix, SquareMatrix};

    #[test]
    fn test_matrix_multiplication() {
        let matrix = RowMajorMatrix::<M31, 3>::new((0..9).map(|x| m31!(x + 1)).collect::<Vec<_>>());
        let vector = (0..3)
            .map(|x| m31!(x + 1))
            .collect::<Vec<_>>()
            .try_into()
            .unwrap();
        let expected_result = [
            m31!(14), // 1 * 1 + 2 * 2 + 3 * 3
            m31!(32), // 4 * 1 + 5 * 2 + 6 * 3
            m31!(50), // 7 * 1 + 8 * 2 + 9 * 3
        ];

        let result = matrix.mul(vector);

        assert_eq!(result, expected_result);
    }

    #[test]
    fn test_circular_matrix() {
        let n_unique_values = 32;
        let matrix = CircularMatrix::<QM31, 24>::new(
            (0..n_unique_values)
                .map(|x| QM31::from(m31!(x)))
                .collect::<Vec<_>>(),
        );

        assert_eq!(matrix.get_at(0, 0), QM31::from(m31!(0)));
        assert_eq!(matrix.get_at(1, 3), QM31::from(m31!(2)));
        assert_eq!(matrix.get_at(3, 1), QM31::from(m31!(n_unique_values - 2)));
    }
}
