use crate::core::fields::m31::BaseField;
use crate::core::fields::ExtensionOf;

pub trait SquareMatrix<F: ExtensionOf<BaseField> + Copy, const N: usize> {
    fn get_at(&self, i: usize, j: usize) -> F;
    fn mul(&self, v: [F; N]) -> [F; N] {
        (0..N)
            .map(|i| {
                (0..N)
                    .map(|j| self.get_at(i, j) * v[j])
                    .fold(F::zero(), |acc, x| acc + x)
            })
            .collect::<Vec<_>>()
            .try_into()
            .unwrap()
    }
}

pub struct RowMajorMatrix<F: ExtensionOf<BaseField>, const N: usize> {
    values: [[F; N]; N],
}

impl<F: ExtensionOf<BaseField> + Copy, const N: usize> RowMajorMatrix<F, N> {
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

impl<F: ExtensionOf<BaseField> + Copy, const N: usize> SquareMatrix<F, N> for RowMajorMatrix<F, N> {
    fn get_at(&self, i: usize, j: usize) -> F {
        self.values[i][j]
    }
}

#[cfg(test)]
mod tests {
    use crate::core::fields::m31::M31;
    use crate::m31;
    use crate::math::matrix::{RowMajorMatrix, SquareMatrix};

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
}
