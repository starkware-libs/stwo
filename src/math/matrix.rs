use crate::core::fields::m31::BaseField;
use crate::core::fields::ExtensionOf;

pub trait Matrix<F: ExtensionOf<BaseField>> {
    fn get_at(&self, i: usize, j: usize) -> F;
    fn size(&self) -> usize;

    fn mul(&self, v: Vec<F>) -> Vec<F> {
        let mut result = vec![F::zero(); self.size()];
        for (i, result_i) in result.iter_mut().enumerate() {
            for (j, v_j) in v.iter().enumerate().take(v.len()) {
                *result_i += self.get_at(i, j) * *v_j;
            }
        }
        result
    }
}

pub struct SquareMatrix<F: ExtensionOf<BaseField>> {
    size: usize,
    values: Vec<Vec<F>>,
}

impl<F: ExtensionOf<BaseField>> SquareMatrix<F> {
    pub fn new(size: usize, values: Vec<Vec<F>>) -> Self {
        assert_eq!(size, values.len());
        for row in values.iter() {
            assert_eq!(size, row.len());
        }
        Self { size, values }
    }
}

impl<F: ExtensionOf<BaseField>> Matrix<F> for SquareMatrix<F> {
    fn get_at(&self, i: usize, j: usize) -> F {
        self.values[i][j]
    }

    fn size(&self) -> usize {
        self.size
    }
}

#[cfg(test)]
mod tests {
    use crate::core::fields::m31::M31;
    use crate::m31;
    use crate::math::matrix::{Matrix, SquareMatrix};

    #[test]
    fn test_matrix_multiplication() {
        let matrix = SquareMatrix::<M31>::new(
            3,
            vec![
                vec![m31!(1), m31!(1), m31!(1)],
                vec![m31!(2), m31!(2), m31!(2)],
                vec![m31!(3), m31!(3), m31!(3)],
            ],
        );
        let vector = (1..4).map(|x| m31!(x)).collect::<Vec<_>>();
        let expected_result = [
            m31!(6),  // 1 * (1 + 2 + 3)
            m31!(12), // 2 * (1 + 2 + 3)
            m31!(18), // 3 * (1 + 2 + 3)
        ];

        let result = matrix.mul(vector);

        assert_eq!(result, expected_result);
    }
}
