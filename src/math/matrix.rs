use crate::core::fields::m31::BaseField;
use crate::core::fields::ExtensionOf;

pub trait SquareMatrix<F: ExtensionOf<BaseField>> {
    fn get_at(&self, i: usize, j: usize) -> F;
    fn size(&self) -> usize;

    fn mul(&self, v: Vec<F>) -> Vec<F> {
        assert_eq!(self.size(), v.len());
        (0..v.len())
            .map(|i| {
                (0..v.len())
                    .map(|j| self.get_at(i, j) * v[j])
                    .fold(F::zero(), |acc, x| acc + x)
            })
            .collect()
    }
}

pub struct RowMajorMatrix<F: ExtensionOf<BaseField>> {
    size: usize,
    values: Vec<F>,
}

impl<F: ExtensionOf<BaseField>> RowMajorMatrix<F> {
    pub fn new(size: usize, values: Vec<F>) -> Self {
        assert_eq!(size.pow(2), values.len());
        Self { size, values }
    }
}

impl<F: ExtensionOf<BaseField>> SquareMatrix<F> for RowMajorMatrix<F> {
    fn get_at(&self, i: usize, j: usize) -> F {
        self.values[i * self.size + j]
    }

    fn size(&self) -> usize {
        self.size
    }
}

#[cfg(test)]
mod tests {
    use crate::core::fields::m31::M31;
    use crate::m31;
    use crate::math::matrix::{RowMajorMatrix, SquareMatrix};

    #[test]
    fn test_matrix_multiplication() {
        let matrix = RowMajorMatrix::<M31>::new(
            3,
            vec![
                m31!(1),
                m31!(2),
                m31!(3),
                m31!(4),
                m31!(5),
                m31!(6),
                m31!(7),
                m31!(8),
                m31!(9),
            ],
        );
        let vector = (1..4).map(|x| m31!(x)).collect::<Vec<_>>();
        let expected_result = [
            m31!(14), // 1 * 1 + 2 * 2 + 3 * 3
            m31!(32), // 4 * 1 + 5 * 2 + 6 * 3
            m31!(50), // 7 * 1 + 8 * 2 + 9 * 3
        ];

        let result = matrix.mul(vector);

        assert_eq!(result, expected_result);
    }
}
