use std::fmt::Display;
use std::ops::{
    Add, AddAssign, Div, DivAssign, Mul, MulAssign, Neg, Rem, RemAssign, Sub, SubAssign,
};

use crate::core::fields::m31::M31;
use crate::{impl_extension_field, impl_field};

pub const P2: u64 = 4611686014132420609; // (2 ** 31 - 1) ** 2

/// Complex extension field of M31.
/// Equivalent to M31\[x\] over (x^2 + 1) as the irreducible polynomial.
/// Represented as (a, b) of a + bi.
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct CM31(M31, M31);

impl_field!(CM31, P2);
impl_extension_field!(CM31, M31);

impl CM31 {
    pub const fn from_u32_unchecked(a: u32, b: u32) -> CM31 {
        Self(M31::from_u32_unchecked(a), M31::from_u32_unchecked(b))
    }

    pub fn from_m31(a: M31, b: M31) -> CM31 {
        Self(a, b)
    }

    pub fn split(self) -> (M31, M31) {
        (self.0, self.1)
    }
}

impl Display for CM31 {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{} + {}i", self.0, self.1)
    }
}

impl Mul for CM31 {
    type Output = Self;

    fn mul(self, rhs: Self) -> Self::Output {
        // (a + bi) * (c + di) = (ac - bd) + (ad + bc)i.
        Self(
            self.0 * rhs.0 - self.1 * rhs.1,
            self.0 * rhs.1 + self.1 * rhs.0,
        )
    }
}

#[cfg(test)]
#[macro_export]
macro_rules! cm31 {
    ($m0:expr, $m1:expr) => {
        CM31::from_u32_unchecked($m0, $m1)
    };
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeSet;

    use nalgebra::SMatrix;
    use num_traits::Zero;

    use super::CM31;
    use crate::core::fields::m31::{M31, P};
    use crate::core::fields::Field;
    use crate::m31;

    #[test]
    fn test_ops() {
        let cm0 = cm31!(1, 2);
        let cm1 = cm31!(4, 5);
        let m = m31!(8);
        let cm = CM31::from(m);
        let cm0_x_cm1 = cm31!(P - 6, 13);

        assert_eq!(cm0 + cm1, cm31!(5, 7));
        assert_eq!(cm1 + m, cm1 + cm);
        assert_eq!(cm0 * cm1, cm0_x_cm1);
        assert_eq!(cm1 * m, cm1 * cm);
        assert_eq!(-cm0, cm31!(P - 1, P - 2));
        assert_eq!(cm0 - cm1, cm31!(P - 3, P - 3));
        assert_eq!(cm1 - m, cm1 - cm);
        assert_eq!(cm0_x_cm1 / cm1, cm31!(1, 2));
        assert_eq!(cm1 / m, cm1 / cm);
    }

    #[test]
    fn test_get_lambda() {
        let one = cm31!(1, 0);
        let omega = cm31!(2, 1268011823);
        let inv_i = cm31!(0, 1).inverse();
        let lambda = inv_i * (omega - one) / (omega + one);
        assert_eq!(lambda.0, m31!(1138498490));
        assert_eq!(lambda.1, m31!(0));
        assert_ne!(omega.pow(2_u128.pow(30)), one);
        assert_eq!(omega.pow(2_u128.pow(31)), one);

        let n_bits = 4;
        let omega_2_n = omega.pow(1 << (31 - n_bits - 1));
        println!("omega_2_n: {}", omega_2_n);
        assert_ne!(omega_2_n.pow(1 << n_bits), one);
        assert_eq!(omega_2_n.pow(1 << n_bits + 1), one);
    }

    #[test]
    fn test_mds_construction() {
        let one = m31!(1);
        let lambda = m31!(1138498490);
        // let lambda = m31!(321);

        const N_BITS: i32 = 5;
        let n = m31!(1 << N_BITS);
        let n_inv = n.inverse();

        let omega = cm31!(2, 1268011823);
        let omega_2n = omega.pow(1 << (31 - N_BITS - 1));

        let mut powers = vec![];
        let mut inv_powers = vec![];
        for i in 0..1 << N_BITS {
            powers.push(omega_2n.pow(1 + (2 * i) as u128));
            inv_powers.push(-omega_2n.pow(1 + (2 * i) as u128));
        }

        let mut mds = [[M31::zero(); 1 << N_BITS]; 1 << N_BITS];
        for i in 0..1 << N_BITS {
            for j in 0..1 << N_BITS {
                let curr_omega = || -> CM31 {
                    if j >= i {
                        return powers[j - i];
                    }
                    return inv_powers[i - j];
                };
                let (real, im) = curr_omega().split();
                mds[i][j] = n_inv * (lambda - (im / (one + real)))
            }
        }

        // print mds
        let mds_set: BTreeSet<M31> = mds.iter().flatten().copied().collect();

        println!("len {}", mds_set.len());
        println!("mds_set {:?}", mds_set);

        // let matrix =
            // SMatrix::<M31, { 1 << N_BITS }, { 1 << N_BITS }>::from_fn(|i, j| mds[i][j] / mds[0][0]);
        let matrix = SMatrix::<M31, 24, 24>::from_fn(|i, j| mds[i][j] / mds[0][0]);
        println!("matrix: {}", matrix);
        // println!(
        //     "det: {}",
        //     calculate_determinant(
        //         mds.iter()
        //             .map(|row| row.iter().copied().collect())
        //             .collect()
        //     )
        // );
    }

    // fn _calculate_determinant(matrix: Vec<Vec<M31>>) -> M31 {
    //     let size = matrix.len();

    //     if size == 1 {
    //         // Base case: determinant of a 1x1 matrix is the element itself
    //         matrix[0][0]
    //     } else {
    //         let mut determinant = M31::zero();
    //         for i in 0..size {
    //             // Calculate the cofactor
    //             let cofactor =
    //                 matrix[0][i] * cofactor_sign(0, i) * calculate_minor(matrix.clone(), 0, i);

    //             // Accumulate the cofactor to the determinant
    //             determinant += cofactor;
    //         }
    //         determinant
    //     }
    // }

    // fn _calculate_minor(matrix: Vec<Vec<M31>>, row: usize, col: usize) -> M31 {
    //     // Calculate the minor by excluding the specified row and column
    //     let size = matrix.len();
    //     let mut minor_matrix = vec![vec![M31::zero(); size - 1]; size - 1];

    //     let mut new_row = 0;
    //     for i in 0..size {
    //         if i == row {
    //             continue;
    //         }

    //         let mut new_col = 0;
    //         for j in 0..size {
    //             if j == col {
    //                 continue;
    //             }

    //             minor_matrix[new_row][new_col] = matrix[i][j];
    //             new_col += 1;
    //         }
    //         new_row += 1;
    //     }

    //     // Calculate the determinant of the minor
    //     // remove last row and column from minor matrix
    //     _calculate_determinant(minor_matrix)
    // }

    // fn _cofactor_sign(row: usize, col: usize) -> M31 {
    //     // Calculate the sign of the cofactor based on the row and column indices
    //     if (row + col) % 2 == 0 {
    //         M31::from(1)
    //     } else {
    //         M31::from(P - 1)
    //     }
    // }
}
