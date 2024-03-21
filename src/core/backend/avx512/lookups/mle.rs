use std::iter::zip;

use bytemuck::Zeroable;
use num_traits::{One, Zero};

use crate::core::air::evaluation::SecureColumn;
use crate::core::backend::avx512::cm31::PackedCM31;
use crate::core::backend::avx512::qm31::PackedQM31;
use crate::core::backend::avx512::{AVX512Backend, BaseFieldVec, PackedBaseField, K_BLOCK_SIZE};
use crate::core::backend::cpu::lookups::mle::eval_mle_at_point;
use crate::core::backend::Column;
use crate::core::fields::m31::BaseField;
use crate::core::fields::qm31::SecureField;
use crate::core::lookups::mle::{Mle, MleOps};
use crate::core::lookups::sumcheck::SumcheckOracle;
use crate::core::lookups::utils::Polynomial;

impl MleOps<BaseField> for AVX512Backend {
    fn eval_at_point(mle: &Mle<Self, BaseField>, point: &[BaseField]) -> BaseField {
        // TODO: Implement AVX version.
        assert_eq!(point.len(), mle.num_variables());
        eval_mle_at_point(mle.as_slice(), point)
    }

    fn fix_first(mle: Mle<Self, BaseField>, assignment: SecureField) -> Mle<Self, SecureField> {
        // Column of SecureField elements (Structure of arrays)
        // TODO: Make this better.
        let length = mle.len();

        if length == 1 {
            let mut c0 = [BaseField::zero(); K_BLOCK_SIZE];
            c0[0] = mle.into_evals().as_slice()[0];

            let c0 = PackedBaseField::from_array(c0);
            let c1 = PackedBaseField::zeroed();
            let c2 = PackedBaseField::zeroed();
            let c3 = PackedBaseField::zeroed();

            return Mle::new(SecureColumn {
                cols: [
                    BaseFieldVec {
                        data: vec![c0],
                        length,
                    },
                    BaseFieldVec {
                        data: vec![c1],
                        length,
                    },
                    BaseFieldVec {
                        data: vec![c2],
                        length,
                    },
                    BaseFieldVec {
                        data: vec![c3],
                        length,
                    },
                ],
            });
        }

        let mut col0 = Vec::new();
        let mut col1 = Vec::new();
        let mut col2 = Vec::new();
        let mut col3 = Vec::new();

        let [a0, a1, a2, a3] = assignment.to_m31_array();
        let assignment0 = PackedBaseField::from_array([a0; K_BLOCK_SIZE]);
        let assignment1 = PackedBaseField::from_array([a1; K_BLOCK_SIZE]);
        let assignment2 = PackedBaseField::from_array([a2; K_BLOCK_SIZE]);
        let assignment3 = PackedBaseField::from_array([a3; K_BLOCK_SIZE]);

        let assignment = PackedQM31([
            PackedCM31([assignment0, assignment1]),
            PackedCM31([assignment2, assignment3]),
        ]);

        let evals = if length >= 2 * K_BLOCK_SIZE {
            mle.into_evals().data
        } else {
            let evals = mle.into_evals();
            let evals = evals.as_slice();

            let mut lhs = [BaseField::zero(); K_BLOCK_SIZE];
            let mut rhs = [BaseField::zero(); K_BLOCK_SIZE];

            let midpoint = evals.len() / 2;
            lhs[0..midpoint].copy_from_slice(&evals[0..midpoint]);
            rhs[0..midpoint].copy_from_slice(&evals[midpoint..]);

            vec![
                PackedBaseField::from_array(lhs),
                PackedBaseField::from_array(rhs),
            ]
        };

        let (lhs_evals, rhs_evals) = evals.split_at(evals.len() / 2);

        for (&lhs_eval, &rhs_eval) in zip(lhs_evals, rhs_evals) {
            // `= eq(0, assignment) * lhs + eq(1, assignment) * rhs`
            let PackedQM31([PackedCM31([c0, c1]), PackedCM31([c2, c3])]) =
                assignment * (rhs_eval - lhs_eval) + lhs_eval;

            col0.push(c0);
            col1.push(c1);
            col2.push(c2);
            col3.push(c3);
        }

        Mle::new(SecureColumn {
            cols: [
                BaseFieldVec { data: col0, length },
                BaseFieldVec { data: col1, length },
                BaseFieldVec { data: col2, length },
                BaseFieldVec { data: col3, length },
            ],
        })
    }
}

impl MleOps<SecureField> for AVX512Backend {
    fn eval_at_point(mle: &Mle<Self, SecureField>, point: &[SecureField]) -> SecureField {
        /// Evaluates the multi-linear extension `mle` at point `p`.
        ///
        /// `secure_mle` is a [`SecureField`] multi-linear extension stored as structure-of-arrays.
        fn eval_mle_at_point(secure_mle: [&[BaseField]; 4], p: &[SecureField]) -> SecureField {
            match p {
                [] => SecureField::from_m31_array(secure_mle.map(|mle| mle[0])),
                [p_i, p @ ..] => {
                    let mle_midpoint = secure_mle[0].len() / 2;
                    let (lhs0, rhs0) = secure_mle[0].split_at(mle_midpoint);
                    let (lhs1, rhs1) = secure_mle[1].split_at(mle_midpoint);
                    let (lhs2, rhs2) = secure_mle[2].split_at(mle_midpoint);
                    let (lhs3, rhs3) = secure_mle[3].split_at(mle_midpoint);
                    let lhs_eval = eval_mle_at_point([lhs0, lhs1, lhs2, lhs3], p);
                    let rhs_eval = eval_mle_at_point([rhs0, rhs1, rhs2, rhs3], p);
                    // `= eq(0, p_i) * lhs + eq(1, p_i) * rhs`
                    *p_i * (rhs_eval - lhs_eval) + lhs_eval
                }
            }
        }

        // TODO: Add AVX implementation.
        assert_eq!(point.len(), mle.num_variables());
        let secure_mle = [
            mle.cols[0].as_slice(),
            mle.cols[1].as_slice(),
            mle.cols[2].as_slice(),
            mle.cols[3].as_slice(),
        ];
        eval_mle_at_point(secure_mle, point)
    }

    fn fix_first(mle: Mle<Self, SecureField>, assignment: SecureField) -> Mle<Self, SecureField> {
        // Column of SecureField elements (Structure of arrays)
        // TODO: Make this better.
        let length = mle.len();

        if length == 1 {
            return mle;
        }

        let [a0, a1, a2, a3] = assignment.to_m31_array();
        let assignment0 = PackedBaseField::from_array([a0; K_BLOCK_SIZE]);
        let assignment1 = PackedBaseField::from_array([a1; K_BLOCK_SIZE]);
        let assignment2 = PackedBaseField::from_array([a2; K_BLOCK_SIZE]);
        let assignment3 = PackedBaseField::from_array([a3; K_BLOCK_SIZE]);

        let assignment = PackedQM31([
            PackedCM31([assignment0, assignment1]),
            PackedCM31([assignment2, assignment3]),
        ]);

        let (mut col0, mut col1, mut col2, mut col3) = if length >= 2 * K_BLOCK_SIZE {
            let evals = mle.into_evals();
            let [col0, col1, col2, col3] = evals.cols;
            (col0.data, col1.data, col2.data, col3.data)
        } else {
            let evals = mle.into_evals();

            // TODO: Rename and better docs.
            let split = |col: &[BaseField]| {
                let mut lhs = [BaseField::zero(); K_BLOCK_SIZE];
                let mut rhs = [BaseField::zero(); K_BLOCK_SIZE];

                let midpoint = col.len() / 2;
                lhs[0..midpoint].copy_from_slice(&col[0..midpoint]);
                rhs[0..midpoint].copy_from_slice(&col[midpoint..]);

                vec![
                    PackedBaseField::from_array(lhs),
                    PackedBaseField::from_array(rhs),
                ]
            };

            (
                split(evals.cols[0].as_slice()),
                split(evals.cols[1].as_slice()),
                split(evals.cols[2].as_slice()),
                split(evals.cols[3].as_slice()),
            )
        };

        let packed_midpoint = col0.len() / 2;

        for i in 0..packed_midpoint {
            let lhs_eval = PackedQM31([
                PackedCM31([col0[i], col1[i]]),
                PackedCM31([col2[i], col3[i]]),
            ]);

            let rhs_eval = PackedQM31([
                PackedCM31([col0[i + packed_midpoint], col1[i + packed_midpoint]]),
                PackedCM31([col2[i + packed_midpoint], col3[i + packed_midpoint]]),
            ]);

            // `= eq(0, assignment) * lhs + eq(1, assignment) * rhs`
            let PackedQM31([PackedCM31([c0, c1]), PackedCM31([c2, c3])]) =
                assignment * (rhs_eval - lhs_eval) + lhs_eval;

            col0[i] = c0;
            col1[i] = c1;
            col2[i] = c2;
            col3[i] = c3;
        }

        col0.truncate(packed_midpoint);
        col1.truncate(packed_midpoint);
        col2.truncate(packed_midpoint);
        col3.truncate(packed_midpoint);

        let length = length / 2;

        Mle::new(SecureColumn {
            cols: [
                BaseFieldVec { data: col0, length },
                BaseFieldVec { data: col1, length },
                BaseFieldVec { data: col2, length },
                BaseFieldVec { data: col3, length },
            ],
        })
    }
}

impl SumcheckOracle for Mle<AVX512Backend, SecureField> {
    fn num_variables(&self) -> usize {
        self.num_variables()
    }

    fn univariate_sum(&self, claim: SecureField) -> Polynomial<SecureField> {
        let x0 = SecureField::zero();
        let x1 = SecureField::one();

        let midpoint = self.len() / 2;

        let y0 = if midpoint < 8 * K_BLOCK_SIZE {
            let c0 = self.cols[0].as_slice()[0..midpoint].iter().sum();
            let c1 = self.cols[1].as_slice()[0..midpoint].iter().sum();
            let c2 = self.cols[2].as_slice()[0..midpoint].iter().sum();
            let c3 = self.cols[3].as_slice()[0..midpoint].iter().sum();

            SecureField::from_m31_array([c0, c1, c2, c3])
        } else {
            fn sum(values: &[PackedBaseField]) -> PackedBaseField {
                // let mut acc = PackedBaseField::zeroed();

                // for &v0 in values {
                //     acc += v0;
                // }

                // acc

                let mut acc0 = PackedBaseField::zeroed();
                let mut acc1 = PackedBaseField::zeroed();

                for &[v0, v1] in values.array_chunks() {
                    acc0 += v0;
                    acc1 += v1;
                }

                acc0 + acc1
            }

            // Sums a slice of [PackedBaseField] elements.
            let sum_packed = |packed: PackedBaseField| packed.to_array().into_iter().sum();

            let packed_midpoint = midpoint / K_BLOCK_SIZE;
            let c0 = sum_packed(sum(&self.cols[0].data[0..packed_midpoint]));
            let c1 = sum_packed(sum(&self.cols[1].data[0..packed_midpoint]));
            let c2 = sum_packed(sum(&self.cols[2].data[0..packed_midpoint]));
            let c3 = sum_packed(sum(&self.cols[3].data[0..packed_midpoint]));

            SecureField::from_m31_array([c0, c1, c2, c3])
        };

        let y1 = claim - y0;

        Polynomial::interpolate_lagrange(&[x0, x1], &[y0, y1])
    }

    fn fix_first(self, challenge: SecureField) -> Self {
        self.fix_first(challenge)
    }
}
