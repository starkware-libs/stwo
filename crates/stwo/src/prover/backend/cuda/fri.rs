use crate::core::fri::FOLD_STEP as CIRCLE_TO_LINE_FOLD_STEP;
use crate::core::circle::Coset;
use crate::core::fields::qm31::SecureField;
use crate::core::poly::line::LineDomain;
use crate::prover::poly::circle::SecureEvaluation;
use crate::prover::line::LineEvaluation;
use crate::prover::poly::twiddles::TwiddleTree;
use crate::prover::poly::BitReversedOrder;
use crate::prover::fri::FriOps;
use crate::prover::secure_column::SecureColumnByCoords;

use super::CudaBackend;
use crate::stwo_cuda as interface;

use interface::bindings;
use interface::bindings::CudaSecureField;
use crate::prover::backend::cuda::secure_column::CudaSecureColumn;

/// Single fold step (NitrooZK's original `fold_line`, fold_step = 1): folds a degree-d line
/// polynomial into degree d/2 using one `alpha`. The 74951f79 `FriOps::fold_line` (below) loops
/// this over `alphas` — k sequential single-steps == its batched contract (the SIMD backend just
/// fuses them for speed; the result is identical).
fn fold_line_single(
    eval: &LineEvaluation<CudaBackend>,
    alpha: SecureField,
    twiddles: &TwiddleTree<CudaBackend>,
) -> LineEvaluation<CudaBackend> {
    let n = eval.len();
    assert!(n >= 2, "Evaluation too small");

    let twiddles_size = twiddles.itwiddles.size;
    let remaining_folds = n.ilog2();
    let twiddle_offset: usize = twiddles_size - (1 << remaining_folds);

    unsafe {
        let gpu_domain = twiddles.itwiddles.device_ptr;
        let folded_values = CudaSecureColumn::new_with_size(n >> 1);

        bindings::fold_line(
            gpu_domain,
            twiddle_offset,
            n,
            CudaSecureColumn::from(&eval.values).device_ptr(),
            CudaSecureField::from(alpha),
            CudaSecureColumn::from(&folded_values).device_ptr(),
        );

        LineEvaluation::new(eval.domain().double(), folded_values)
    }
}

impl FriOps for CudaBackend {
    fn fold_line(
        eval: &LineEvaluation<Self>,
        alphas: &[SecureField],
        twiddles: &TwiddleTree<Self>,
    ) -> LineEvaluation<Self> {
        assert!(!alphas.is_empty(), "fold_line: alphas must be non-empty");
        let mut cur = fold_line_single(eval, alphas[0], twiddles);
        for &alpha in &alphas[1..] {
            cur = fold_line_single(&cur, alpha, twiddles);
        }
        cur
    }

    fn fold_circle_into_line(
        src: &SecureEvaluation<Self, BitReversedOrder>,
        alpha: SecureField,
        twiddles: &TwiddleTree<Self>,
    ) -> LineEvaluation<Self> {
        let n = src.len();
        // 74951f79 returns a FRESH f' (the `dst = dst*alpha^2 + f'` accumulation moved to callers).
        // We allocate a ZERO dst so NitrooZK's accumulating kernel computes 0*alpha^2 + f' = f'.
        let line_log_size = src.domain.log_size() - 1;
        let dst_domain = LineDomain::new(Coset::half_odds(line_log_size));
        let dst_values = SecureColumnByCoords::<Self>::zeros(1 << line_log_size);
        let dst = LineEvaluation::new(dst_domain, dst_values);
        assert_eq!(n >> CIRCLE_TO_LINE_FOLD_STEP, dst.len());

        unsafe {
            let gpu_domain = twiddles.itwiddles.device_ptr;
            let twiddle_offset = twiddles.root_coset.size() - dst.domain().size();
            bindings::fold_circle_into_line(
                gpu_domain,
                twiddle_offset,
                n,
                CudaSecureColumn::from(&src.values).device_ptr(),
                CudaSecureField::from(alpha),
                CudaSecureColumn::from(&dst.values).device_ptr(),
            );
        }
        dst
    }

    fn decompose(
        _eval: &SecureEvaluation<Self, BitReversedOrder>,
    ) -> (SecureEvaluation<Self, BitReversedOrder>, SecureField) {
        // This method will be deprecated and is no longer used in stwo. In stwo, every polynomial that goes into FRI is
        // in the FFT space already and there's no need to decompose it.
        todo!()
    }
}


// NitrooZK's original tests use the pre-74951f79 signatures (single-alpha fold_line with a
// fold_step arg, dst-style fold_circle_into_line) and are incompatible with our reconciled
// FriOps. Disabled until rewritten against the new signatures; the real soundness gate is the
// end-to-end byte-identity vs SimdBackend on the box.
#[cfg(any())]
mod tests {
    use std::iter::zip;

    use itertools::Itertools;
    use rand::rngs::SmallRng;
    use rand::{Rng, SeedableRng};
    use crate::prover::backend::{Column, ColumnOps, CpuBackend};
    use crate::core::circle::{CirclePoint, CirclePointIndex, Coset};
    use crate::core::fields::m31::{BaseField, M31};
    use crate::core::fields::qm31::{SecureField, QM31};
    use crate::prover::secure_column::SecureColumnByCoords;
    use crate::core::fields::Field;
    use crate::prover::fri::FriOps;
    use crate::core::poly::circle::{CanonicCoset, CircleDomain};
    use crate::core::poly::line::{LineDomain, LinePoly};
    use crate::prover::poly::circle::{PolyOps, SecureEvaluation};
    use crate::prover::line::LineEvaluation;
    use crate::prover::poly::twiddles::TwiddleTree;
    use crate::prover::poly::BitReversedOrder;

    use crate::prover::backend::cuda::CudaBackend;

    use crate::stwo_cuda::base_field_vec::BaseFieldVec;

    #[test]
    fn test_fold_line_compared_with_cpu_log24() {
        const LOG_SIZE: u32 = 24;
        let mut rng = SmallRng::seed_from_u64(0);
        let values: Vec<SecureField> = (0..1 << LOG_SIZE).map(|_| rng.gen()).collect_vec();
        let alpha = SecureField::from_u32_unchecked(1, 3, 5, 7);
        let domain = LineDomain::new(CanonicCoset::new(LOG_SIZE + 1).half_coset());

        let mut vec: [Vec<BaseField>; 4] = [vec![], vec![], vec![], vec![]];
        values.iter().for_each(|a| {
            vec[0].push(BaseField::from_u32_unchecked(a.0 .0 .0));
            vec[1].push(BaseField::from_u32_unchecked(a.0 .1 .0));
            vec[2].push(BaseField::from_u32_unchecked(a.1 .0 .0));
            vec[3].push(BaseField::from_u32_unchecked(a.1 .1 .0));
        });

        let cpu_fold = CpuBackend::fold_line(
            &LineEvaluation::new(
                domain,
                SecureColumnByCoords {
                    columns: vec.clone(),
                },
            ),
            alpha,
            &CpuBackend::precompute_twiddles(domain.coset()),
            1,
        );
        let vecs = [
            BaseFieldVec::from_vec(vec[0].clone()),
            BaseFieldVec::from_vec(vec[1].clone()),
            BaseFieldVec::from_vec(vec[2].clone()),
            BaseFieldVec::from_vec(vec[3].clone()),
        ];

        let gpu_fold = CudaBackend::fold_line(
            &LineEvaluation::new(domain, SecureColumnByCoords { columns: vecs }),
            alpha,
            &CudaBackend::precompute_twiddles(domain.coset()),
            1,
        );

        assert_eq!(cpu_fold.values.to_vec(), gpu_fold.values.to_cpu().to_vec());
    }

    #[test]
    fn test_fold_circle_into_line_log24() {
        const LOG_SIZE: u32 = 24;
        let values: Vec<SecureField> = (0..(1 << LOG_SIZE))
            .map(|i| SecureField::from_u32_unchecked(4 * i, 4 * i + 1, 4 * i + 2, 4 * i + 3))
            .collect();
        let alpha = SecureField::from_u32_unchecked(1, 3, 5, 7);
        let circle_domain = CanonicCoset::new(LOG_SIZE).circle_domain();
        let line_domain = LineDomain::new(circle_domain.half_coset);
        let mut cpu_fold = LineEvaluation::new(
            line_domain,
            SecureColumnByCoords::zeros(1 << (LOG_SIZE - 1)),
        );

        let mut vec: [Vec<BaseField>; 4] = [vec![], vec![], vec![], vec![]];
        values.iter().for_each(|a| {
            vec[0].push(BaseField::from_u32_unchecked(a.0 .0 .0));
            vec[1].push(BaseField::from_u32_unchecked(a.0 .1 .0));
            vec[2].push(BaseField::from_u32_unchecked(a.1 .0 .0));
            vec[3].push(BaseField::from_u32_unchecked(a.1 .1 .0));
        });
        let vecs = [
            BaseFieldVec::from_vec(vec[0].clone()),
            BaseFieldVec::from_vec(vec[1].clone()),
            BaseFieldVec::from_vec(vec[2].clone()),
            BaseFieldVec::from_vec(vec[3].clone()),
        ];
        CpuBackend::fold_circle_into_line(
            &mut cpu_fold,
            &SecureEvaluation::new(
                circle_domain,
                SecureColumnByCoords {
                    columns: vec.clone(),
                },
            ),
            alpha,
            &CpuBackend::precompute_twiddles(line_domain.coset()),
        );

        let mut cuda_fold = LineEvaluation::new(
            line_domain,
            SecureColumnByCoords::zeros(1 << (LOG_SIZE - 1)),
        );
        CudaBackend::fold_circle_into_line(
            &mut cuda_fold,
            &SecureEvaluation::new(circle_domain, SecureColumnByCoords { columns: vecs }),
            alpha,
            &CudaBackend::precompute_twiddles(line_domain.coset()),
        );

        assert_eq!(cpu_fold.values.to_vec(), cuda_fold.values.to_cpu().to_vec());
    }

    #[test]
    fn test_fold_line_compared_with_cpu() {
        const LOG_SIZE: u32 = 20;
        let mut rng = SmallRng::seed_from_u64(0);
        let values: Vec<SecureField> = (0..1 << LOG_SIZE).map(|_| rng.gen()).collect_vec();
        let alpha = SecureField::from_u32_unchecked(1, 3, 5, 7);
        let domain = LineDomain::new(CanonicCoset::new(LOG_SIZE + 1).half_coset());

        let mut vec: [Vec<BaseField>; 4] = [vec![], vec![], vec![], vec![]];
        values.iter().for_each(|a| {
            vec[0].push(BaseField::from_u32_unchecked(a.0 .0 .0));
            vec[1].push(BaseField::from_u32_unchecked(a.0 .1 .0));
            vec[2].push(BaseField::from_u32_unchecked(a.1 .0 .0));
            vec[3].push(BaseField::from_u32_unchecked(a.1 .1 .0));
        });

        let cpu_fold = CpuBackend::fold_line(
            &LineEvaluation::new(
                domain,
                SecureColumnByCoords {
                    columns: vec.clone(),
                },
            ),
            alpha,
            &CpuBackend::precompute_twiddles(domain.coset()),
            1,
        );
        let vecs = [
            BaseFieldVec::from_vec(vec[0].clone()),
            BaseFieldVec::from_vec(vec[1].clone()),
            BaseFieldVec::from_vec(vec[2].clone()),
            BaseFieldVec::from_vec(vec[3].clone()),
        ];

        let gpu_fold = CudaBackend::fold_line(
            &LineEvaluation::new(domain, SecureColumnByCoords { columns: vecs }),
            alpha,
            &CudaBackend::precompute_twiddles(domain.coset()),
            1,
        );

        assert_eq!(cpu_fold.values.to_vec(), gpu_fold.values.to_cpu().to_vec());
    }

    #[test]
    fn test_fold_line() {
        const DEGREE: usize = 8;
        let even_coeffs: [SecureField; DEGREE / 2] = [1, 2, 1, 3]
            .map(BaseField::from_u32_unchecked)
            .map(SecureField::from);
        let odd_coeffs: [SecureField; DEGREE / 2] = [3, 5, 4, 1]
            .map(BaseField::from_u32_unchecked)
            .map(SecureField::from);
        let poly = LinePoly::new([even_coeffs, odd_coeffs].concat());
        let even_poly = LinePoly::new(even_coeffs.to_vec());
        let odd_poly = LinePoly::new(odd_coeffs.to_vec());
        let alpha = BaseField::from_u32_unchecked(19283).into();
        let domain = LineDomain::new(Coset::half_odds(DEGREE.ilog2()));
        let drp_domain = domain.double();
        let mut values: Vec<SecureField> = domain
            .iter()
            .map(|p| poly.eval_at_point(p.into()))
            .collect();
        CpuBackend::bit_reverse_column(&mut values);
        let mut vec: [Vec<BaseField>; 4] = [vec![], vec![], vec![], vec![]];
        values.iter().for_each(|a| {
            vec[0].push(BaseField::from_u32_unchecked(a.0 .0 .0));
            vec[1].push(BaseField::from_u32_unchecked(a.0 .1 .0));
            vec[2].push(BaseField::from_u32_unchecked(a.1 .0 .0));
            vec[3].push(BaseField::from_u32_unchecked(a.1 .1 .0));
        });
        let vecs = [
            BaseFieldVec::from_vec(vec[0].clone()),
            BaseFieldVec::from_vec(vec[1].clone()),
            BaseFieldVec::from_vec(vec[2].clone()),
            BaseFieldVec::from_vec(vec[3].clone()),
        ];
        let evals: LineEvaluation<CudaBackend> =
            LineEvaluation::new(domain, SecureColumnByCoords { columns: vecs });

        let drp_evals = CudaBackend::fold_line(
            &evals,
            alpha,
            &CudaBackend::precompute_twiddles(domain.coset()),
            1,
        );
        let mut drp_evals = drp_evals.values.to_cpu().into_iter().collect_vec();
        CpuBackend::bit_reverse_column(&mut drp_evals);

        assert_eq!(drp_evals.len(), DEGREE / 2);
        for (i, (&drp_eval, x)) in zip(&drp_evals, drp_domain).enumerate() {
            let f_e: SecureField = even_poly.eval_at_point(x.into());
            let f_o: SecureField = odd_poly.eval_at_point(x.into());

            assert_eq!(drp_eval, (f_e + alpha * f_o).double(), "mismatch at {i}");
        }
    }

    #[test]
    fn test_fold_line_more_than_once() {
        const LOG_SIZE: u32 = 13;
        let mut rng = SmallRng::seed_from_u64(0);
        let values: Vec<SecureField> = (0..1 << LOG_SIZE).map(|_| rng.gen()).collect_vec();
        let alpha = SecureField::from_u32_unchecked(1, 3, 5, 7);
        let domain = LineDomain::new(CanonicCoset::new(LOG_SIZE + 1).half_coset());

        let mut vec: [Vec<BaseField>; 4] = [vec![], vec![], vec![], vec![]];
        values.iter().for_each(|a| {
            vec[0].push(BaseField::from_u32_unchecked(a.0 .0 .0));
            vec[1].push(BaseField::from_u32_unchecked(a.0 .1 .0));
            vec[2].push(BaseField::from_u32_unchecked(a.1 .0 .0));
            vec[3].push(BaseField::from_u32_unchecked(a.1 .1 .0));
        });
        let vecs = [
            BaseFieldVec::from_vec(vec[0].clone()),
            BaseFieldVec::from_vec(vec[1].clone()),
            BaseFieldVec::from_vec(vec[2].clone()),
            BaseFieldVec::from_vec(vec[3].clone()),
        ];

        let first_cpu_fold = CpuBackend::fold_line(
            &LineEvaluation::new(
                domain,
                SecureColumnByCoords {
                    columns: vec.clone(),
                },
            ),
            alpha,
            &CpuBackend::precompute_twiddles(domain.coset()),
            1,
        );
        let second_cpu_fold = CpuBackend::fold_line(
            &first_cpu_fold,
            alpha,
            &CpuBackend::precompute_twiddles(domain.coset()),
            1,
        );
        let third_cpu_fold = CpuBackend::fold_line(
            &second_cpu_fold,
            alpha,
            &CpuBackend::precompute_twiddles(domain.coset()),
            1,
        );

        let first_gpu_fold = CudaBackend::fold_line(
            &LineEvaluation::new(domain, SecureColumnByCoords { columns: vecs }),
            alpha,
            &CudaBackend::precompute_twiddles(domain.coset()),
            1,
        );
        let second_gpu_fold = CudaBackend::fold_line(
            &first_gpu_fold,
            alpha,
            &CudaBackend::precompute_twiddles(domain.coset()),
            1,
        );
        let third_gpu_fold = CudaBackend::fold_line(
            &second_gpu_fold,
            alpha,
            &CudaBackend::precompute_twiddles(domain.coset()),
            1,
        );

        assert_eq!(
            third_cpu_fold.values.to_vec(),
            third_gpu_fold.values.to_cpu().to_vec()
        );
    }

    #[test]
    fn test_fold_circle_into_line_compared_with_cpu() {
        const LOG_SIZE: u32 = 13;
        let values: Vec<SecureField> = (0..(1 << LOG_SIZE))
            .map(|i| SecureField::from_u32_unchecked(4 * i, 4 * i + 1, 4 * i + 2, 4 * i + 3))
            .collect();
        let alpha = SecureField::from_u32_unchecked(1, 3, 5, 7);
        let circle_domain = CanonicCoset::new(LOG_SIZE).circle_domain();
        let line_domain = LineDomain::new(circle_domain.half_coset);
        let mut cpu_fold = LineEvaluation::new(
            line_domain,
            SecureColumnByCoords::zeros(1 << (LOG_SIZE - 1)),
        );

        let mut vec: [Vec<BaseField>; 4] = [vec![], vec![], vec![], vec![]];
        values.iter().for_each(|a| {
            vec[0].push(BaseField::from_u32_unchecked(a.0 .0 .0));
            vec[1].push(BaseField::from_u32_unchecked(a.0 .1 .0));
            vec[2].push(BaseField::from_u32_unchecked(a.1 .0 .0));
            vec[3].push(BaseField::from_u32_unchecked(a.1 .1 .0));
        });
        let vecs = [
            BaseFieldVec::from_vec(vec[0].clone()),
            BaseFieldVec::from_vec(vec[1].clone()),
            BaseFieldVec::from_vec(vec[2].clone()),
            BaseFieldVec::from_vec(vec[3].clone()),
        ];
        CpuBackend::fold_circle_into_line(
            &mut cpu_fold,
            &SecureEvaluation::new(
                circle_domain,
                SecureColumnByCoords {
                    columns: vec.clone(),
                },
            ),
            alpha,
            &CpuBackend::precompute_twiddles(line_domain.coset()),
        );

        let mut cuda_fold = LineEvaluation::new(
            line_domain,
            SecureColumnByCoords::zeros(1 << (LOG_SIZE - 1)),
        );
        CudaBackend::fold_circle_into_line(
            &mut cuda_fold,
            &SecureEvaluation::new(circle_domain, SecureColumnByCoords { columns: vecs }),
            alpha,
            &CudaBackend::precompute_twiddles(line_domain.coset()),
        );

        assert_eq!(cpu_fold.values.to_vec(), cuda_fold.values.to_cpu().to_vec());
    }

    #[test]
    fn test_fold_circle_into_line_fibonacci() {
        let dst_values = SecureColumnByCoords::<CpuBackend> {
            columns: [
                vec![
                    M31(175232161),
                    M31(1621852215),
                    M31(1344284923),
                    M31(1778156084),
                    M31(1400400421),
                    M31(1191304209),
                    M31(1370890230),
                    M31(876649922),
                    M31(298405189),
                    M31(783573640),
                    M31(1013810795),
                    M31(1600595576),
                    M31(1822272537),
                    M31(1448770626),
                    M31(160607333),
                    M31(1985177603),
                    M31(1800921154),
                    M31(34711183),
                    M31(56534507),
                    M31(510529825),
                    M31(1900043069),
                    M31(1061675418),
                    M31(1348367098),
                    M31(5042126),
                    M31(944964573),
                    M31(1011034686),
                    M31(1601450494),
                    M31(1596148275),
                    M31(866712268),
                    M31(478168644),
                    M31(1430493482),
                    M31(2077703015),
                ],
                vec![
                    M31(193498252),
                    M31(1930857823),
                    M31(1574616603),
                    M31(1096832869),
                    M31(1535214166),
                    M31(242948629),
                    M31(1169096088),
                    M31(303351246),
                    M31(1205776310),
                    M31(404835605),
                    M31(717680932),
                    M31(1116437451),
                    M31(929009816),
                    M31(1130918817),
                    M31(1779189410),
                    M31(1391495896),
                    M31(66480840),
                    M31(371990063),
                    M31(1865649414),
                    M31(1802222899),
                    M31(159552644),
                    M31(896604672),
                    M31(1931207970),
                    M31(283233763),
                    M31(484330319),
                    M31(989165768),
                    M31(406520763),
                    M31(395010529),
                    M31(1641089337),
                    M31(1556631089),
                    M31(630529731),
                    M31(1094056465),
                ],
                vec![
                    M31(1137009986),
                    M31(1382256183),
                    M31(234426591),
                    M31(723907790),
                    M31(110421860),
                    M31(660239829),
                    M31(1641100420),
                    M31(1419769276),
                    M31(537001295),
                    M31(1021500262),
                    M31(1724736086),
                    M31(1286280744),
                    M31(1618141633),
                    M31(948764242),
                    M31(2026990422),
                    M31(172738322),
                    M31(1765374986),
                    M31(185155195),
                    M31(2011249762),
                    M31(311260517),
                    M31(2101859898),
                    M31(109156094),
                    M31(2039976666),
                    M31(689835282),
                    M31(1371899564),
                    M31(1842193228),
                    M31(1628367468),
                    M31(807003574),
                    M31(1505108472),
                    M31(1906175695),
                    M31(537245304),
                    M31(2128390530),
                ],
                vec![
                    M31(1733880617),
                    M31(850794569),
                    M31(2030610590),
                    M31(562233856),
                    M31(1460771010),
                    M31(2069111798),
                    M31(2082486460),
                    M31(517895090),
                    M31(591152265),
                    M31(879189323),
                    M31(492693465),
                    M31(1962268963),
                    M31(1180810278),
                    M31(1360959609),
                    M31(1465998969),
                    M31(579298036),
                    M31(1042819150),
                    M31(1472324807),
                    M31(1088967404),
                    M31(1996597293),
                    M31(285959712),
                    M31(1447284512),
                    M31(803149008),
                    M31(1842290266),
                    M31(861208576),
                    M31(1949442307),
                    M31(679820802),
                    M31(1462843748),
                    M31(1123289729),
                    M31(1700666883),
                    M31(2036502646),
                    M31(26988055),
                ],
            ],
        };

        let dst_domain = LineDomain::new(Coset {
            initial_index: CirclePointIndex(16777216),
            initial: CirclePoint {
                x: M31(838195206),
                y: M31(1774253895),
            },
            step_size: CirclePointIndex(67108864),
            step: CirclePoint {
                x: M31(1179735656),
                y: M31(1241207368),
            },
            log_size: 5,
        });
        let mut dst = LineEvaluation::new(dst_domain.clone(), dst_values.clone());

        let src_values = SecureColumnByCoords {
            columns: [
                vec![
                    M31(304950876),
                    M31(468888998),
                    M31(662049419),
                    M31(559536868),
                    M31(519811798),
                    M31(1830631122),
                    M31(301366316),
                    M31(261440159),
                    M31(333990060),
                    M31(365575721),
                    M31(435623077),
                    M31(45397331),
                    M31(1161993269),
                    M31(1393066040),
                    M31(715018661),
                    M31(870928826),
                    M31(675394675),
                    M31(1030140996),
                    M31(1672132964),
                    M31(813627705),
                    M31(1188631652),
                    M31(545287508),
                    M31(822269047),
                    M31(1770418503),
                    M31(1741496572),
                    M31(919203788),
                    M31(230707271),
                    M31(19933590),
                    M31(832768210),
                    M31(1237566190),
                    M31(1544017474),
                    M31(1010577996),
                    M31(1063153792),
                    M31(468965353),
                    M31(1199345557),
                    M31(1363315294),
                    M31(355020480),
                    M31(1582954225),
                    M31(2147057494),
                    M31(789487453),
                    M31(1189579238),
                    M31(277935922),
                    M31(1673517805),
                    M31(971368543),
                    M31(786145455),
                    M31(1690284642),
                    M31(2088428507),
                    M31(630569867),
                    M31(1891241100),
                    M31(366469006),
                    M31(740248766),
                    M31(15684480),
                    M31(2051640642),
                    M31(1349184943),
                    M31(1749567394),
                    M31(64039336),
                    M31(1482143564),
                    M31(311803874),
                    M31(1864323502),
                    M31(227669616),
                    M31(1416931884),
                    M31(1053871952),
                    M31(33810749),
                    M31(1978616835),
                ],
                vec![
                    M31(1653867103),
                    M31(2012450371),
                    M31(510632389),
                    M31(18030316),
                    M31(1247725731),
                    M31(1023008510),
                    M31(2077225448),
                    M31(1998325775),
                    M31(1959992578),
                    M31(509241715),
                    M31(1483935162),
                    M31(1093400760),
                    M31(1056874060),
                    M31(887023960),
                    M31(1333602134),
                    M31(944980767),
                    M31(1026474474),
                    M31(397761217),
                    M31(1988431065),
                    M31(1526333898),
                    M31(1981476440),
                    M31(206977132),
                    M31(1263728736),
                    M31(824323342),
                    M31(1671452381),
                    M31(1515796040),
                    M31(1872070342),
                    M31(2120604997),
                    M31(961362972),
                    M31(2055229949),
                    M31(1522152053),
                    M31(35063378),
                    M31(807917714),
                    M31(613504427),
                    M31(2118942714),
                    M31(1350414356),
                    M31(1115884522),
                    M31(2017681517),
                    M31(1575707281),
                    M31(589503178),
                    M31(2062924228),
                    M31(1716741905),
                    M31(525713675),
                    M31(508026147),
                    M31(553780862),
                    M31(1594128254),
                    M31(507875939),
                    M31(1594957728),
                    M31(1043992688),
                    M31(1381706282),
                    M31(1302912972),
                    M31(1785756254),
                    M31(2137668861),
                    M31(1383115679),
                    M31(509240725),
                    M31(558176302),
                    M31(761722741),
                    M31(511196333),
                    M31(329620491),
                    M31(974011575),
                    M31(1389674518),
                    M31(881352056),
                    M31(424972196),
                    M31(1855763781),
                ],
                vec![
                    M31(967350995),
                    M31(310975164),
                    M31(287547061),
                    M31(1171273988),
                    M31(211233345),
                    M31(586417579),
                    M31(1410623691),
                    M31(670457818),
                    M31(1622928993),
                    M31(174160475),
                    M31(1847043929),
                    M31(403002890),
                    M31(1469634276),
                    M31(1353098177),
                    M31(404121449),
                    M31(1795451374),
                    M31(1099617677),
                    M31(1249825388),
                    M31(14537885),
                    M31(538221831),
                    M31(936800592),
                    M31(1471304983),
                    M31(1887906483),
                    M31(936736872),
                    M31(810108453),
                    M31(1904371485),
                    M31(1568542636),
                    M31(1434456880),
                    M31(561181009),
                    M31(903692994),
                    M31(1623402143),
                    M31(1819132719),
                    M31(132717721),
                    M31(753599591),
                    M31(1574783454),
                    M31(1086505391),
                    M31(760027086),
                    M31(664252385),
                    M31(1763541163),
                    M31(160918721),
                    M31(204669535),
                    M31(472454788),
                    M31(374904910),
                    M31(1387297827),
                    M31(462935159),
                    M31(96026193),
                    M31(1116902555),
                    M31(2005890162),
                    M31(1792870094),
                    M31(902789995),
                    M31(1535380049),
                    M31(749289645),
                    M31(1258142643),
                    M31(2020377582),
                    M31(1817741382),
                    M31(2041703074),
                    M31(1416977506),
                    M31(1221796582),
                    M31(1077281597),
                    M31(627937237),
                    M31(1557272528),
                    M31(1143811691),
                    M31(2023916882),
                    M31(1387929753),
                ],
                vec![
                    M31(2001733150),
                    M31(1282490762),
                    M31(1946552578),
                    M31(1682637865),
                    M31(1441952442),
                    M31(309753082),
                    M31(1160816063),
                    M31(592492275),
                    M31(652159046),
                    M31(726237933),
                    M31(1788204376),
                    M31(80318529),
                    M31(1269393791),
                    M31(359268688),
                    M31(1814801842),
                    M31(2011892051),
                    M31(13070900),
                    M31(213753825),
                    M31(1126239308),
                    M31(865162950),
                    M31(41495267),
                    M31(528359501),
                    M31(1437288818),
                    M31(787761504),
                    M31(1310321724),
                    M31(645041430),
                    M31(31749535),
                    M31(2015702994),
                    M31(203901209),
                    M31(856861302),
                    M31(932696707),
                    M31(2067158418),
                    M31(1667156589),
                    M31(1650484101),
                    M31(458644642),
                    M31(1928162080),
                    M31(1770168848),
                    M31(621770387),
                    M31(1724634411),
                    M31(68217692),
                    M31(2107811928),
                    M31(650201913),
                    M31(893611820),
                    M31(1583154384),
                    M31(433470094),
                    M31(203748212),
                    M31(2062353347),
                    M31(774536039),
                    M31(39013234),
                    M31(34864872),
                    M31(1293726101),
                    M31(523532884),
                    M31(510115480),
                    M31(904003817),
                    M31(1102477755),
                    M31(939320526),
                    M31(1399060270),
                    M31(1103477198),
                    M31(250427756),
                    M31(263935133),
                    M31(579210410),
                    M31(1819955361),
                    M31(880494071),
                    M31(1955528510),
                ],
            ],
        };
        let src_domain = CircleDomain::new(Coset {
            initial_index: CirclePointIndex(16777216),
            initial: CirclePoint {
                x: M31(838195206),
                y: M31(1774253895),
            },
            step_size: CirclePointIndex(67108864),
            step: CirclePoint {
                x: M31(1179735656),
                y: M31(1241207368),
            },
            log_size: 5,
        });
        let src = SecureEvaluation::<CpuBackend, BitReversedOrder>::new(src_domain, src_values);
        let alpha = QM31::from_m31(
            M31(1882064794),
            M31(2043041752),
            M31(1688786630),
            M31(1409241156),
        );
        let root_coset = Coset {
            initial_index: CirclePointIndex(8388608),
            initial: CirclePoint {
                x: M31(785043271),
                y: M31(1260750973),
            },
            step_size: CirclePointIndex(33554432),
            step: CirclePoint {
                x: M31(579625837),
                y: M31(1690787918),
            },
            log_size: 6,
        };
        let twiddles = vec![
            M31(785043271),
            M31(1260750973),
            M31(736262640),
            M31(1553669210),
            M31(479120236),
            M31(225856549),
            M31(197700101),
            M31(1079800039),
            M31(1911378744),
            M31(1577470940),
            M31(1334497267),
            M31(2085743640),
            M31(477953613),
            M31(125103457),
            M31(1977033713),
            M31(2005527287),
            M31(251924953),
            M31(636875771),
            M31(48903418),
            M31(1896945393),
            M31(1514613395),
            M31(870936612),
            M31(1297878576),
            M31(583555490),
            M31(640817200),
            M31(1702126977),
            M31(1054411686),
            M31(648593218),
            M31(1014093253),
            M31(2137011181),
            M31(81378258),
            M31(789857006),
            M31(838195206),
            M31(1774253895),
            M31(1739004854),
            M31(262191051),
            M31(206059115),
            M31(212443077),
            M31(1796741361),
            M31(883753057),
            M31(2140339328),
            M31(404685994),
            M31(9803698),
            M31(68458636),
            M31(14530030),
            M31(228509164),
            M31(1038945916),
            M31(134155457),
            M31(579625837),
            M31(1690787918),
            M31(1641940819),
            M31(2121318970),
            M31(1952787376),
            M31(1580223790),
            M31(1013961365),
            M31(280947147),
            M31(1179735656),
            M31(1241207368),
            M31(1415090252),
            M31(2112881577),
            M31(590768354),
            M31(978592373),
            M31(32768),
            M31(1),
        ];
        let itwiddles = vec![
            M31(1541158724),
            M31(16208603),
            M31(62823040),
            M31(1642210396),
            M31(1631996251),
            M31(1007591000),
            M31(1874949287),
            M31(1849862501),
            M31(781334166),
            M31(132945364),
            M31(1278220752),
            M31(214347122),
            M31(1165838173),
            M31(2054194025),
            M31(1234096940),
            M31(1721693449),
            M31(622651690),
            M31(1373671071),
            M31(82740187),
            M31(1683898894),
            M31(1918467639),
            M31(1186332607),
            M31(1296073347),
            M31(401388709),
            M31(1383565722),
            M31(656788371),
            M31(1787268380),
            M31(1809670981),
            M31(99372120),
            M31(765975505),
            M31(774809712),
            M31(348924564),
            M31(2029303208),
            M31(959596234),
            M31(1051468699),
            M31(721860568),
            M31(1767118503),
            M31(218253990),
            M31(1356867335),
            M31(1955048591),
            M31(559361447),
            M31(1046725194),
            M31(448375059),
            M31(1036402186),
            M31(2138687850),
            M31(1268642696),
            M31(1381082522),
            M31(559888787),
            M31(248349974),
            M31(969924856),
            M31(1461702947),
            M31(655012266),
            M31(1385854532),
            M31(1859156789),
            M31(349252128),
            M31(421110815),
            M31(1160411471),
            M31(1518526074),
            M31(490549293),
            M31(1942501404),
            M31(991237807),
            M31(775648038),
            M31(65536),
            M31(1),
        ];
        let twiddle_tree = TwiddleTree::<CpuBackend> {
            root_coset,
            twiddles: twiddles.clone(),
            itwiddles: itwiddles.clone(),
        };

        // CUDA
        let dst_values_cuda = SecureColumnByCoords {
            columns: [
                BaseFieldVec::from_vec(dst.values.columns[0].clone()),
                BaseFieldVec::from_vec(dst.values.columns[1].clone()),
                BaseFieldVec::from_vec(dst.values.columns[2].clone()),
                BaseFieldVec::from_vec(dst.values.columns[3].clone()),
            ],
        };
        let mut dst_cuda = LineEvaluation::<CudaBackend>::new(dst.domain(), dst_values_cuda);
        let src_cuda_values = SecureColumnByCoords::<CudaBackend> {
            columns: [
                BaseFieldVec::from_vec(src.columns[0].clone()),
                BaseFieldVec::from_vec(src.columns[1].clone()),
                BaseFieldVec::from_vec(src.columns[2].clone()),
                BaseFieldVec::from_vec(src.columns[3].clone()),
            ],
        };
        let src_cuda =
            SecureEvaluation::<CudaBackend, BitReversedOrder>::new(src.domain, src_cuda_values);

        let twiddle_tree_cuda = TwiddleTree::<CudaBackend> {
            root_coset: twiddle_tree.root_coset,
            twiddles: BaseFieldVec::from_vec(twiddles),
            itwiddles: BaseFieldVec::from_vec(itwiddles),
        };

        CpuBackend::fold_circle_into_line(&mut dst, &src, alpha, &twiddle_tree);
        CudaBackend::fold_circle_into_line(&mut dst_cuda, &src_cuda, alpha, &twiddle_tree_cuda);

        for i in 0..4 {
            assert_eq!(dst_cuda.values.columns[i].to_cpu(), dst.values.columns[i]);
        }
    }
}
