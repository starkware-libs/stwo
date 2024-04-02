use bytemuck::Zeroable;

use crate::core::air::evaluation::SecureColumn;
use crate::core::backend::avx512::cm31::PackedCM31;
use crate::core::backend::avx512::qm31::PackedQM31;
use crate::core::backend::avx512::{AVX512Backend, AvxMle, BaseFieldVec, K_BLOCK_SIZE};
use crate::core::backend::cpu::CpuMle;
use crate::core::backend::CPUBackend;
use crate::core::fields::qm31::SecureField;
use crate::core::fields::Field;
use crate::core::lookups::gkr::GkrLayer;
use crate::core::lookups::logup::{LogupOps, LogupOracle, LogupTrace};
use crate::core::lookups::mle::{ColumnV2, Mle, MleOps};
use crate::core::lookups::sumcheck::{SumcheckOracle, UnivariateEvals};

impl LogupOps for AVX512Backend {
    fn next_layer(layer: &LogupTrace<Self>) -> LogupTrace<Self> {
        // Fallback to CPU backend for small instances to avoid complexity with packed AVX types.
        if layer.len() < 2 * K_BLOCK_SIZE {
            return CPUBackend::next_layer(&layer.to_cpu()).to_avx();
        }

        let packed_midpoint = layer.len() / K_BLOCK_SIZE / 2;

        let mut next_numerators_col0 = Vec::new();
        let mut next_numerators_col1 = Vec::new();
        let mut next_numerators_col2 = Vec::new();
        let mut next_numerators_col3 = Vec::new();

        let mut next_denominators_col0 = Vec::new();
        let mut next_denominators_col1 = Vec::new();
        let mut next_denominators_col2 = Vec::new();
        let mut next_denominators_col3 = Vec::new();

        match layer {
            LogupTrace::Singles { denominators } => {
                for i in 0..packed_midpoint {
                    let (c0_evens, c0_odds) = denominators.cols[0].data[i * 2]
                        .deinterleave_with(denominators.cols[0].data[i * 2 + 1]);
                    let (c1_evens, c1_odds) = denominators.cols[1].data[i * 2]
                        .deinterleave_with(denominators.cols[1].data[i * 2 + 1]);
                    let (c2_evens, c2_odds) = denominators.cols[2].data[i * 2]
                        .deinterleave_with(denominators.cols[2].data[i * 2 + 1]);
                    let (c3_evens, c3_odds) = denominators.cols[3].data[i * 2]
                        .deinterleave_with(denominators.cols[3].data[i * 2 + 1]);

                    let evens = PackedQM31([
                        PackedCM31([c0_evens, c1_evens]),
                        PackedCM31([c2_evens, c3_evens]),
                    ]);

                    let odds = PackedQM31([
                        PackedCM31([c0_odds, c1_odds]),
                        PackedCM31([c2_odds, c3_odds]),
                    ]);

                    let PackedQM31(
                        [PackedCM31([numerators_c0, numerators_c1]), PackedCM31([numerators_c2, numerators_c3])],
                    ) = evens + odds;

                    next_numerators_col0.push(numerators_c0);
                    next_numerators_col1.push(numerators_c1);
                    next_numerators_col2.push(numerators_c2);
                    next_numerators_col3.push(numerators_c3);

                    let PackedQM31(
                        [PackedCM31([denominators_c0, denominators_c1]), PackedCM31([denominators_c2, denominators_c3])],
                    ) = evens * odds;

                    next_denominators_col0.push(denominators_c0);
                    next_denominators_col1.push(denominators_c1);
                    next_denominators_col2.push(denominators_c2);
                    next_denominators_col3.push(denominators_c3);
                }
            }
            LogupTrace::Multiplicities {
                numerators,
                denominators,
            } => {
                for i in 0..packed_midpoint {
                    let (numerator_evens, numerator_odds) =
                        numerators.data[i * 2].deinterleave_with(numerators.data[i * 2 + 1]);

                    let (denom_c0_evens, denom_c0_odds) = denominators.cols[0].data[i * 2]
                        .deinterleave_with(denominators.cols[0].data[i * 2 + 1]);
                    let (denom_c1_evens, denom_c1_odds) = denominators.cols[1].data[i * 2]
                        .deinterleave_with(denominators.cols[1].data[i * 2 + 1]);
                    let (denom_c2_evens, denom_c2_odds) = denominators.cols[2].data[i * 2]
                        .deinterleave_with(denominators.cols[2].data[i * 2 + 1]);
                    let (denom_c3_evens, denom_c3_odds) = denominators.cols[3].data[i * 2]
                        .deinterleave_with(denominators.cols[3].data[i * 2 + 1]);

                    let denominator_evens = PackedQM31([
                        PackedCM31([denom_c0_evens, denom_c1_evens]),
                        PackedCM31([denom_c2_evens, denom_c3_evens]),
                    ]);

                    let denominator_odds = PackedQM31([
                        PackedCM31([denom_c0_odds, denom_c1_odds]),
                        PackedCM31([denom_c2_odds, denom_c3_odds]),
                    ]);

                    let PackedQM31(
                        [PackedCM31([numerators_c0, numerators_c1]), PackedCM31([numerators_c2, numerators_c3])],
                    ) = denominator_odds * numerator_evens + denominator_evens * numerator_odds;

                    next_numerators_col0.push(numerators_c0);
                    next_numerators_col1.push(numerators_c1);
                    next_numerators_col2.push(numerators_c2);
                    next_numerators_col3.push(numerators_c3);

                    let PackedQM31(
                        [PackedCM31([denominators_c0, denominators_c1]), PackedCM31([denominators_c2, denominators_c3])],
                    ) = denominator_odds * denominator_evens;

                    next_denominators_col0.push(denominators_c0);
                    next_denominators_col1.push(denominators_c1);
                    next_denominators_col2.push(denominators_c2);
                    next_denominators_col3.push(denominators_c3);
                }
            }
            LogupTrace::Generic {
                numerators,
                denominators,
            } => {
                for i in 0..packed_midpoint {
                    let (numer_c0_evens, numer_c0_odds) = numerators.cols[0].data[i * 2]
                        .deinterleave_with(numerators.cols[0].data[i * 2 + 1]);
                    let (numer_c1_evens, numer_c1_odds) = numerators.cols[1].data[i * 2]
                        .deinterleave_with(numerators.cols[1].data[i * 2 + 1]);
                    let (numer_c2_evens, numer_c2_odds) = numerators.cols[2].data[i * 2]
                        .deinterleave_with(numerators.cols[2].data[i * 2 + 1]);
                    let (numer_c3_evens, numer_c3_odds) = numerators.cols[3].data[i * 2]
                        .deinterleave_with(numerators.cols[3].data[i * 2 + 1]);

                    let numerator_evens = PackedQM31([
                        PackedCM31([numer_c0_evens, numer_c1_evens]),
                        PackedCM31([numer_c2_evens, numer_c3_evens]),
                    ]);

                    let numerator_odds = PackedQM31([
                        PackedCM31([numer_c0_odds, numer_c1_odds]),
                        PackedCM31([numer_c2_odds, numer_c3_odds]),
                    ]);

                    let (denom_c0_evens, denom_c0_odds) = denominators.cols[0].data[i * 2]
                        .deinterleave_with(denominators.cols[0].data[i * 2 + 1]);
                    let (denom_c1_evens, denom_c1_odds) = denominators.cols[1].data[i * 2]
                        .deinterleave_with(denominators.cols[1].data[i * 2 + 1]);
                    let (denom_c2_evens, denom_c2_odds) = denominators.cols[2].data[i * 2]
                        .deinterleave_with(denominators.cols[2].data[i * 2 + 1]);
                    let (denom_c3_evens, denom_c3_odds) = denominators.cols[3].data[i * 2]
                        .deinterleave_with(denominators.cols[3].data[i * 2 + 1]);

                    let denominator_evens = PackedQM31([
                        PackedCM31([denom_c0_evens, denom_c1_evens]),
                        PackedCM31([denom_c2_evens, denom_c3_evens]),
                    ]);

                    let denominator_odds = PackedQM31([
                        PackedCM31([denom_c0_odds, denom_c1_odds]),
                        PackedCM31([denom_c2_odds, denom_c3_odds]),
                    ]);

                    let PackedQM31(
                        [PackedCM31([numerators_c0, numerators_c1]), PackedCM31([numerators_c2, numerators_c3])],
                    ) = numerator_evens * denominator_odds + numerator_odds * denominator_evens;

                    next_numerators_col0.push(numerators_c0);
                    next_numerators_col1.push(numerators_c1);
                    next_numerators_col2.push(numerators_c2);
                    next_numerators_col3.push(numerators_c3);

                    let PackedQM31(
                        [PackedCM31([denominators_c0, denominators_c1]), PackedCM31([denominators_c2, denominators_c3])],
                    ) = denominator_odds * denominator_evens;

                    next_denominators_col0.push(denominators_c0);
                    next_denominators_col1.push(denominators_c1);
                    next_denominators_col2.push(denominators_c2);
                    next_denominators_col3.push(denominators_c3);
                }
            }
        }

        let length = packed_midpoint * K_BLOCK_SIZE;

        let next_numerators = SecureColumn {
            cols: [
                BaseFieldVec {
                    data: next_numerators_col0,
                    length,
                },
                BaseFieldVec {
                    data: next_numerators_col1,
                    length,
                },
                BaseFieldVec {
                    data: next_numerators_col2,
                    length,
                },
                BaseFieldVec {
                    data: next_numerators_col3,
                    length,
                },
            ],
        };

        let next_denominators = SecureColumn {
            cols: [
                BaseFieldVec {
                    data: next_denominators_col0,
                    length,
                },
                BaseFieldVec {
                    data: next_denominators_col1,
                    length,
                },
                BaseFieldVec {
                    data: next_denominators_col2,
                    length,
                },
                BaseFieldVec {
                    data: next_denominators_col3,
                    length,
                },
            ],
        };

        LogupTrace::Generic {
            numerators: Mle::new(next_numerators),
            denominators: Mle::new(next_denominators),
        }
    }

    fn univariate_sum_evals(oracle: &LogupOracle<'_, Self>) -> UnivariateEvals {
        let num_terms = 1 << (oracle.num_variables() - 1);
        println!("NUM TERMS: {num_terms}");
        let lambda = oracle.lambda();
        let eq_evals = oracle.eq_evals();
        let trace = oracle.trace();

        // Fallback to CPU backend for small instances to avoid complexity with packed AVX types.
        if num_terms < 2 * K_BLOCK_SIZE {
            let eq_evals = {
                let mut evals = Vec::new();

                for i in 0..usize::min(K_BLOCK_SIZE, eq_evals.len()) {
                    let eq_eval = SecureField::from_m31_array([
                        eq_evals.cols[0].as_slice()[i],
                        eq_evals.cols[1].as_slice()[i],
                        eq_evals.cols[2].as_slice()[i],
                        eq_evals.cols[3].as_slice()[i],
                    ]);

                    evals.push(eq_eval)
                }

                evals
            };

            let trace = trace.to_cpu();
            let oracle = trace.into_sumcheck_oracle(lambda, oracle.z(), &eq_evals);
            return CPUBackend::univariate_sum_evals(&oracle);
        }

        let lambda = PackedQM31::broadcast(lambda);

        let num_packed_terms = num_terms / K_BLOCK_SIZE;

        let mut packed_eval_at_0 = PackedQM31::zeroed();
        let mut packed_eval_at_2 = PackedQM31::zeroed();

        match trace {
            LogupTrace::Singles { denominators } => {
                let col0 = &denominators.cols[0];
                let col1 = &denominators.cols[1];
                let col2 = &denominators.cols[2];
                let col3 = &denominators.cols[3];

                for i in 0..num_packed_terms {
                    // NOTE: The deinterleaves can be avoided by changing the wiring of the GKR
                    // circuit. Instead of inputs being neighbouring even and
                    // odd pairs have LHS and RHS pairs.
                    let (c0_lhs_evens, c0_lhs_odds) =
                        col0.data[i * 2].deinterleave_with(col0.data[i * 2 + 1]);
                    let (c1_lhs_evens, c1_lhs_odds) =
                        col1.data[i * 2].deinterleave_with(col1.data[i * 2 + 1]);
                    let (c2_lhs_evens, c2_lhs_odds) =
                        col2.data[i * 2].deinterleave_with(col2.data[i * 2 + 1]);
                    let (c3_lhs_evens, c3_lhs_odds) =
                        col3.data[i * 2].deinterleave_with(col3.data[i * 2 + 1]);

                    let lhs_evens = PackedQM31([
                        PackedCM31([c0_lhs_evens, c1_lhs_evens]),
                        PackedCM31([c2_lhs_evens, c3_lhs_evens]),
                    ]);

                    let lhs_odds = PackedQM31([
                        PackedCM31([c0_lhs_odds, c1_lhs_odds]),
                        PackedCM31([c2_lhs_odds, c3_lhs_odds]),
                    ]);

                    let (c0_rhs_evens, c0_rhs_odds) = col0.data[(num_packed_terms + i) * 2]
                        .deinterleave_with(col0.data[(num_packed_terms + i) * 2 + 1]);
                    let (c1_rhs_evens, c1_rhs_odds) = col1.data[(num_packed_terms + i) * 2]
                        .deinterleave_with(col1.data[(num_packed_terms + i) * 2 + 1]);
                    let (c2_rhs_evens, c2_rhs_odds) = col2.data[(num_packed_terms + i) * 2]
                        .deinterleave_with(col2.data[(num_packed_terms + i) * 2 + 1]);
                    let (c3_rhs_evens, c3_rhs_odds) = col3.data[(num_packed_terms + i) * 2]
                        .deinterleave_with(col3.data[(num_packed_terms + i) * 2 + 1]);

                    let rhs_evens = PackedQM31([
                        PackedCM31([c0_rhs_evens, c1_rhs_evens]),
                        PackedCM31([c2_rhs_evens, c3_rhs_evens]),
                    ]);

                    let rhs_odds = PackedQM31([
                        PackedCM31([c0_rhs_odds, c1_rhs_odds]),
                        PackedCM31([c2_rhs_odds, c3_rhs_odds]),
                    ]);

                    let product2 = {
                        let d0 = rhs_evens.double() - lhs_evens;
                        let d1 = rhs_odds.double() - lhs_odds;

                        let numerator = d0 + d1;
                        let denominator = d0 * d1;

                        numerator + lambda * denominator
                    };

                    let product0 = {
                        let numerator = lhs_evens + lhs_odds;
                        let denominator = lhs_evens * lhs_odds;
                        numerator + lambda * denominator
                    };

                    let eq_eval = PackedQM31([
                        PackedCM31([eq_evals.cols[0].data[i], eq_evals.cols[1].data[i]]),
                        PackedCM31([eq_evals.cols[2].data[i], eq_evals.cols[3].data[i]]),
                    ]);

                    packed_eval_at_0 += eq_eval * product0;
                    packed_eval_at_2 += eq_eval * product2;
                }
            }
            LogupTrace::Multiplicities {
                numerators,
                denominators,
            } => {
                let denom_col0 = &denominators.cols[0];
                let denom_col1 = &denominators.cols[1];
                let denom_col2 = &denominators.cols[2];
                let denom_col3 = &denominators.cols[3];

                for i in 0..num_packed_terms {
                    let (numerator_lhs_evens, numerator_lhs_odds) =
                        numerators.data[i * 2].deinterleave_with(numerators.data[i * 2 + 1]);

                    let (denom_c0_lhs_evens, denom_c0_lhs_odds) =
                        denom_col0.data[i * 2].deinterleave_with(denom_col0.data[i * 2 + 1]);
                    let (denom_c1_lhs_evens, denom_c1_lhs_odds) =
                        denom_col1.data[i * 2].deinterleave_with(denom_col1.data[i * 2 + 1]);
                    let (denom_c2_lhs_evens, denom_c2_lhs_odds) =
                        denom_col2.data[i * 2].deinterleave_with(denom_col2.data[i * 2 + 1]);
                    let (denom_c3_lhs_evens, denom_c3_lhs_odds) =
                        denom_col3.data[i * 2].deinterleave_with(denom_col3.data[i * 2 + 1]);

                    let denom_lhs_evens = PackedQM31([
                        PackedCM31([denom_c0_lhs_evens, denom_c1_lhs_evens]),
                        PackedCM31([denom_c2_lhs_evens, denom_c3_lhs_evens]),
                    ]);

                    let denom_lhs_odds = PackedQM31([
                        PackedCM31([denom_c0_lhs_odds, denom_c1_lhs_odds]),
                        PackedCM31([denom_c2_lhs_odds, denom_c3_lhs_odds]),
                    ]);

                    let (numerator_rhs_evens, numerator_rhs_odds) = numerators.data
                        [(num_packed_terms + i) * 2]
                        .deinterleave_with(numerators.data[(num_packed_terms + i) * 2 + 1]);

                    let (denom_c0_rhs_evens, denom_c0_rhs_odds) = denom_col0.data
                        [(num_packed_terms + i) * 2]
                        .deinterleave_with(denom_col0.data[(num_packed_terms + i) * 2 + 1]);
                    let (denom_c1_rhs_evens, denom_c1_rhs_odds) = denom_col1.data
                        [(num_packed_terms + i) * 2]
                        .deinterleave_with(denom_col1.data[(num_packed_terms + i) * 2 + 1]);
                    let (denom_c2_rhs_evens, denom_c2_rhs_odds) = denom_col2.data
                        [(num_packed_terms + i) * 2]
                        .deinterleave_with(denom_col2.data[(num_packed_terms + i) * 2 + 1]);
                    let (denom_c3_rhs_evens, denom_c3_rhs_odds) = denom_col3.data
                        [(num_packed_terms + i) * 2]
                        .deinterleave_with(denom_col3.data[(num_packed_terms + i) * 2 + 1]);

                    let denom_rhs_evens = PackedQM31([
                        PackedCM31([denom_c0_rhs_evens, denom_c1_rhs_evens]),
                        PackedCM31([denom_c2_rhs_evens, denom_c3_rhs_evens]),
                    ]);

                    let denom_rhs_odds = PackedQM31([
                        PackedCM31([denom_c0_rhs_odds, denom_c1_rhs_odds]),
                        PackedCM31([denom_c2_rhs_odds, denom_c3_rhs_odds]),
                    ]);

                    let product2 = {
                        let n0 = numerator_rhs_evens.double() - numerator_lhs_evens;
                        let n1 = numerator_rhs_odds.double() - numerator_lhs_odds;
                        let d0 = denom_rhs_evens.double() - denom_lhs_evens;
                        let d1 = denom_rhs_odds.double() - denom_lhs_odds;

                        let numerator = d1 * n0 + d0 * n1;
                        let denominator = d0 * d1;

                        numerator + lambda * denominator
                    };

                    let product0 = {
                        let numerator = denom_lhs_odds * numerator_lhs_evens
                            + denom_lhs_evens * numerator_lhs_odds;
                        let denominator = denom_lhs_evens * denom_lhs_odds;
                        numerator + lambda * denominator
                    };

                    let eq_eval = PackedQM31([
                        PackedCM31([eq_evals.cols[0].data[i], eq_evals.cols[1].data[i]]),
                        PackedCM31([eq_evals.cols[2].data[i], eq_evals.cols[3].data[i]]),
                    ]);

                    packed_eval_at_0 += eq_eval * product0;
                    packed_eval_at_2 += eq_eval * product2;
                }
            }
            LogupTrace::Generic {
                numerators,
                denominators,
            } => {
                let numer_col0 = &numerators.cols[0];
                let numer_col1 = &numerators.cols[1];
                let numer_col2 = &numerators.cols[2];
                let numer_col3 = &numerators.cols[3];

                let denom_col0 = &denominators.cols[0];
                let denom_col1 = &denominators.cols[1];
                let denom_col2 = &denominators.cols[2];
                let denom_col3 = &denominators.cols[3];

                for i in 0..num_packed_terms {
                    let (numer_c0_lhs_evens, numer_c0_lhs_odds) =
                        numer_col0.data[i * 2].deinterleave_with(numer_col0.data[i * 2 + 1]);
                    let (numer_c1_lhs_evens, numer_c1_lhs_odds) =
                        numer_col1.data[i * 2].deinterleave_with(numer_col1.data[i * 2 + 1]);
                    let (numer_c2_lhs_evens, numer_c2_lhs_odds) =
                        numer_col2.data[i * 2].deinterleave_with(numer_col2.data[i * 2 + 1]);
                    let (numer_c3_lhs_evens, numer_c3_lhs_odds) =
                        numer_col3.data[i * 2].deinterleave_with(numer_col3.data[i * 2 + 1]);

                    let numer_lhs_evens = PackedQM31([
                        PackedCM31([numer_c0_lhs_evens, numer_c1_lhs_evens]),
                        PackedCM31([numer_c2_lhs_evens, numer_c3_lhs_evens]),
                    ]);

                    let numer_lhs_odds = PackedQM31([
                        PackedCM31([numer_c0_lhs_odds, numer_c1_lhs_odds]),
                        PackedCM31([numer_c2_lhs_odds, numer_c3_lhs_odds]),
                    ]);

                    let (denom_c0_lhs_evens, denom_c0_lhs_odds) =
                        denom_col0.data[i * 2].deinterleave_with(denom_col0.data[i * 2 + 1]);
                    let (denom_c1_lhs_evens, denom_c1_lhs_odds) =
                        denom_col1.data[i * 2].deinterleave_with(denom_col1.data[i * 2 + 1]);
                    let (denom_c2_lhs_evens, denom_c2_lhs_odds) =
                        denom_col2.data[i * 2].deinterleave_with(denom_col2.data[i * 2 + 1]);
                    let (denom_c3_lhs_evens, denom_c3_lhs_odds) =
                        denom_col3.data[i * 2].deinterleave_with(denom_col3.data[i * 2 + 1]);

                    let denom_lhs_evens = PackedQM31([
                        PackedCM31([denom_c0_lhs_evens, denom_c1_lhs_evens]),
                        PackedCM31([denom_c2_lhs_evens, denom_c3_lhs_evens]),
                    ]);

                    let denom_lhs_odds = PackedQM31([
                        PackedCM31([denom_c0_lhs_odds, denom_c1_lhs_odds]),
                        PackedCM31([denom_c2_lhs_odds, denom_c3_lhs_odds]),
                    ]);

                    let (numer_c0_rhs_evens, numer_c0_rhs_odds) = numer_col0.data
                        [(num_packed_terms + i) * 2]
                        .deinterleave_with(numer_col0.data[(num_packed_terms + i) * 2 + 1]);
                    let (numer_c1_rhs_evens, numer_c1_rhs_odds) = numer_col1.data
                        [(num_packed_terms + i) * 2]
                        .deinterleave_with(numer_col1.data[(num_packed_terms + i) * 2 + 1]);
                    let (numer_c2_rhs_evens, numer_c2_rhs_odds) = numer_col2.data
                        [(num_packed_terms + i) * 2]
                        .deinterleave_with(numer_col2.data[(num_packed_terms + i) * 2 + 1]);
                    let (numer_c3_rhs_evens, numer_c3_rhs_odds) = numer_col3.data
                        [(num_packed_terms + i) * 2]
                        .deinterleave_with(numer_col3.data[(num_packed_terms + i) * 2 + 1]);

                    let numer_rhs_evens = PackedQM31([
                        PackedCM31([numer_c0_rhs_evens, numer_c1_rhs_evens]),
                        PackedCM31([numer_c2_rhs_evens, numer_c3_rhs_evens]),
                    ]);

                    let numer_rhs_odds = PackedQM31([
                        PackedCM31([numer_c0_rhs_odds, numer_c1_rhs_odds]),
                        PackedCM31([numer_c2_rhs_odds, numer_c3_rhs_odds]),
                    ]);

                    let (denom_c0_rhs_evens, denom_c0_rhs_odds) = denom_col0.data
                        [(num_packed_terms + i) * 2]
                        .deinterleave_with(denom_col0.data[(num_packed_terms + i) * 2 + 1]);
                    let (denom_c1_rhs_evens, denom_c1_rhs_odds) = denom_col1.data
                        [(num_packed_terms + i) * 2]
                        .deinterleave_with(denom_col1.data[(num_packed_terms + i) * 2 + 1]);
                    let (denom_c2_rhs_evens, denom_c2_rhs_odds) = denom_col2.data
                        [(num_packed_terms + i) * 2]
                        .deinterleave_with(denom_col2.data[(num_packed_terms + i) * 2 + 1]);
                    let (denom_c3_rhs_evens, denom_c3_rhs_odds) = denom_col3.data
                        [(num_packed_terms + i) * 2]
                        .deinterleave_with(denom_col3.data[(num_packed_terms + i) * 2 + 1]);

                    let denom_rhs_evens = PackedQM31([
                        PackedCM31([denom_c0_rhs_evens, denom_c1_rhs_evens]),
                        PackedCM31([denom_c2_rhs_evens, denom_c3_rhs_evens]),
                    ]);

                    let denom_rhs_odds = PackedQM31([
                        PackedCM31([denom_c0_rhs_odds, denom_c1_rhs_odds]),
                        PackedCM31([denom_c2_rhs_odds, denom_c3_rhs_odds]),
                    ]);

                    let product2 = {
                        let n0 = numer_rhs_evens.double() - numer_lhs_evens;
                        let n1 = numer_rhs_odds.double() - numer_lhs_odds;
                        let d0 = denom_rhs_evens.double() - denom_lhs_evens;
                        let d1 = denom_rhs_odds.double() - denom_lhs_odds;

                        let numerator = d1 * n0 + d0 * n1;
                        let denominator = d0 * d1;

                        numerator + lambda * denominator
                    };

                    let product0 = {
                        let numerator =
                            denom_lhs_odds * numer_lhs_evens + denom_lhs_evens * numer_lhs_odds;
                        let denominator = denom_lhs_evens * denom_lhs_odds;
                        numerator + lambda * denominator
                    };

                    let eq_eval = PackedQM31([
                        PackedCM31([eq_evals.cols[0].data[i], eq_evals.cols[1].data[i]]),
                        PackedCM31([eq_evals.cols[2].data[i], eq_evals.cols[3].data[i]]),
                    ]);

                    packed_eval_at_0 += eq_eval * product0;
                    packed_eval_at_2 += eq_eval * product2;
                }
            }
        }

        let eval_at_0 = packed_eval_at_0.to_array().into_iter().sum::<SecureField>();
        let eval_at_2 = packed_eval_at_2.to_array().into_iter().sum::<SecureField>();

        UnivariateEvals {
            eval_at_0,
            eval_at_2,
        }
    }
}

impl LogupTrace<AVX512Backend> {
    fn to_cpu(&self) -> LogupTrace<CPUBackend> {
        fn to_cpu_mle<F: Field>(mle: &AvxMle<F>) -> CpuMle<F>
        where
            AVX512Backend: MleOps<F>,
            CPUBackend: MleOps<F>,
        {
            CpuMle::new(mle.to_vec().into_iter().collect())
        }

        match self {
            LogupTrace::Singles { denominators } => LogupTrace::Singles {
                denominators: to_cpu_mle(denominators),
            },
            LogupTrace::Multiplicities {
                numerators,
                denominators,
            } => LogupTrace::Multiplicities {
                numerators: to_cpu_mle(numerators),
                denominators: to_cpu_mle(denominators),
            },
            LogupTrace::Generic {
                numerators,
                denominators,
            } => LogupTrace::Generic {
                numerators: to_cpu_mle(numerators),
                denominators: to_cpu_mle(denominators),
            },
        }
    }
}

impl LogupTrace<CPUBackend> {
    fn to_avx(&self) -> LogupTrace<AVX512Backend> {
        fn to_avx_mle<F: Field>(mle: &CpuMle<F>) -> AvxMle<F>
        where
            AVX512Backend: MleOps<F>,
            CPUBackend: MleOps<F>,
        {
            AvxMle::new(mle.to_vec().into_iter().collect())
        }

        match self {
            LogupTrace::Singles { denominators } => LogupTrace::Singles {
                denominators: to_avx_mle(denominators),
            },
            LogupTrace::Multiplicities {
                numerators,
                denominators,
            } => LogupTrace::Multiplicities {
                numerators: to_avx_mle(numerators),
                denominators: to_avx_mle(denominators),
            },
            LogupTrace::Generic {
                numerators,
                denominators,
            } => LogupTrace::Generic {
                numerators: to_avx_mle(numerators),
                denominators: to_avx_mle(denominators),
            },
        }
    }
}
