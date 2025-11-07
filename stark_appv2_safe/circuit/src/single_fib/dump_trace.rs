use std::fs::File;
use std::io::Write;
use stwo::core::fields::m31::BaseField;
use stwo::prover::backend::simd::SimdBackend;
use stwo::prover::backend::Column;
use stwo::prover::poly::circle::CircleEvaluation;
use stwo::prover::poly::BitReversedOrder;

/// Dumps all trace columns to a file
pub fn dump_trace_to_file(
    filename: &str,
    computing_trace: &[CircleEvaluation<SimdBackend, BaseField, BitReversedOrder>],
    scheduler_trace: &[CircleEvaluation<SimdBackend, BaseField, BitReversedOrder>],
    target_element: usize,
) -> std::io::Result<()> {
    let mut file = File::create(filename)?;

    let log_size = computing_trace[0].domain.log_size();
    let n_rows = 1 << log_size;

    writeln!(file, "=== SINGLE FIBONACCI TRACE DUMP ===")?;
    writeln!(file, "Log size: {}", log_size)?;
    writeln!(file, "Number of rows: {}", n_rows)?;
    writeln!(file, "Target element: {}", target_element)?;
    writeln!(file)?;

    writeln!(file, "=== COMPUTING COMPONENT (3 columns: a, b, c) ===")?;
    writeln!(file, "Row | a | b | c")?;
    writeln!(file, "----+-------+-------+-------")?;

    for row in 0..n_rows {
        let a = computing_trace[0].values.at(row).0;
        let b = computing_trace[1].values.at(row).0;
        let c = computing_trace[2].values.at(row).0;

        writeln!(file, "{:4} | {:5} | {:5} | {:5}", row, a, b, c)?;
    }

    writeln!(file)?;
    writeln!(file, "=== SCHEDULER COMPONENT (1 column: fib_c) ===")?;
    writeln!(file, "Row | fib_c")?;
    writeln!(file, "----+-------")?;

    for row in 0..n_rows {
        let fib_c = scheduler_trace[0].values.at(row).0;

        writeln!(file, "{:4} | {:5}", row, fib_c)?;
    }

    writeln!(file)?;
    writeln!(file, "=== END OF TRACE DUMP ===")?;

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::single_fib::{
        gen_computing_trace, gen_scheduler_trace,
        prove_single_fib, verify_single_fib
    };
    use stwo::core::channel::Blake2sChannel;
    use stwo::core::pcs::PcsConfig;
    use stwo::core::poly::circle::CanonicCoset;
    use stwo::core::vcs::blake2_merkle::Blake2sMerkleChannel;
    use stwo::prover::backend::simd::SimdBackend;
    use stwo::prover::poly::circle::PolyOps;
    use stwo::prover::CommitmentSchemeProver;

    #[test]
    fn test_full_flow_with_dump() {
        println!("\n==================================================");
        println!("  FULL FLOW: TRACE GENERATION + PROVE + VERIFY + DUMP");
        println!("==================================================\n");

        let target_element = 3;

        // Setup prover (log_size will be computed dynamically)
        let config = PcsConfig::default();

        // Compute expected log_size for twiddles
        // let min_log_size = if target_element + 1 <= 1 { 0 } else { (target_element as u32).ilog2() + 1 };
        let log_size =3;


        println!("==================================================");
        println!("STEP 1: TRACE GENERATION");
        println!("==================================================");

        let (trace_computing, fib_c_value) = gen_computing_trace(log_size, 1, 1, target_element);
        let trace_scheduler = gen_scheduler_trace(log_size, fib_c_value);

        println!("\n✓ Traces generated successfully!");
        println!("  Computing trace: {} rows (3 columns: a, b, c)", 1 << log_size);
        println!("  Scheduler trace: {} rows (1 column: fib_c)", 1 << log_size);
        println!("  Target Fibonacci value: {}", fib_c_value.0);

        println!("\n==================================================");
        println!("STEP 2: PROVING");
        println!("==================================================");

        let twiddles = SimdBackend::precompute_twiddles(
            CanonicCoset::new(log_size + 1 + config.fri_config.log_blowup_factor)
                .circle_domain()
                .half_coset,
        );

        let channel = &mut Blake2sChannel::default();
        let commitment_scheme = CommitmentSchemeProver::<SimdBackend, Blake2sMerkleChannel>::new(
            config,
            &twiddles,
        );

        let result = prove_single_fib(target_element, channel, commitment_scheme);

        match result {
            Ok((proof, _component, _scheduler, statement0, statement1)) => {
                println!("\n✓ Proof generated successfully!");
                println!("  - Number of commitments: {}", proof.commitments.len());

                println!("\n==================================================");
                println!("STEP 3: VERIFYING");
                println!("==================================================");

                let verify_result = verify_single_fib(
                    proof,
                    target_element,
                    statement0,
                    statement1,
                    config,
                );

                match verify_result {
                    Ok(()) => {
                        println!("\n✓ Proof verified successfully!");

                        println!("\n==================================================");
                        println!("STEP 4: DUMPING TRACE TO FILE");
                        println!("==================================================");

                        let dump_result = dump_trace_to_file(
                            "single_fib_trace.txt",
                            &trace_computing,
                            &trace_scheduler,
                            target_element,
                        );

                        match dump_result {
                            Ok(()) => {
                                println!("\n✓ Trace dumped to single_fib_trace.txt");

                                println!("\n==================================================");
                                println!("  ✓✓✓ FULL FLOW COMPLETED SUCCESSFULLY! ✓✓✓");
                                println!("==================================================");
                                println!("\nCompleted steps:");
                                println!("  1. ✓ Generated traces for Computing and Scheduler");
                                println!("  2. ✓ Generated STARK proof with LogUp");
                                println!("  3. ✓ Verified proof");
                                println!("  4. ✓ Dumped all rows to file");
                                println!("\nFile created: single_fib_trace.txt");
                            }
                            Err(e) => {
                                panic!("Failed to dump trace: {:?}", e);
                            }
                        }
                    }
                    Err(e) => {
                        panic!("Verification failed: {:?}", e);
                    }
                }
            }
            Err(e) => {
                panic!("Proof generation failed: {:?}", e);
            }
        }
    }

    #[test]
    fn test_small_trace_dump() {
        println!("\n==================================================");
        println!("  SMALL TRACE GENERATION AND DUMP");
        println!("==================================================\n");

        let target_element = 5;
        let log_size = 4; // 16 rows

        println!("Generating small trace with target_element={}...", target_element);

        let (trace_computing, fib_c_value) = gen_computing_trace(log_size, 1, 1, target_element);
        let trace_scheduler = gen_scheduler_trace(log_size, fib_c_value);

        println!("✓ Traces generated!");

        let dump_result = dump_trace_to_file(
            "single_fib_trace_small.txt",
            &trace_computing,
            &trace_scheduler,
            target_element,
        );

        match dump_result {
            Ok(()) => {
                println!("✓ Small trace dumped to single_fib_trace_small.txt");
            }
            Err(e) => {
                panic!("Failed to dump trace: {:?}", e);
            }
        }
    }

    #[test]
    fn test_medium_trace_dump() {
        println!("\n==================================================");
        println!("  MEDIUM TRACE GENERATION AND DUMP");
        println!("==================================================\n");

        let target_element = 20;
        let log_size = 3; // 32 rows

        println!("Generating medium trace with target_element={}...", target_element);

        let (trace_computing, fib_c_value) = gen_computing_trace(log_size, 1, 1, target_element);
        let trace_scheduler = gen_scheduler_trace(log_size, fib_c_value);

        println!("✓ Traces generated!");

        let dump_result = dump_trace_to_file(
            "single_fib_trace_medium.txt",
            &trace_computing,
            &trace_scheduler,
            target_element,
        );

        match dump_result {
            Ok(()) => {
                println!("✓ Medium trace dumped to single_fib_trace_medium.txt");
            }
            Err(e) => {
                panic!("Failed to dump trace: {:?}", e);
            }
        }
    }
}
