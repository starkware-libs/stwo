use std::fs::File;
use std::io::Write;
use stwo::core::fields::m31::BaseField;
use stwo::core::utils::bit_reverse_coset_to_circle_domain_order;
use stwo::prover::backend::simd::SimdBackend;
use stwo::prover::backend::{Col, Column};

/// Dumps trace columns to a file for inspection
/// Shows both before and after bit-reverse order
pub fn dump_trace_to_file(
    log_size: u32,
    initial_a: u32,
    initial_b: u32,
    target_element: usize,
    filename: &str,
) -> Result<(), std::io::Error> {
    let n_rows = 1 << log_size;

    let mut col_a = Col::<SimdBackend, BaseField>::zeros(n_rows);
    let mut col_b = Col::<SimdBackend, BaseField>::zeros(n_rows);
    let mut col_c = Col::<SimdBackend, BaseField>::zeros(n_rows);

    let mut a = BaseField::from_u32_unchecked(initial_a);
    let mut b = BaseField::from_u32_unchecked(initial_b);

    // Generate Fibonacci sequence ONLY up to target_element
    let rows_to_compute = (target_element + 1).min(n_rows);

    for row in 0..rows_to_compute {
        let c = a + b;

        col_a.set(row, a);
        col_b.set(row, b);
        col_c.set(row, c);

        a = b;
        b = c;
    }

    // Rows from rows_to_compute..n_rows remain zeros

    // Create output file
    let mut file = File::create(filename)?;

    writeln!(file, "=== MULTI_FIB TRACE DUMP ===")?;
    writeln!(file, "log_size: {}", log_size)?;
    writeln!(file, "n_rows: {}", n_rows)?;
    writeln!(file, "initial_a: {}", initial_a)?;
    writeln!(file, "initial_b: {}", initial_b)?;
    writeln!(file, "target_element: {}", target_element)?;
    writeln!(file, "rows_to_compute: {}", rows_to_compute)?;
    writeln!(file)?;

    // Dump BEFORE bit-reverse
    writeln!(file, "=== BEFORE BIT-REVERSE (Natural Order) ===")?;
    writeln!(file, "Row | A | B | C")?;
    writeln!(file, "----|---|---|---")?;
    for row in 0..n_rows {
        let a_val = col_a.at(row).0;
        let b_val = col_b.at(row).0;
        let c_val = col_c.at(row).0;
        writeln!(file, "{:3} | {:8} | {:8} | {:8}", row, a_val, b_val, c_val)?;
    }
    writeln!(file)?;

    // Apply bit-reverse
    bit_reverse_coset_to_circle_domain_order(col_a.as_mut_slice());
    bit_reverse_coset_to_circle_domain_order(col_b.as_mut_slice());
    bit_reverse_coset_to_circle_domain_order(col_c.as_mut_slice());

    // Dump AFTER bit-reverse
    writeln!(file, "=== AFTER BIT-REVERSE (Circle Domain Order) ===")?;
    writeln!(file, "Row | A | B | C")?;
    writeln!(file, "----|---|---|---")?;
    for row in 0..n_rows {
        let a_val = col_a.at(row).0;
        let b_val = col_b.at(row).0;
        let c_val = col_c.at(row).0;
        writeln!(file, "{:3} | {:8} | {:8} | {:8}", row, a_val, b_val, c_val)?;
    }
    writeln!(file)?;

    // Summary statistics
    writeln!(file, "=== SUMMARY ===")?;
    let zero_rows_before = (rows_to_compute..n_rows).count();
    writeln!(file, "Active rows: {}", rows_to_compute)?;
    writeln!(file, "Padding rows (all zeros): {}", zero_rows_before)?;
    writeln!(file, "Padding percentage: {:.2}%", (zero_rows_before as f64 / n_rows as f64) * 100.0)?;

    println!("Trace dumped to: {}", filename);
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dump_multi_fib_trace() {
        // Test case 1: Small target_element (lots of padding)
        dump_trace_to_file(4, 1, 1, 5, "multi_fib_trace_small.txt")
            .expect("Failed to dump trace");

        // Test case 2: Larger target_element (less padding)
        dump_trace_to_file(4, 1, 1, 10, "multi_fib_trace_medium.txt")
            .expect("Failed to dump trace");

        // Test case 3: Full utilization (no padding)
        dump_trace_to_file(4, 1, 1, 15, "multi_fib_trace_full.txt")
            .expect("Failed to dump trace");

        println!("\n=== Trace dumps created successfully! ===");
        println!("Check the following files:");
        println!("  - multi_fib_trace_small.txt  (target=5, lots of padding)");
        println!("  - multi_fib_trace_medium.txt (target=10, some padding)");
        println!("  - multi_fib_trace_full.txt   (target=15, no padding)");
    }
}
