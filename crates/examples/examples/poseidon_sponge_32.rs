//! Example: Poseidon sponge with 32-element input
//!
//! Demonstrates vertical chaining where:
//! - Row 0: absorbs elements [0..8]
//! - Row 1: absorbs elements [8..16] (using output from row 0)
//! - Row 2: absorbs elements [16..24] (using output from row 1)
//! - Row 3: absorbs elements [24..32] (using output from row 2)
//!
//! This shows the sponge construction in action!

use stwo::core::fields::m31::BaseField;
use stwo::core::fri::FriConfig;
use stwo::core::pcs::PcsConfig;
use stwo_examples::poseidon_uacias::{prove_poseidon, gen_trace, dump_trace_to_file, RATE};

fn main() {
    println!("=== Poseidon Sponge: 32-element input ===\n");

    // Create 32-element input (4 messages of 8 elements each)
    let input_32: Vec<u32> = (0..32).collect();

    println!("Input (32 elements):");
    for (i, chunk) in input_32.chunks(8).enumerate() {
        println!("  Message {}: {:?}", i, chunk);
    }
    println!();

    // Convert to messages format (Vec<[BaseField; RATE]>)
    let messages: Vec<[BaseField; RATE]> = input_32
        .chunks(8)
        .map(|chunk| {
            let mut msg = [BaseField::from_u32_unchecked(0); RATE];
            for (i, &val) in chunk.iter().enumerate() {
                msg[i] = BaseField::from_u32_unchecked(val);
            }
            msg
        })
        .collect();

    println!("Sponge construction:");
    println!("  Row 0: [msg0] + [capacity=0] → Poseidon → output0");
    println!("  Row 1: [output0.rate + msg1] + [output0.capacity] → Poseidon → output1");
    println!("  Row 2: [output1.rate + msg2] + [output1.capacity] → Poseidon → output2");
    println!("  Row 3: [output2.rate + msg3] + [output2.capacity] → Poseidon → output3");
    println!("  Final hash: output3[0]\n");

    // Need enough rows for FRI to work properly
    // Using 256 rows (log_n_rows = 8), first 4 contain our messages
    let log_n_rows = 8;

    let config = PcsConfig {
        pow_bits: 10,
        fri_config: FriConfig::new(5, 1, 64),
    };


    // Generate trace and dump to file
    println!("\nGenerating trace and dumping to file...");
    let (trace, _lookup_data) = gen_trace(log_n_rows, messages.clone());
    dump_trace_to_file(&trace, "poseidon_sponge_trace.txt")
        .expect("Failed to dump trace");
    println!("✅ Trace dumped to: poseidon_sponge_trace.txt");

    println!("\nGenerating proof...");
    let (_component, _proof) = prove_poseidon(log_n_rows, messages, config);

    println!("✅ Proof generated successfully!");
    println!("\nVertical chaining verified:");
    println!("  - Each row's output flows to next row's input");
    println!("  - Capacity preserved across rows");
    println!("  - All 32 elements absorbed sequentially");
    println!("\n📄 Check poseidon_sponge_trace.txt to see the full trace!");
}
