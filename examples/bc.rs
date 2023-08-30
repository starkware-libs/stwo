#![feature(sync_unsafe_cell)]
#![feature(portable_simd)]
#![feature(stdsimd)]
#![feature(iter_array_chunks)]

use prover_research::benches::commitment::run_standalone;

fn main() {
    run_standalone();
}
