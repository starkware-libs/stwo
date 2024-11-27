use wasm_bindgen::prelude::*;
// use wasm_bindgen_test::wasm_bindgen_test;
mod execute_gpu;
use crate::wasm::execute_gpu::run;

#[wasm_bindgen]
pub fn run_test(log_size: u32) {
    wasm_bindgen_futures::spawn_local(run(log_size));
}

#[test]
fn test_run() {
    run_test(5);
}

// #[wasm_bindgen_test]
// fn test_run_wasm() {
//     run_test(5);
// }
