use wasm_bindgen::prelude::*;

mod execute_gpu;
use js_sys::Promise;
use wasm_bindgen_futures::future_to_promise;

use crate::wasm::execute_gpu::run;

// #[wasm_bindgen]
// pub fn run_test(log_size: u32) {
//     wasm_bindgen_futures::spawn_local(run(log_size));
// }

#[wasm_bindgen]
pub fn run_test(log_size: u32) -> Promise {
    future_to_promise(run_wrapper(log_size))
}

async fn run_wrapper(log_size: u32) -> Result<JsValue, JsValue> {
    run(log_size).await;
    Ok(JsValue::null())
}
