#[cfg(all(feature = "parallel", target_family = "wasm", not(target_os = "wasi")))]
pub use wasm_bindgen_rayon::init_thread_pool;

#[cfg(all(feature = "parallel", target_family = "wasm", not(target_os = "wasi")))]
async fn rayon_init_thread_pool(num_threads: usize) {
    let promise = init_thread_pool(num_threads);
    // Alternatively, num_threads can be set dynamically using hardware_concurrency()
    if let Err(err) = wasm_bindgen_futures::JsFuture::from(promise).await {
        web_sys::console::error_1(&format!("Failed to start pool: {:?}", err).into());
    } else {
        web_sys::console::info_1(
            &format!("Rayon pool started with {} threads", num_threads).into(),
        );
    }
}

#[cfg(all(target_family = "wasm", not(target_os = "wasi")))]
pub async fn init_wasm_mt(num_threads: usize) {
    #[cfg(feature = "parallel")]
    rayon_init_thread_pool(num_threads).await;

    #[cfg(not(feature = "parallel"))]
    web_sys::console::log_1(
        &format!(
            "Parallel feature is not enabled, num_threads: {} will be ignored",
            num_threads
        )
        .into(),
    );
}

#[cfg(all(target_family = "wasm", not(target_os = "wasi")))]
pub fn wasm_timer_now() -> f64 {
    web_sys::window()
        .and_then(|win| win.performance())
        .map(|p| p.now())
        .unwrap_or_else(|| {
            web_sys::console::warn_1(&"window.performance.now() unavailable; using 0.0".into());
            0.0
        })
}
