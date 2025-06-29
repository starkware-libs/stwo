use web_sys::console;

use super::eval_composition_poly::{compute_composition_polynomial_wgpu, GpuContext};
use super::{ComputeCompositionPolynomialInput, ComputeCompositionPolynomialOutput};

#[allow(dead_code)]
pub async fn runner_eval_composition_polynomial(
    request_rx: flume::Receiver<Box<ComputeCompositionPolynomialInput>>,
    response_tx: flume::Sender<Box<ComputeCompositionPolynomialOutput>>,
) {
    let gpu = GpuContext::new().await;

    let input_data = request_rx.recv_async().await.unwrap();

    console::time_with_label("wgpu-runner-timer");
    let output_data = compute_composition_polynomial_wgpu(input_data, &gpu).await;

    response_tx.send(output_data).unwrap();
    console::time_end_with_label("wgpu-runner-timer");
}
