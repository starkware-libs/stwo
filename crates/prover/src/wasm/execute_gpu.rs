use crate::core::backend::cpu::CpuCirclePoly;
use crate::core::backend::gpu::circle::{circle_eval_to_gpu_input, interpolate_gpu};
use crate::core::fields::m31::BaseField;
use crate::core::poly::circle::CanonicCoset;

pub async fn run(log_size: u32) {
    let poly = CpuCirclePoly::new((1..=1 << log_size).map(BaseField::from).collect());
    let domain = CanonicCoset::new(log_size).circle_domain();
    let evals = poly.evaluate(domain);
    let input = circle_eval_to_gpu_input(evals, log_size);
    println!("log size: {}", log_size);
    let gpu_output = interpolate_gpu(input).await;

    assert_eq!(
        gpu_output.results.to_vec()[..poly.coeffs.len()],
        poly.coeffs
    );
}
