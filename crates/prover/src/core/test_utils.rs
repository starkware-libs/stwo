use super::backend::cpu::CpuCircleEvaluation;
use super::channel::{Blake2sChannel, Poseidon252Channel};
use super::fields::m31::BaseField;
use super::fields::qm31::SecureField;

pub fn secure_eval_to_base_eval<EvalOrder>(
    eval: &CpuCircleEvaluation<SecureField, EvalOrder>,
) -> CpuCircleEvaluation<BaseField, EvalOrder> {
    CpuCircleEvaluation::new(
        eval.domain,
        eval.values.iter().map(|x| x.to_m31_array()[0]).collect(),
    )
}

pub fn test_channel() -> Blake2sChannel {
    Blake2sChannel::default()
}

pub fn test_poseidon_channel() -> Poseidon252Channel {
    Poseidon252Channel::default()
}
