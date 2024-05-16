use super::backend::cpu::CpuCircleEvaluation;
use super::channel::Blake2sChannel;
use super::fields::m31::BaseField;
use super::fields::qm31::SecureField;
use crate::core::channel::Channel;

pub fn secure_eval_to_base_eval<EvalOrder>(
    eval: &CpuCircleEvaluation<SecureField, EvalOrder>,
) -> CpuCircleEvaluation<BaseField, EvalOrder> {
    CpuCircleEvaluation::new(
        eval.domain,
        eval.values.iter().map(|x| x.to_m31_array()[0]).collect(),
    )
}

pub fn test_channel() -> Blake2sChannel {
    use crate::core::vcs::blake2_hash::Blake2sHash;

    let seed = Blake2sHash::from(vec![0; 32]);
    Blake2sChannel::new(seed)
}
