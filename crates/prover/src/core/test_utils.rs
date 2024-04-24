use stwo_verifier::core::fields::m31::BaseField;
use stwo_verifier::core::fields::qm31::SecureField;

use super::backend::cpu::CPUCircleEvaluation;
use super::channel::Blake2sChannel;
use crate::core::channel::Channel;

pub fn secure_eval_to_base_eval<EvalOrder>(
    eval: &CPUCircleEvaluation<SecureField, EvalOrder>,
) -> CPUCircleEvaluation<BaseField, EvalOrder> {
    CPUCircleEvaluation::new(
        eval.domain,
        eval.values.iter().map(|x| x.to_m31_array()[0]).collect(),
    )
}

pub fn test_channel() -> Blake2sChannel {
    use crate::commitment_scheme::blake2_hash::Blake2sHash;

    let seed = Blake2sHash::from(vec![0; 32]);
    Blake2sChannel::new(seed)
}
