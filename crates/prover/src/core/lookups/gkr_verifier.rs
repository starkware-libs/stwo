//! GKR batch verifier for Grand Product and LogUp lookup arguments.
use thiserror::Error;

use super::sumcheck::{SumcheckError, SumcheckProof};
use super::utils::{eq, fold_mle_evals, random_linear_combination};
use crate::core::channel::Channel;
use crate::core::fields::m31::BaseField;
use crate::core::fields::qm31::SecureField;
use crate::core::lookups::sumcheck;
use crate::core::lookups::utils::Fraction;

/// Partially verifies a batch GKR proof.
///
/// On successful verification the function returns a [`GkrArtifact`] which stores the out-of-domain
/// point and claimed evaluations in the input layer columns for each instance at the OOD point.
/// These claimed evaluations are not checked in this function - hence partial verification.
pub fn partially_verify_batch(
    gate_by_instance: Vec<Gate>,
    proof: &GkrBatchProof,
    channel: &mut impl Channel,
) -> Result<GkrArtifact, GkrError> {
    let GkrBatchProof {
        sumcheck_proofs,
        layer_masks_by_instance,
        output_claims_by_instance,
    } = proof;

    if layer_masks_by_instance.len() != output_claims_by_instance.len() {
        return Err(GkrError::MalformedProof);
    }

    let n_instances = layer_masks_by_instance.len();
    let instance_n_layers = |instance: usize| layer_masks_by_instance[instance].len();
    let n_layers = (0..n_instances).map(instance_n_layers).max().unwrap();

    if n_layers != sumcheck_proofs.len() {
        return Err(GkrError::MalformedProof);
    }

    if gate_by_instance.len() != n_instances {
        return Err(GkrError::NumInstancesMismatch {
            given: gate_by_instance.len(),
            proof: n_instances,
        });
    }

    let mut ood_point = vec![];
    let mut claims_to_verify_by_instance = vec![None; n_instances];

    for (layer, sumcheck_proof) in sumcheck_proofs.iter().enumerate() {
        let n_remaining_layers = n_layers - layer;

        // Check for output layers.
        for instance in 0..n_instances {
            if instance_n_layers(instance) == n_remaining_layers {
                let output_claims = output_claims_by_instance[instance].clone();
                claims_to_verify_by_instance[instance] = Some(output_claims);
            }
        }

        // Seed the channel with layer claims.
        for claims_to_verify in claims_to_verify_by_instance.iter().flatten() {
            channel.mix_felts(claims_to_verify);
        }

        let sumcheck_alpha = channel.draw_felt();
        let instance_lambda = channel.draw_felt();

        let mut sumcheck_claims = Vec::new();
        let mut sumcheck_instances = Vec::new();

        // Prepare the sumcheck claim.
        for (instance, claims_to_verify) in claims_to_verify_by_instance.iter().enumerate() {
            if let Some(claims_to_verify) = claims_to_verify {
                let n_unused_variables = n_layers - instance_n_layers(instance);
                let doubling_factor = BaseField::from(1 << n_unused_variables);
                let claim =
                    random_linear_combination(claims_to_verify, instance_lambda) * doubling_factor;
                sumcheck_claims.push(claim);
                sumcheck_instances.push(instance);
            }
        }

        let sumcheck_claim = random_linear_combination(&sumcheck_claims, sumcheck_alpha);
        let (sumcheck_ood_point, sumcheck_eval) =
            sumcheck::partially_verify(sumcheck_claim, sumcheck_proof, channel)
                .map_err(|source| GkrError::InvalidSumcheck { layer, source })?;

        let mut layer_evals = Vec::new();

        // Evaluate the circuit locally at sumcheck OOD point.
        for &instance in &sumcheck_instances {
            let n_unused = n_layers - instance_n_layers(instance);
            let mask = &layer_masks_by_instance[instance][layer - n_unused];
            let gate = &gate_by_instance[instance];
            let gate_output = gate.eval(mask).map_err(|InvalidNumMaskColumnsError| {
                let instance_layer = instance_n_layers(layer) - n_remaining_layers;
                GkrError::InvalidMask {
                    instance,
                    instance_layer,
                }
            })?;
            // TODO: Consider simplifying the code by just using the same eq eval for all instances
            // regardless of size.
            let eq_eval = eq(&ood_point[n_unused..], &sumcheck_ood_point[n_unused..]);
            layer_evals.push(eq_eval * random_linear_combination(&gate_output, instance_lambda));
        }

        let layer_eval = random_linear_combination(&layer_evals, sumcheck_alpha);

        if sumcheck_eval != layer_eval {
            return Err(GkrError::CircuitCheckFailure {
                claim: sumcheck_eval,
                output: layer_eval,
                layer,
            });
        }

        // Seed the channel with the layer masks.
        for &instance in &sumcheck_instances {
            let n_unused = n_layers - instance_n_layers(instance);
            let mask = &layer_masks_by_instance[instance][layer - n_unused];
            channel.mix_felts(mask.columns().flatten());
        }

        // Set the OOD evaluation point for layer above.
        let challenge = channel.draw_felt();
        ood_point = sumcheck_ood_point;
        ood_point.push(challenge);

        // Set the claims to verify in the layer above.
        for instance in sumcheck_instances {
            let n_unused = n_layers - instance_n_layers(instance);
            let mask = &layer_masks_by_instance[instance][layer - n_unused];
            claims_to_verify_by_instance[instance] = Some(mask.reduce_at_point(challenge));
        }
    }

    let claims_to_verify_by_instance = claims_to_verify_by_instance
        .into_iter()
        .map(Option::unwrap)
        .collect();

    Ok(GkrArtifact {
        ood_point,
        gate_by_instance,
        claims_to_verify_by_instance,
        n_variables_by_instance: (0..n_instances).map(instance_n_layers).collect(),
    })
}

/// Batch GKR proof.
pub struct GkrBatchProof {
    /// Sum-check proof for each layer.
    pub sumcheck_proofs: Vec<SumcheckProof>,
    /// Mask for each layer for each instance.
    pub layer_masks_by_instance: Vec<Vec<GkrMask>>,
    /// Column circuit outputs for each instance.
    pub output_claims_by_instance: Vec<Vec<SecureField>>,
}

/// Values of interest obtained from the execution of the GKR protocol.
pub struct GkrArtifact {
    /// Out-of-domain (OOD) point for evaluating columns in the input layer.
    pub ood_point: Vec<SecureField>,
    /// The gate of each instance.
    pub gate_by_instance: Vec<Gate>,
    /// The claimed evaluation at `ood_point` for each column in the input layer of each instance.
    pub claims_to_verify_by_instance: Vec<Vec<SecureField>>,
    /// The number of variables that interpolate the input layer of each instance.
    pub n_variables_by_instance: Vec<usize>,
}

impl GkrArtifact {
    pub fn ood_point(&self, instance_n_variables: usize) -> &[SecureField] {
        &self.ood_point[self.ood_point.len() - instance_n_variables..]
    }
}

pub struct LookupArtifactInstanceIter<'proof, 'artifact> {
    instance: usize,
    gkr_proof: &'proof GkrBatchProof,
    gkr_artifact: &'artifact GkrArtifact,
}

impl<'proof, 'artifact> LookupArtifactInstanceIter<'proof, 'artifact> {
    pub fn new(gkr_proof: &'proof GkrBatchProof, gkr_artifact: &'artifact GkrArtifact) -> Self {
        Self {
            instance: 0,
            gkr_proof,
            gkr_artifact,
        }
    }
}

impl<'proof, 'artifact> Iterator for LookupArtifactInstanceIter<'proof, 'artifact> {
    type Item = LookupArtifactInstance;

    fn next(&mut self) -> Option<LookupArtifactInstance> {
        if self.instance >= self.gkr_proof.output_claims_by_instance.len() {
            return None;
        }

        let instance = self.instance;
        let input_n_variables = self.gkr_artifact.n_variables_by_instance[instance];
        let eval_point = self.gkr_artifact.ood_point(input_n_variables).to_vec();
        let output_claim = &*self.gkr_proof.output_claims_by_instance[instance];
        let input_claims = &*self.gkr_artifact.claims_to_verify_by_instance[instance];
        let gate = self.gkr_artifact.gate_by_instance[instance];

        let res = Some(match gate {
            Gate::LogUp => {
                let [numerator, denominator] = output_claim.try_into().unwrap();
                let claimed_sum = Fraction::new(numerator, denominator);
                let [input_numerators_claim, input_denominators_claim] =
                    input_claims.try_into().unwrap();

                LookupArtifactInstance::LogUp(LogUpArtifactInstance {
                    eval_point,
                    input_n_variables,
                    input_numerators_claim,
                    input_denominators_claim,
                    claimed_sum,
                })
            }
            Gate::GrandProduct => {
                let [claimed_product] = output_claim.try_into().unwrap();
                let [input_claim] = input_claims.try_into().unwrap();

                LookupArtifactInstance::GrandProduct(GrandProductArtifactInstance {
                    eval_point,
                    input_n_variables,
                    input_claim,
                    claimed_product,
                })
            }
        });

        self.instance += 1;
        res
    }
}

// TODO: Consider making the GKR artifact just a Vec<LookupArtifactInstance>.
pub enum LookupArtifactInstance {
    GrandProduct(GrandProductArtifactInstance),
    LogUp(LogUpArtifactInstance),
}

pub struct GrandProductArtifactInstance {
    /// GKR input layer eval point.
    pub eval_point: Vec<SecureField>,
    /// Number of variables the MLE in the GKR input layer had.
    pub input_n_variables: usize,
    /// Claimed input MLE evaluation at `eval_point`.
    pub input_claim: SecureField,
    /// Output claim from the circuit.
    pub claimed_product: SecureField,
}

pub struct LogUpArtifactInstance {
    /// GKR input layer eval point.
    pub eval_point: Vec<SecureField>,
    /// Number of variables the MLEs in the GKR input layer had.
    pub input_n_variables: usize,
    /// Claimed input numerators MLE evaluation at `eval_point`.
    pub input_numerators_claim: SecureField,
    /// Claimed input denominators MLE evaluation at `eval_point`.
    pub input_denominators_claim: SecureField,
    /// Output claim from the circuit.
    pub claimed_sum: Fraction<SecureField, SecureField>,
}

/// Defines how a circuit operates locally on two input rows to produce a single output row.
/// This local 2-to-1 constraint is what gives the whole circuit its "binary tree" structure.
///
/// Binary tree structured circuits have a highly regular wiring pattern that fit the structure of
/// the circuits defined in [Thaler13] which allow for efficient linear time (linear in size of the
/// circuit) GKR prover implementations.
///
/// [Thaler13]: https://eprint.iacr.org/2013/351.pdf
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Gate {
    LogUp,
    GrandProduct,
}

impl Gate {
    /// Returns the output after applying the gate to the mask.
    fn eval(&self, mask: &GkrMask) -> Result<Vec<SecureField>, InvalidNumMaskColumnsError> {
        Ok(match self {
            Self::LogUp => {
                if mask.columns().len() != 2 {
                    return Err(InvalidNumMaskColumnsError);
                }

                let [numerator_a, numerator_b] = mask.columns()[0];
                let [denominator_a, denominator_b] = mask.columns()[1];

                let a = Fraction::new(numerator_a, denominator_a);
                let b = Fraction::new(numerator_b, denominator_b);
                let res = a + b;

                vec![res.numerator, res.denominator]
            }
            Self::GrandProduct => {
                if mask.columns().len() != 1 {
                    return Err(InvalidNumMaskColumnsError);
                }

                let [a, b] = mask.columns()[0];
                vec![a * b]
            }
        })
    }
}

/// Mask has an invalid number of columns
#[derive(Debug)]
struct InvalidNumMaskColumnsError;

/// Stores two evaluations of each column in a GKR layer.
#[derive(Debug, Clone)]
pub struct GkrMask {
    columns: Vec<[SecureField; 2]>,
}

impl GkrMask {
    pub fn new(columns: Vec<[SecureField; 2]>) -> Self {
        Self { columns }
    }

    pub fn to_rows(&self) -> [Vec<SecureField>; 2] {
        self.columns.iter().map(|[a, b]| (a, b)).unzip().into()
    }

    pub fn columns(&self) -> &[[SecureField; 2]] {
        &self.columns
    }

    /// Returns all `p_i(x)` where `p_i` interpolates column `i` of the mask on `{0, 1}`.
    pub fn reduce_at_point(&self, x: SecureField) -> Vec<SecureField> {
        self.columns
            .iter()
            .map(|&[v0, v1]| fold_mle_evals(x, v0, v1))
            .collect()
    }
}

/// Error encountered during GKR protocol verification.
#[derive(Error, Debug)]
pub enum GkrError {
    /// The proof is malformed.
    #[error("proof data is invalid")]
    MalformedProof,
    /// Mask has an invalid number of columns.
    #[error("mask in layer {instance_layer} of instance {instance} is invalid")]
    InvalidMask {
        instance: usize,
        /// Layer of the instance (but not necessarily the batch).
        instance_layer: LayerIndex,
    },
    /// There is a mismatch between the number of instances in the proof and the number of
    /// instances passed for verification.
    #[error("provided an invalid number of instances (given {given}, proof expects {proof})")]
    NumInstancesMismatch { given: usize, proof: usize },
    /// There was an error with one of the sumcheck proofs.
    #[error("sum-check invalid in layer {layer}: {source}")]
    InvalidSumcheck {
        layer: LayerIndex,
        source: SumcheckError,
    },
    /// The circuit polynomial the verifier evaluated doesn't match claim from sumcheck.
    #[error("circuit check failed in layer {layer} (calculated {output}, claim {claim})")]
    CircuitCheckFailure {
        claim: SecureField,
        output: SecureField,
        layer: LayerIndex,
    },
}

/// GKR layer index where 0 corresponds to the output layer.
pub type LayerIndex = usize;

#[cfg(test)]
mod tests {
    use super::{partially_verify_batch, Gate, GkrArtifact, GkrError};
    use crate::core::backend::CpuBackend;
    use crate::core::channel::Channel;
    use crate::core::fields::qm31::SecureField;
    use crate::core::lookups::gkr_prover::{prove_batch, Layer};
    use crate::core::lookups::mle::Mle;
    use crate::core::test_utils::test_channel;

    #[test]
    fn prove_batch_works() -> Result<(), GkrError> {
        const LOG_N: usize = 5;
        let mut channel = test_channel();
        let col0 = Mle::<CpuBackend, SecureField>::new(channel.draw_felts(1 << LOG_N));
        let col1 = Mle::<CpuBackend, SecureField>::new(channel.draw_felts(1 << LOG_N));
        let product0 = col0.iter().product::<SecureField>();
        let product1 = col1.iter().product::<SecureField>();
        let input_layers = vec![
            Layer::GrandProduct(col0.clone()),
            Layer::GrandProduct(col1.clone()),
        ];
        let (proof, _) = prove_batch(&mut test_channel(), input_layers);

        let GkrArtifact {
            ood_point,
            gate_by_instance,
            claims_to_verify_by_instance,
            n_variables_by_instance,
        } = partially_verify_batch(vec![Gate::GrandProduct; 2], &proof, &mut test_channel())?;

        assert_eq!(n_variables_by_instance, [LOG_N, LOG_N]);
        assert_eq!(gate_by_instance, [Gate::GrandProduct, Gate::GrandProduct]);
        assert_eq!(proof.output_claims_by_instance.len(), 2);
        assert_eq!(claims_to_verify_by_instance.len(), 2);
        assert_eq!(proof.output_claims_by_instance[0], &[product0]);
        assert_eq!(proof.output_claims_by_instance[1], &[product1]);
        let claim0 = &claims_to_verify_by_instance[0];
        let claim1 = &claims_to_verify_by_instance[1];
        assert_eq!(claim0, &[col0.eval_at_point(&ood_point)]);
        assert_eq!(claim1, &[col1.eval_at_point(&ood_point)]);
        Ok(())
    }

    #[test]
    fn prove_batch_with_different_sizes_works() -> Result<(), GkrError> {
        const LOG_N0: usize = 5;
        const LOG_N1: usize = 7;
        let mut channel = test_channel();
        let col0 = Mle::<CpuBackend, SecureField>::new(channel.draw_felts(1 << LOG_N0));
        let col1 = Mle::<CpuBackend, SecureField>::new(channel.draw_felts(1 << LOG_N1));
        let product0 = col0.iter().product::<SecureField>();
        let product1 = col1.iter().product::<SecureField>();
        let input_layers = vec![
            Layer::GrandProduct(col0.clone()),
            Layer::GrandProduct(col1.clone()),
        ];
        let (proof, _) = prove_batch(&mut test_channel(), input_layers);

        let GkrArtifact {
            ood_point,
            gate_by_instance,
            claims_to_verify_by_instance,
            n_variables_by_instance,
        } = partially_verify_batch(vec![Gate::GrandProduct; 2], &proof, &mut test_channel())?;

        assert_eq!(n_variables_by_instance, [LOG_N0, LOG_N1]);
        assert_eq!(gate_by_instance, [Gate::GrandProduct, Gate::GrandProduct]);
        assert_eq!(proof.output_claims_by_instance.len(), 2);
        assert_eq!(claims_to_verify_by_instance.len(), 2);
        assert_eq!(proof.output_claims_by_instance[0], &[product0]);
        assert_eq!(proof.output_claims_by_instance[1], &[product1]);
        let claim0 = &claims_to_verify_by_instance[0];
        let claim1 = &claims_to_verify_by_instance[1];
        let n_vars = ood_point.len();
        assert_eq!(claim0, &[col0.eval_at_point(&ood_point[n_vars - LOG_N0..])]);
        assert_eq!(claim1, &[col1.eval_at_point(&ood_point[n_vars - LOG_N1..])]);
        Ok(())
    }
}
