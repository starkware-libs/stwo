//! GKR batch verifier for Grand Product and LogUp lookup arguments.
use thiserror::Error;

use super::sumcheck::{SumcheckError, SumcheckProof};
use super::utils::{eq, fold_mle_evals, random_linear_combination};
use crate::core::channel::Channel;
use crate::core::fields::m31::BaseField;
use crate::core::fields::qm31::SecureField;
use crate::core::lookups::sumcheck;

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
    /// The claimed evaluation at `ood_point` for each column in the input layer of each instance.
    pub claims_to_verify_by_instance: Vec<Vec<SecureField>>,
    /// The number of variables that interpolate the input layer of each instance.
    pub n_variables_by_instance: Vec<usize>,
}

/// Defines how a circuit operates locally on two input rows to produce a single output row.
/// This local 2-to-1 constraint is what gives the whole circuit its "binary tree" structure.
///
/// Binary tree structured circuits have a highly regular wiring pattern that fit the structure of
/// the circuits defined in [Thaler13] which allow for efficient linear time (linear in size of the
/// circuit) GKR prover implementations.
///
/// [Thaler13]: https://eprint.iacr.org/2013/351.pdf
pub enum Gate {
    _LogUp,
    _GrandProduct,
}

impl Gate {
    /// Returns the output after applying the gate to the mask.
    fn eval(&self, _mask: &GkrMask) -> Result<Vec<SecureField>, InvalidNumMaskColumnsError> {
        todo!()
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
