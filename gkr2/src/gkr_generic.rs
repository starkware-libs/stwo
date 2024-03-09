#![allow(clippy::useless_transmute, clippy::useless_conversion)]

use std::iter::{successors, zip};
use std::time::Duration;

use itertools::Itertools;
use num_traits::{One, Zero};
use prover_research::core::channel::Channel;
use prover_research::core::fields::qm31::{SecureField, SecureField as FastSecureField};
use prover_research::core::fields::Field;
use thiserror::Error;

use crate::mle::{Mle, MleTrace};
use crate::sumcheck::{self, SumcheckError, SumcheckOracle, SumcheckProof};
use crate::utils::{horner_eval, Fraction};

#[derive(Debug, Clone)]
#[allow(dead_code)]
pub struct MleLayer {
    p: Mle<SecureField>,
    q: Mle<SecureField>,
    num_variables: u32,
}

impl MleLayer {
    pub fn new<F: Field + Into<SecureField>>(f: &Vec<Fraction<F>>) -> Self {
        assert!(f.len().is_power_of_two());
        let (p, q) = f
            .iter()
            .map(
                |&Fraction {
                     numerator,
                     denominator,
                 }| (numerator.into(), denominator.into()),
            )
            .unzip();

        Self {
            // n_vars: f.len().ilog2() as usize,
            p: Mle::new(p),
            q: Mle::new(q),
            num_variables: f.len().ilog2(),
        }
    }
}

/// Evaluates the lagrange kernel of the boolean hypercube.
///
/// The lagrange kernel of the boolean hyperbube is a multilinear extension of the function that
/// when given `x, y` in `{0, 1}^n` evaluates to 1 if `x = z`, and evaluates to 0 otherwise.
pub fn eq(x_assignments: &[SecureField], y_assignments: &[SecureField]) -> SecureField {
    assert_eq!(x_assignments.len(), y_assignments.len());
    zip(x_assignments, y_assignments)
        .map(|(&xi, &wi)| xi * wi + (SecureField::one() - xi) * (SecureField::one() - wi))
        .product::<SecureField>()
}

/// Defines a circuit with a binary tree structure.
///
/// These circuits define how they operate locally on pairs of neighboring input rows to produce a
/// single output row. This local 2-to-1 constraint is what gives the whole circuit its "binary
/// tree" structure. Circuit examples: [`LogupCircuit`], [`GrandProductCircuit`].
///
/// A binary tree circuit has a highly "regular" wiring pattern. It fits the structure of the
/// circuits defined in [Thaler13] which allows for an efficient linear time (linear in size of the
/// circuit) implementation of a GKR prover.
///
/// [Thaler13]: https://eprint.iacr.org/2013/351.pdf
pub trait BinaryTreeCircuit {
    /// Returns the output row after applying the circuit to the provided neighboring input rows.
    fn eval(even_row: &[SecureField], odd_row: &[SecureField]) -> Vec<SecureField>;
}

/// Logup circuit from <https://eprint.iacr.org/2023/1284.pdf> (section 3.1)
struct LogupCircuit;

impl BinaryTreeCircuit for LogupCircuit {
    fn eval(even_row: &[SecureField], odd_row: &[SecureField]) -> Vec<SecureField> {
        assert_eq!(even_row.len(), 2);
        assert_eq!(odd_row.len(), 2);

        let a = Fraction::new(even_row[0], even_row[1]);
        let b = Fraction::new(odd_row[0], odd_row[1]);
        let c = a + b;

        vec![c.numerator, c.denominator]
    }
}

/// Circuit multiplies all values together.
struct GrandProductCircuit;

impl BinaryTreeCircuit for GrandProductCircuit {
    fn eval(even_row: &[SecureField], odd_row: &[SecureField]) -> Vec<SecureField> {
        assert_eq!(even_row.len(), 1);
        assert_eq!(odd_row.len(), 1);

        let a = even_row[0];
        let b = odd_row[0];
        let c = a * b;

        vec![c]
    }
}

/// Partially verifies a GKR proof.
///
/// Returns the variable assignment and claimed evaluation in the top layer. Neither (1) the top
/// layer evaluation claim or (2) checks on the output layer are validated by this function - hence
/// partial verification.
///
/// Output of the form `(variable_assignment, claimed_evals)`.
pub fn partially_verify<C: BinaryTreeCircuit>(
    proof: &GkrProof,
    channel: &mut impl Channel,
) -> Result<(Vec<SecureField>, Vec<SecureField>), GkrError> {
    let zero = SecureField::zero();
    let one = SecureField::one();

    let GkrProof {
        output_layer,
        layer_proofs,
    } = proof;

    output_layer.iter().for_each(|c| channel.mix_felts(c));

    let mut layer_assignment = channel.draw_felts(output_layer.num_variables());
    let mut layer_claim = output_layer.eval(&layer_assignment);

    for (layer, layer_proof) in (1..).zip(layer_proofs) {
        let GkrLayerProof {
            sumcheck_proof,
            input_encoding,
        } = layer_proof;

        let lambda = channel.draw_felt();
        let sumcheck_claim = horner_eval(&layer_claim, lambda);
        let (sumcheck_assignment, sumcheck_eval) =
            sumcheck::partially_verify(sumcheck_claim, sumcheck_proof, channel)
                .map_err(|source| GkrError::InvalidSumcheck { layer, source })?;

        assert_eq!(input_encoding.num_variables(), 1);
        let circuit_output = C::eval(&input_encoding.eval(&[zero]), &input_encoding.eval(&[one]));
        let folded_output = horner_eval(&circuit_output, lambda);
        let layer_eval = eq(&layer_assignment, &sumcheck_assignment) * folded_output;

        if sumcheck_eval != layer_eval {
            return Err(GkrError::CircuitCheckFailure {
                claim: sumcheck_eval,
                output: layer_eval,
                layer,
            });
        }

        input_encoding.iter().for_each(|c| channel.mix_felts(c));

        let r_star = channel.draw_felt();
        layer_assignment = sumcheck_assignment;
        layer_assignment.push(r_star);

        layer_claim = input_encoding.eval(&[r_star])
    }

    Ok((layer_assignment, layer_claim))
}

/// Error encountered during GKR protocol verification.
///
/// Layer 1 corresponds to the output layer.
#[derive(Error, Debug)]
pub enum GkrError {
    #[error("sum-check invalid in layer {layer}: {source}")]
    InvalidSumcheck { layer: usize, source: SumcheckError },
    #[error("circuit check failed in layer {layer} (calculated {output}, claim {claim})")]
    CircuitCheckFailure {
        claim: SecureField,
        output: SecureField,
        layer: usize,
    },
}

#[derive(Debug, Clone)]
pub struct GkrProof {
    layer_proofs: Vec<GkrLayerProof>,
    output_layer: MleTrace<SecureField>,
}

#[derive(Debug, Clone)]
struct GkrLayerProof {
    sumcheck_proof: SumcheckProof,
    input_encoding: MleTrace<SecureField>,
}

struct Oracle {
    /// p_{i + 1}
    // TODO: Consider `Cow<Vec<F>>`
    p: Mle<FastSecureField>,
    /// q_{i + 1}
    q: Mle<FastSecureField>,
    // TODO: docs.
    c: Vec<FastSecureField>,
    num_variables: usize,
    z: Vec<FastSecureField>,
    lambda: FastSecureField,
    claim: SecureField,
}

impl Oracle {
    #[allow(dead_code)]
    fn new(
        z: &[SecureField],
        p: Mle<SecureField>,
        q: Mle<SecureField>,
        lambda: SecureField,
        claim: SecureField,
    ) -> Self {
        let num_variables = z.len();
        assert_eq!(p.len(), 2 << num_variables);
        assert_eq!(q.len(), 2 << num_variables);
        // let now = Instant::now();
        let mut c = c0(unsafe { std::mem::transmute(z) });
        // TODO: Only require LHS evaluations (i.e. where the first variable assignment equals `0`).
        c.truncate(c.len() / 2);
        // println!("c gen time: {:?}", now.elapsed());
        Self {
            p: unsafe { std::mem::transmute(p) },
            q: unsafe { std::mem::transmute(q) },
            num_variables,
            c,
            z: unsafe { std::mem::transmute(z.to_vec()) },
            lambda: lambda.into(),
            claim,
        }
    }
}

static mut UNIVARIATE_SUM_DUR: Duration = Duration::ZERO;
static mut COLLAPSE_PQ_DURATION: Duration = Duration::ZERO;
static mut COLLAPSE_C_DURATION: Duration = Duration::ZERO;

pub static mut SUMCHECK_ADDS: usize = 0;
pub static mut SUMCHECK_MULTS: usize = 0;

/// Computes all TODO in `O(2^|z|)`
///
/// Source: <https://eprint.iacr.org/2013/351.pdf> (Section 5.4.1)
#[allow(dead_code)]
fn c0(z: &[FastSecureField]) -> Vec<FastSecureField> {
    match z {
        &[z1] => vec![FastSecureField::one() - z1, z1],
        &[zj, ref z @ ..] => {
            let c = c0(z);
            let zj_bar = FastSecureField::one() - zj;
            // TODO: this can be reduced to single mult and addition
            unsafe { SUMCHECK_ADDS += c.len() };
            unsafe { SUMCHECK_MULTS += c.len() };
            let lhs = c.iter().map(|&v| zj_bar * v);
            let rhs = c.iter().map(|&v| zj * v);
            Iterator::chain(lhs, rhs).collect()
        }
        [] => panic!(),
    }
}
// /// Evaluations of the polynomial `eq(x_1, ..., x_n) = (z_1 * x_1 + (1 - z_1) * (1 - x_1)) * ...
// * /// (z_n * x_n + (1 - z_n) * (1 - x_n))` over the boolean hypercube `{0, 1}^n`.
// struct EqEvaluation {
//     evals: Vec<SecureField>,
//     r: Vec<SecureField>,
// }

// impl EqEvaluation {
//     fn new(r: &[SecureField]) -> Self {
//         // let num_
//     }
// }

/// Source: <https://eprint.iacr.org/2013/351.pdf> (Section 5.4.1)
fn collapse_c(
    mut c: Vec<FastSecureField>,
    z: FastSecureField,
    r: FastSecureField,
) -> Vec<FastSecureField> {
    // TODO: `z` can be one! Just divide out z (instead of `(1 - z)`) and take rhs values of `c`.
    // Don't want to implement this just noting here.
    assert!(!z.is_one());

    let z_bar = FastSecureField::one() - z;
    let r_bar = FastSecureField::one() - r;
    // TODO: Shift not right word.
    let shift = z_bar.inverse() * (r * z + r_bar * z_bar);

    unsafe { SUMCHECK_MULTS += c.len() / 2 };

    c.truncate(c.len() / 2);
    c.iter_mut().for_each(|v| *v *= shift);

    c
}

pub trait GkrLayer: Sized {
    type SumcheckOracle: GkrSumcheckOracle;

    /// Produces the next GKR layer from the current one.
    fn next(&self) -> Option<Self>;

    /// Create an oracle for sumcheck
    fn into_sumcheck_oracle(
        self,
        lambda: SecureField,
        layer_assignment: &[SecureField],
    ) -> Self::SumcheckOracle;

    /// Returns this layer as a [`MleTrace`].
    ///
    /// Note that this function is only called on the GKR output layer. The output layer needs to be
    /// stored in the proof. To prevent making proofs generic, all output layers are stored in the
    /// proof as a [`MleTrace`].
    fn into_trace(self) -> MleTrace<SecureField>;
}

pub trait GkrSumcheckOracle: SumcheckOracle {
    /// Returns the input multi-linear extensions that define `g`.
    ///
    /// Let `g(x_1, ..., x_n)` represent the multivariate polynomial that the sum-check protocol is
    /// applied to in layer `l` of the GKR protocol. What's special about this `g` is that it is
    /// defined by values in layer `l+1`. This function returns the layer `l+1` values for `g`.
    // TODO: Document better.
    fn into_inputs(self) -> MleTrace<SecureField> {
        todo!()
    }
}

// <https://people.cs.georgetown.edu/jthaler/ProofsArgsAndZK.pdf> (page 65)
// TODO: add a mix felts over multiple slices.
// `channel.mix_felt_chunks(output_layer.iter().map(|mle| &mle));`
pub fn prove(channel: &mut impl Channel, top_layer: impl GkrLayer) -> GkrProof {
    let layers = successors(Some(top_layer), |layer| layer.next()).collect_vec();
    let mut layers = layers.into_iter().rev();

    let output_layer = layers.next().unwrap().into_trace();
    output_layer.iter().for_each(|c| channel.mix_felts(c));

    let mut layer_assignment = channel.draw_felts(output_layer.num_variables());
    let mut layer_evals = output_layer.eval(&layer_assignment);

    let layer_proofs = layers
        .map(|layer| {
            let lambda = channel.draw_felt();
            let sumcheck_oracle = layer.into_sumcheck_oracle(lambda, &layer_assignment);
            let sumcheck_claim = horner_eval(&layer_evals, lambda);
            let (sumcheck_proof, sumcheck_assignment, oracle) =
                sumcheck::prove(sumcheck_claim, sumcheck_oracle, channel);

            let input_encoding = oracle.into_inputs();
            input_encoding.iter().for_each(|c| channel.mix_felts(c));

            assert_eq!(input_encoding.num_variables(), 1);
            let r_star = channel.draw_felt();
            layer_assignment = sumcheck_assignment;
            layer_assignment.push(r_star);

            layer_evals = input_encoding.eval(&[r_star]);

            GkrLayerProof {
                sumcheck_proof,
                input_encoding,
            }
        })
        .collect();

    GkrProof {
        layer_proofs,
        output_layer,
    }
}

#[cfg(test)]
mod tests {
    use std::iter::zip;
    use std::time::Instant;

    use prover_research::commitment_scheme::blake2_hash::Blake2sHasher;
    use prover_research::commitment_scheme::hasher::Hasher;
    use prover_research::core::channel::{Blake2sChannel, Channel};
    use prover_research::core::fields::qm31::SecureField;

    use super::{
        partially_verify, prove, MleLayer, COLLAPSE_C_DURATION, COLLAPSE_PQ_DURATION,
        UNIVARIATE_SUM_DUR,
    };
    use crate::gkr_logup::LogupTrace;
    use crate::utils::Fraction;

    #[test]
    fn mle_bench() {
        const N: usize = 1 << 20;

        let mut channel = test_channel();
        let mut random_fractions = zip(
            channel.draw_felts(N).into_iter().map(SecureField::from),
            channel.draw_felts(N).into_iter().map(SecureField::from),
        )
        .map(|(numerator, denominator)| Fraction::new(numerator, denominator))
        .collect::<Vec<Fraction<SecureField>>>();

        // Make the fractions sum to zero.
        let now = Instant::now();
        let sum = random_fractions.iter().sum::<Fraction<SecureField>>();
        println!("layer sum time: {:?}", now.elapsed());
        random_fractions[0] = random_fractions[0] - sum;

        let now = Instant::now();
        let mut layers = Vec::new();
        while random_fractions.len() > 1 {
            layers.push(MleLayer::new(&random_fractions));
            let mut chunks = random_fractions.array_chunks();

            let mut res = (&mut chunks)
                .flat_map(|&[a, b, c, d, e, f, g, h]| [a + b, c + d, e + f, g + h])
                .collect::<Vec<_>>();

            chunks
                .remainder()
                .array_chunks()
                .for_each(|&[a, b]| res.push(a + b));

            random_fractions = res;
        }
        layers.reverse();

        println!("layer gen time: {:?}", now.elapsed());

        // println!("yo: {}"
        let MleLayer { p, q, .. } = MleLayer::new(&random_fractions);
        let top_layer = LogupTrace::new(p, q);

        let now = Instant::now();
        let proof = prove(&mut test_channel(), top_layer);
        println!("proof gen time: {:?}", now.elapsed());

        println!("total collapsing c time: {:?}", unsafe {
            COLLAPSE_C_DURATION
        });
        println!("collapse pq duration: {:?}", unsafe {
            COLLAPSE_PQ_DURATION
        });
        println!("univariate eval duration: {:?}", unsafe {
            UNIVARIATE_SUM_DUR
        });
        // println!("sumcheck duration: {:?}", unsafe { SUMCHECK_DURATION });

        struct LogupCircuit;

        impl GkrCircuit for LogupCircuit {
            fn eval(row1: &[SecureField], row2: &[SecureField]) -> Vec<SecureField> {
                assert_eq!(row1.len(), 2);
                assert_eq!(row2.len(), 2);
                let a = Fraction::new(row1[0], row1[1]);
                let b = Fraction::new(row2[0], row2[1]);
                let res = a + b;
                vec![res.numerator, res.denominator]
            }
        }

        // let (assignment, p3_claim, q3_claim) =
        //     partially_verify(&proof, &mut test_channel()).unwrap();
        let now = Instant::now();
        let res = partially_verify::<LogupCircuit>(&proof, &mut test_channel());
        println!("verify time: {:?}", now.elapsed());
        assert!(res.is_ok());

        // // List of fractions that sum to zero.
        // let fractions = [
        //     a + a + a, //
        //     a + b + b + c,
        //     -a,
        //     -a,
        //     -a,
        //     c - b - b,
        //     -a - c,
        //     -c,
        // ];
    }

    fn test_channel() -> Blake2sChannel {
        let seed = Blake2sHasher::hash(&[]);
        Blake2sChannel::new(seed)
    }
}
