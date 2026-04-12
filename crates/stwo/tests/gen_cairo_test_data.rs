use num_traits::One;
use stwo::core::channel::Blake2sChannel;
use stwo::core::circle::Coset;
use stwo::core::fields::m31::BaseField;
use stwo::core::fields::qm31::SecureField;
use stwo::core::fri::{FriConfig, FriLayerProof, FriProof};
use stwo::core::poly::circle::CircleDomain;
use stwo::core::poly::line::LinePoly;
use stwo::core::queries::Queries;
use stwo::core::vcs::blake2_hash::Blake2sHash;
use stwo::core::vcs_lifted::blake2_merkle::Blake2sMerkleChannel;
use stwo::prover::backend::cpu::CpuCirclePoly;
use stwo::prover::backend::CpuBackend;
use stwo::prover::poly::circle::{PolyOps, SecureEvaluation};
use stwo::prover::poly::BitReversedOrder;

type H = <Blake2sMerkleChannel as stwo::core::channel::MerkleChannel>::H;
type FriProver<'a> = stwo::prover::fri::FriProver<'a, CpuBackend, Blake2sMerkleChannel>;

fn polynomial_evaluation(
    log_degree: u32,
    log_blowup_factor: u32,
) -> SecureEvaluation<CpuBackend, BitReversedOrder> {
    let poly = CpuCirclePoly::new(vec![BaseField::one(); 1 << log_degree]);
    let coset = Coset::half_odds(log_degree + log_blowup_factor - 1);
    let domain = CircleDomain::new(coset);
    let values = poly.evaluate(domain);
    SecureEvaluation::new(domain, values.into_iter().map(SecureField::from).collect())
}

fn query_polynomial(
    polynomial: &SecureEvaluation<CpuBackend, BitReversedOrder>,
    queries: &Queries,
) -> Vec<SecureField> {
    let queries = queries.fold(queries.log_domain_size - polynomial.domain.log_size());
    queries.positions.iter().map(|p| polynomial.at(*p)).collect()
}

fn serialize_hash(hash: Blake2sHash, output: &mut Vec<u32>) {
    for chunk in hash.0.chunks_exact(4) {
        output.push(u32::from_le_bytes(chunk.try_into().unwrap()));
    }
}

fn serialize_qm31(sf: SecureField, output: &mut Vec<u32>) {
    for c in sf.to_m31_array() {
        output.push(c.0);
    }
}

fn serialize_fri_layer_proof(proof: &FriLayerProof<H>, output: &mut Vec<u32>) {
    output.push(proof.fri_witness.len() as u32);
    for v in &proof.fri_witness {
        serialize_qm31(*v, output);
    }
    output.push(proof.decommitment.hash_witness.len() as u32);
    for h in &proof.decommitment.hash_witness {
        serialize_hash(*h, output);
    }
    serialize_hash(proof.commitment, output);
}

fn serialize_line_poly(poly: &LinePoly, output: &mut Vec<u32>) {
    output.push(poly.len() as u32);
    for c in poly.iter() {
        serialize_qm31(*c, output);
    }
    output.push(poly.len().ilog2());
}

fn serialize_fri_proof(proof: &FriProof<H>, output: &mut Vec<u32>) {
    serialize_fri_layer_proof(&proof.first_layer, output);
    output.push(proof.inner_layers.len() as u32);
    for layer in &proof.inner_layers {
        serialize_fri_layer_proof(layer, output);
    }
    serialize_line_poly(&proof.last_layer_poly, output);
}

fn generate(log_degree: u32, fold_step: u32, query_positions: Vec<usize>) {
    const LOG_BLOWUP_FACTOR: u32 = 2;

    let column = polynomial_evaluation(log_degree, LOG_BLOWUP_FACTOR);
    let twiddles = CpuBackend::precompute_twiddles(column.domain.half_coset);
    let queries = Queries::new(&query_positions, column.domain.log_size());
    let config = FriConfig::new(0, LOG_BLOWUP_FACTOR, queries.len(), fold_step);
    let decommitment_values = query_polynomial(&column, &queries);

    let prover = FriProver::commit(&mut Blake2sChannel::default(), config, &column, &twiddles);
    let proof = prover.decommit_on_queries(&queries).proof;

    let column_log_size = log_degree + LOG_BLOWUP_FACTOR;

    // Serialize.
    let mut serialized = Vec::new();
    serialize_fri_proof(&proof, &mut serialized);

    println!("// fold_step={fold_step}, log_degree={log_degree}, log_blowup={LOG_BLOWUP_FACTOR}");
    println!("// column_log_bound={log_degree}, column_log_size={column_log_size}");
    println!("// queries={query_positions:?}");
    for (i, v) in decommitment_values.iter().enumerate() {
        let a = v.to_m31_array();
        println!(
            "// query_eval[{i}] = qm31_const::<{}, {}, {}, {}>()",
            a[0].0, a[1].0, a[2].0, a[3].0
        );
    }
    let s: Vec<String> = serialized.iter().map(|v| v.to_string()).collect();
    println!("// proof_data ({} elements):", s.len());
    for chunk in s.chunks(12) {
        println!("    {},", chunk.join(", "));
    }
}

/// Generates serialized FRI proof data for Cairo verifier tests with fold_step > 1.
///
/// Run with:
///   cargo test --features prover -p stwo --test gen_cairo_test_data -- --nocapture
#[test]
fn gen_cairo_fri_test_data() {
    println!("\n=== fold_step=2, log_degree=6 ===");
    generate(6, 2, vec![5, 13, 29]);

    println!("\n=== fold_step=3, log_degree=7 ===");
    generate(7, 3, vec![5, 13, 29]);
}
