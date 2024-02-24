#![feature(array_chunks)]

use std::iter::zip;
use std::time::Instant;

use gkr2::gkr::{partially_verify, prove, MleLayer};
use gkr2::utils::Fraction;
use prover_research::commitment_scheme::blake2_hash::Blake2sHasher;
use prover_research::commitment_scheme::hasher::Hasher;
use prover_research::core::channel::{Blake2sChannel, Channel};
use prover_research::core::fields::qm31::SecureField;

fn main() {
    const N: usize = 1 << 21;

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
        random_fractions = random_fractions
            .array_chunks()
            .map(|&[a, b]| a + b)
            .collect();
    }
    layers.reverse();

    println!("layer gen time: {:?}", now.elapsed());

    // println!("yo: {}" )

    let now = Instant::now();
    let proof = prove(&mut test_channel(), layers);
    println!("proof gen time: {:?}", now.elapsed());

    // let (assignment, p3_claim, q3_claim) =
    //     partially_verify(&proof, &mut test_channel()).unwrap();
    let now = Instant::now();
    let res = partially_verify(&proof, &mut test_channel());
    println!("verify time: {:?}", now.elapsed());
    assert!(res.is_ok());
}

fn test_channel() -> Blake2sChannel {
    let seed = Blake2sHasher::hash(&[]);
    Blake2sChannel::new(seed)
}
