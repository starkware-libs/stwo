#[allow(clippy::modulo_one)]
#[allow(clippy::identity_op)]
#[allow(clippy::uninit_vec)]
#[allow(dead_code)]
pub mod fibonacci_code;
mod ops;

#[test]
fn test_fibonacci_generation() {
    use self::fibonacci_code::compute;
    use crate::core::fields::m31::M31;
    let out = compute(&fibonacci_code::Input {
        secret: vec![M31::from_u32_unchecked(1)],
    });
    println!("{:?}", out.f);
}
