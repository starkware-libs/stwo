use super::field::M31;
use super::field::P;
use rand::Rng;

fn mul_p(a: u32, b: u32) -> u32 {
    ((a as u64 * b as u64) % P as u64) as u32
}

fn add_p(a: u32, b: u32) -> u32 {
    (a + b) % P
}

fn sub_p(a: u32, b: u32) -> u32 {
    if a > b {
        a - b
    } else {
        a + P - b
    }
}


#[test]
fn test_ops() {
    let mut rng = rand::thread_rng();
    for _ in 0..(10000) {
        let x: u32 = rng.gen::<u32>() % P;
        let y: u32 = rng.gen::<u32>() % P;
        assert_eq!(
            M31::from_u32_unchecked(add_p(x, y)),
            M31::from_u32_unchecked(x) + M31::from_u32_unchecked(y)
        );
        assert_eq!(
            M31::from_u32_unchecked(mul_p(x, y)),
            M31::from_u32_unchecked(x) * M31::from_u32_unchecked(y)
        );
        assert_eq!(
            M31::from_u32_unchecked(sub_p(x, y)),
            M31::from_u32_unchecked(x) - M31::from_u32_unchecked(y)
        );
    }
}
