#[allow(clippy::modulo_one)]
#[allow(clippy::identity_op)]
#[allow(clippy::uninit_vec)]
#[allow(dead_code)]
pub mod bit_unpack_code;
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

#[test]
fn test_bit_unpack_generation() {
    use crate::core::fields::Field;
    use num_traits::Zero;
    use self::bit_unpack_code::compute;
    use crate::core::fields::m31::M31;
    let two = M31::from_u32_unchecked(2);
    let values: Vec<u16> = (0_u16..64).map(|x| x.pow(2) + 3).collect();
    let out = compute(&bit_unpack_code::Input {
        values: values.clone(),
    });
    for (i, value) in values.into_iter().enumerate() {
        let mut curr = M31::zero();
        for j in 0..15 {
            curr += (out.unpacked[i * 16 + j] - two * out.unpacked[i * 16 + j + 1])
                * two.pow((j as u32).into());
        }
        assert_eq!(curr, value.into());
    }
}
