#[allow(clippy::modulo_one)]
#[allow(clippy::identity_op)]
#[allow(clippy::uninit_vec)]
#[allow(dead_code)]
pub mod bit_unpack_code;
pub mod bit_unpack_constraint_code;
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
    use num_traits::Zero;

    use self::bit_unpack_code::compute;
    use crate::core::fields::m31::M31;
    use crate::core::fields::qm31::QM31;
    use crate::core::fields::Field;

    let two = M31::from_u32_unchecked(2);
    let random_element = QM31::from([354897, 3212, 129, 5]);
    let values: Vec<u16> = (0_u16..64).map(|x| x.pow(2) + 3).collect();
    let out = compute(&bit_unpack_code::Input {
        values: values.clone(),
        random_element: vec![random_element; 1],
    });
    for (i, value) in values.iter().cloned().enumerate() {
        let mut curr = M31::zero();
        for j in 0..15 {
            let bit_j = out.unpacked[i * 16 + j] - two * out.unpacked[i * 16 + j + 1];
            assert_eq!(bit_j.pow(2), bit_j);
            curr += (bit_j) * two.pow((j as u32).into());
        }
        assert_eq!(curr, value.into());
    }

    let mut curr_shifted_sum = QM31::zero();
    for (i, out_value) in out.unpacked.iter().enumerate() {
        curr_shifted_sum += (QM31::from(M31::from(*out_value)) - random_element).inverse();
        assert_eq!(
            curr_shifted_sum * out.rc_logup_denom[i],
            out.rc_logup_num[i]
        )
    }
}
