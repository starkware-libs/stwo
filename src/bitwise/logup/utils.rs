const DILUTED_BITS: usize = 15;

pub fn dilute(x: u32) -> u32 {
    (0..DILUTED_BITS)
        .into_iter()
        .map(|i| (x & 2u32.pow(i as u32)) << i)
        .fold(0, |acc, x| acc | x)
}

pub fn undilute(_x: u32) -> u32 {
    unimplemented!()
}

pub fn gen_diluted_numbers(_n_bits: usize) -> Vec<u32> {
    unimplemented!()
}

#[cfg(test)]
mod tests {
    use super::{gen_diluted_numbers, DILUTED_BITS};
    use crate::bitwise::logup::utils::dilute;

    #[test]
    fn test_diluted() {
        let diluted_numbers = gen_diluted_numbers(DILUTED_BITS);
        let x = 17;
        // println!("{:?}", diluted_numbers);
        assert_eq!(dilute(x), diluted_numbers[x as usize])
    }
}
