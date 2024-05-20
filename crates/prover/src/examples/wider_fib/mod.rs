pub mod trace;

#[cfg(test)]
mod tests {
    use num_traits::Zero;

    use crate::{core::fields::m31::BaseField, m31};

    use super::trace::{write_trace_row, Input};


    #[test]
    fn test_trace_row_constraints() {
        let inputs = (2..10).map(|i| Input::new(m31!(i))).collect::<Vec<Input>>();
        let mut trace: Vec<Vec<BaseField>> = vec![vec![BaseField::zero(); 8]; inputs.len()];
        inputs.iter().enumerate().for_each(|(i, input)| {
            write_trace_row(&mut trace, input, i);
        });
        for i in 0..8 {
            println!("trace[{}]: {:?}", i, trace[i][0]);
        }
    }
}
