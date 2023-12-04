/// Repeats each value sequentially `duplicity` many times.
///
/// # Examples
///
/// ```rust,ignore
/// assert_eq!(repeat_value(&[1, 2, 3], 2), vec![1, 1, 2, 2, 3, 3]);
/// ```
pub(super) fn repeat_value<T: Copy>(values: &[T], duplicity: usize) -> Vec<T> {
    let n = values.len();
    let mut res: Vec<T> = Vec::with_capacity(n * duplicity);

    // Fill each chunk with its corresponding value.
    for &v in values {
        for _ in 0..duplicity {
            res.push(v)
        }
    }

    res
}

#[cfg(test)]
mod tests {
    use super::repeat_value;

    #[test]
    fn repeat_value_0_times_works() {
        assert!(repeat_value(&[1, 2, 3], 0).is_empty());
    }

    #[test]
    fn repeat_value_2_times_works() {
        assert_eq!(repeat_value(&[1, 2, 3], 2), vec![1, 1, 2, 2, 3, 3]);
    }

    #[test]
    fn repeat_value_3_times_works() {
        assert_eq!(repeat_value(&[1, 2], 3), vec![1, 1, 1, 2, 2, 2]);
    }
}
