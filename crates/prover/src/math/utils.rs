/// Returns s, t, g such that g = gcd(x,y),  sx + ty = g.
pub fn egcd(x: isize, y: isize) -> (isize, isize, isize) {
    if x == 0 {
        return (0, 1, y);
    }
    let k = y / x;
    let (s, t, g) = egcd(y % x, x);
    (t - s * k, s, g)
}

#[cfg(test)]
mod tests {
    use crate::math::utils::egcd;

    #[test]
    fn test_egcd() {
        let pairs = [(17, 5, 1), (6, 4, 2), (7, 7, 7)];
        for (x, y, res) in pairs.into_iter() {
            let (a, b, gcd) = egcd(x, y);
            assert_eq!(gcd, res);
            assert_eq!(a * x + b * y, gcd);
        }
    }
}
