pub trait MontgomeryField {
    fn to_montgomery(&self) -> Self;
    fn from_montgomery(&self) -> Self;
    fn montgomery_mul(&self, rhs: &Self) -> Self;
    fn unreduced_montgomery_mul(&self, rhs: &Self) -> u32;
}
