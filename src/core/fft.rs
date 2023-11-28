use super::fields::m31::BaseField;
use crate::core::fields::Field;

pub fn butterfly<F: Field>(v0: &mut F, v1: &mut F, twid: BaseField) {
    let tmp = *v1 * twid;
    *v1 = *v0 - tmp;
    *v0 += tmp;
}

pub fn ibutterfly<F: Field>(v0: &mut F, v1: &mut F, itwid: BaseField) {
    let tmp = *v0;
    *v0 = tmp + *v1;
    *v1 = (tmp - *v1) * itwid;
}

/// Maps from the x coordinate of a point on the circle to the x coordinate of its double.
pub fn psi_x<F: Field>(x: F) -> F {
    x.square().double() - F::one()
}
