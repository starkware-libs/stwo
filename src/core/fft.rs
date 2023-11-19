use num_traits::One;

use super::fields::m31::BaseField;

pub fn butterfly(v0: &mut BaseField, v1: &mut BaseField, twid: BaseField) {
    let tmp = *v1 * twid;
    *v1 = *v0 - tmp;
    *v0 += tmp;
}

pub fn ibutterfly(v0: &mut BaseField, v1: &mut BaseField, itwid: BaseField) {
    let tmp = *v0;
    *v0 = tmp + *v1;
    *v1 = (tmp - *v1) * itwid;
}

/// Maps from the x coordinate of a point on the circle to the x coordinate of its double.
pub fn psi_x(x: BaseField) -> BaseField {
    x.square().double() - BaseField::one()
}
