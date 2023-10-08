use super::fields::m31::Field;

pub fn butterfly(v0: &mut Field, v1: &mut Field, twid: Field) {
    let tmp = *v1 * twid;
    *v1 = *v0 - tmp;
    *v0 += tmp;
}
pub fn ibutterfly(v0: &mut Field, v1: &mut Field, itwid: Field) {
    let tmp = *v0;
    *v0 = tmp + *v1;
    *v1 = (tmp - *v1) * itwid;
}
