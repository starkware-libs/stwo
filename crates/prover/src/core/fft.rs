use std::ops::{Add, AddAssign, Mul, Sub};

use super::fields::m31::BaseField;

pub fn butterfly<F>(v0: &mut F, v1: &mut F, twid: BaseField)
where
    F: AddAssign<F> + Sub<F, Output = F> + Mul<BaseField, Output = F> + Copy,
{
    let tmp = *v1 * twid;
    *v1 = *v0 - tmp;
    *v0 += tmp;
}

pub fn ibutterfly<F>(v0: &mut F, v1: &mut F, itwid: BaseField)
where
    F: AddAssign<F> + Add<F, Output = F> + Sub<F, Output = F> + Mul<BaseField, Output = F> + Copy,
{
    let tmp = *v0;
    *v0 = tmp + *v1;
    *v1 = (tmp - *v1) * itwid;
}
