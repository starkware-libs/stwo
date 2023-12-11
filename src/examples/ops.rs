use std::ops::{Add, Mul, Sub};

pub fn const_val<T: From<S>, S>(x: S) -> T {
    T::from(x)
}

pub fn mul<T: Mul<T, Output = T>>(x: T, y: T) -> T {
    x * y
}
pub fn add<T: Add<T, Output = T>>(x: T, y: T) -> T {
    x + y
}
#[allow(dead_code)]
pub fn sub<T: Sub<T, Output = T>>(x: T, y: T) -> T {
    x - y
}
