#[cfg(test)]
macro_rules! secure_col {
    ($a:expr, $b:expr, $c:expr, $d:expr) => {
        crate::expr::ExtExpr::secure_col([
            $a.into(),
            $b.into(),
            $c.into(),
            $d.into(),
        ])
    };
}
#[cfg(test)]
pub(crate) use secure_col;

#[cfg(test)]
macro_rules! col {
    ($interaction:expr, $idx:expr, $offset:expr) => {
        crate::expr::BaseExpr::col(($interaction, $idx, $offset).into())
    };
}
#[cfg(test)]
pub(crate) use col;

#[cfg(test)]
macro_rules! var {
    ($var:expr) => {
        crate::expr::BaseExpr::param($var)
    };
}
#[cfg(test)]
pub(crate) use var;

#[cfg(test)]
macro_rules! qvar {
    ($var:expr) => {
        crate::expr::ExtExpr::param($var)
    };
}
#[cfg(test)]
pub(crate) use qvar;

#[cfg(test)]
macro_rules! felt {
    ($val:expr) => {
        crate::expr::BaseExpr::from(stwo::core::fields::m31::BaseField::from($val))
    };
}
#[cfg(test)]
pub(crate) use felt;

#[cfg(test)]
macro_rules! qfelt {
    ($a:expr, $b:expr, $c:expr, $d:expr) => {
        crate::expr::ExtExpr::from(stwo::core::fields::qm31::SecureField::from_m31_array([
            $a.into(),
            $b.into(),
            $c.into(),
            $d.into(),
        ]))
    };
}
#[cfg(test)]
pub(crate) use qfelt;
