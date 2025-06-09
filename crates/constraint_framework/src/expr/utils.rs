#[cfg(test)]
macro_rules! secure_col {
    ($a:expr, $b:expr, $c:expr, $d:expr) => {
        crate::expr::ExtExpr::SecureCol([
            Box::new($a.into()),
            Box::new($b.into()),
            Box::new($c.into()),
            Box::new($d.into()),
        ])
    };
}
#[cfg(test)]
pub(crate) use secure_col;

#[cfg(test)]
macro_rules! col {
    ($interaction:expr, $idx:expr, $offset:expr) => {
        crate::expr::BaseExpr::Col(($interaction, $idx, $offset).into())
    };
}
#[cfg(test)]
pub(crate) use col;

#[cfg(test)]
macro_rules! var {
    ($var:expr) => {
        crate::expr::BaseExpr::Param($var.to_string())
    };
}
#[cfg(test)]
pub(crate) use var;

#[cfg(test)]
macro_rules! qvar {
    ($var:expr) => {
        crate::expr::ExtExpr::Param($var.to_string())
    };
}
#[cfg(test)]
pub(crate) use qvar;

#[cfg(test)]
macro_rules! felt {
    ($val:expr) => {
        crate::expr::BaseExpr::Const($val.into())
    };
}
#[cfg(test)]
pub(crate) use felt;

#[cfg(test)]
macro_rules! qfelt {
    ($a:expr, $b:expr, $c:expr, $d:expr) => {
        crate::expr::ExtExpr::Const(
            stwo_prover::core::fields::qm31::SecureField::from_m31_array([
                $a.into(),
                $b.into(),
                $c.into(),
                $d.into(),
            ]),
        )
    };
}
#[cfg(test)]
pub(crate) use qfelt;
